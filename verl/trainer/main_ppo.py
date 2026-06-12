# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
from verl.utils.reward_score import qa_em
from verl.utils.reward_score import qa_metrics
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import re
import numpy as np
from search_r1.budgeting import BudgetRewardConfig, compute_budget_reward


def _select_rm_score_fn(data_source):
    if data_source in [
        "nq",
        "triviaqa",
        "popqa",
        "hotpotqa",
        "2wikimultihopqa",
        "musique",
        "bamboogle",
    ]:
        return qa_em.compute_score_em
    else:
        raise NotImplementedError


class RewardManager:
    """The reward manager."""

    def __init__(
        self, tokenizer, num_examine, format_score=0.0, cost_reward_config=None
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.format_score = format_score
        self.cost_reward_config = cost_reward_config
        self.last_metrics: list[dict] = []

    def reset_metrics(self) -> None:
        """Clear ``last_metrics`` before a fresh batch is scored."""
        self.last_metrics = []

    def _cost_reward_enabled(self) -> bool:
        """Return True iff the cost-penalized reward is on."""
        return bool(
            self.cost_reward_config is not None
            and self.cost_reward_config.get("enabled", False)
        )

    def _budget_reward_config(self) -> BudgetRewardConfig:
        """Build the frozen alpha/beta/gamma config from Hydra."""
        return BudgetRewardConfig(
            alpha=float(self.cost_reward_config.get("alpha", 0.05)),
            beta=float(self.cost_reward_config.get("beta", 0.0001)),
            gamma=float(self.cost_reward_config.get("gamma", 0.01)),
        )

    def _token_costs(self, data_item, prompt_length) -> tuple[int, int]:
        """Return ``(generated, retrieved)`` token counts.

        Generated comes from ``info_mask``, retrieved is the rest of
        ``attention_mask`` in the response region.
        """
        response_attention = data_item.batch["attention_mask"][prompt_length:]
        generated_mask = data_item.batch.get(
            "info_mask", data_item.batch["attention_mask"]
        )[prompt_length:]
        generated_tokens = int(generated_mask.sum().item())
        total_response_tokens = int(response_attention.sum().item())
        retrieved_tokens = max(0, total_response_tokens - generated_tokens)
        return generated_tokens, retrieved_tokens

    def _scalar_batch_value(self, data_item, key, default):
        """Return scalar at ``data_item.batch[key]`` or ``default``."""
        if key not in data_item.batch.keys():
            return default
        value = data_item.batch[key]
        if hasattr(value, "item"):
            return value.item()
        return value

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        already_print_data_sources = {}
        self.reset_metrics()

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][
                :prompt_length
            ].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][
                prompt_length:
            ].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            # select rm_score
            data_source = data_item.non_tensor_batch["data_source"]
            compute_score_fn = _select_rm_score_fn(data_source)

            answer_score = compute_score_fn(
                solution_str=sequences_str,
                ground_truth=ground_truth,
                format_score=self.format_score,
            )

            qa = qa_metrics.compute_qa_metrics(sequences_str, ground_truth)
            generated_tokens, retrieved_tokens = self._token_costs(
                data_item, prompt_length
            )
            valid_search_calls = int(
                self._scalar_batch_value(data_item, "valid_search_count", 0)
            )
            declared_budget_raw = self._scalar_batch_value(
                data_item,
                "declared_budget",
                (self.cost_reward_config or {}).get("max_budget", 5)
                if isinstance(self.cost_reward_config, dict)
                else -1,
            )
            declared_budget = (
                int(declared_budget_raw) if declared_budget_raw is not None else -1
            )

            if self._cost_reward_enabled():
                effective_budget = declared_budget
                if effective_budget < 0:
                    effective_budget = int(self.cost_reward_config.get("max_budget", 5))
                score, _ = compute_budget_reward(
                    answer_score=answer_score,
                    valid_search_calls=valid_search_calls,
                    generated_tokens=generated_tokens,
                    retrieved_tokens=retrieved_tokens,
                    declared_budget=effective_budget,
                    config=self._budget_reward_config(),
                )
            else:
                score = answer_score

            reward_tensor[i, valid_response_length - 1] = score

            self.last_metrics.append(
                {
                    "data_source": str(data_source),
                    "em": float(qa["em"]),
                    "f1": float(qa["f1"]),
                    "has_answer": bool(qa["has_answer"]),
                    "generated_tokens": int(generated_tokens),
                    "retrieved_tokens": int(retrieved_tokens),
                    "valid_search_calls": int(valid_search_calls),
                    "declared_budget": int(declared_budget),
                    "reward": float(score),
                    "answer_score": float(answer_score),
                }
            )

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        return reward_tensor


import ray
import hydra


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    if not ray.is_initialized():
        # stub psutil + drop stale RAY_ADDRESS for the locked-down node
        import os
        import psutil
        psutil.pids = lambda: []
        psutil.Process.parents = lambda self: []
        os.environ.pop("RAY_ADDRESS", None)

        # Ray's accelerator probe listdirs /dev/vfio which HPC blocks
        _orig_listdir = os.listdir
        def _hpc_safe_listdir(path):
            try:
                return _orig_listdir(path)
            except PermissionError:
                p = path.decode(errors="replace") if isinstance(path, bytes) else str(path)
                if p.startswith("/dev") or p.startswith("/proc"):
                    return []
                raise
        os.listdir = _hpc_safe_listdir

        # set env before ray.init, runtime_env= would use bash (blocked)
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
        os.environ.setdefault("NCCL_DEBUG", "WARN")

        # disable Ray's child killer it SIGABRTs reading /proc
        os.environ.setdefault("RAY_kill_child_processes_on_worker_exit", "false")

        # skip Ray's TPU chip probe it listdirs /dev/vfio in the worker
        os.environ.setdefault(
            "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS", "1"
        )

        # surface the sharding-manager mem trace for rollout-sync OOMs
        os.environ.setdefault("VERL_PPO_LOGGING_LEVEL", "INFO")

        # disable TorchInductor gcc JIT-link fails here, eager is equiv
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

        # expandable_segments reuses fragments so vLLM + actor fit 24gb
        os.environ.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
        )

        # this is for local ray cluster.
        # keep Ray's object store off /dev/shm which HPC restricts
        ray.init(
            address="local",
            object_store_memory=4 * 1024 ** 3,
            _plasma_directory="/tmp",
        )

    ray.get(main_task.remote(config))


def _send_training_alert(config, title: str, text: str, level: str) -> None:
    """Send a wandb alert for the run end, never raise.

    Fires only with wandb enabled and a run active. Any wandb-side
    failure is logged and swallowed so it cannot take down training.
    """
    if "wandb" not in config.trainer.logger:
        return
    try:
        import wandb

        if wandb.run is None:
            return
        wandb.alert(title=title, text=text, level=level)
    except Exception as exc:  # best-effort notifier; log and move on
        print(f"[wandb-alert] could not send alert: {exc}")


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf

    pprint(
        OmegaConf.to_container(config, resolve=True)
    )  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # env_class = ENV_CLASS_MAPPING[config.env.name]

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer

    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == "fsdp":
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup

        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == "megatron":
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup

        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == "fsdp":
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == "megatron":
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(
        tokenizer=tokenizer, num_examine=0, cost_reward_config=config.cost_reward
    )

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(
        tokenizer=tokenizer, num_examine=1, cost_reward_config=config.cost_reward
    )

    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec, mapping=mapping
    )
    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
    )
    trainer.init_workers()
    exp_name = config.trainer.experiment_name
    try:
        trainer.fit()
    except Exception as exc:
        _send_training_alert(
            config,
            title=f"Search-R1 training FAILED: {exp_name}",
            text=f"{type(exc).__name__}: {exc}",
            level="ERROR",
        )
        raise
    else:
        _send_training_alert(
            config,
            title=f"Search-R1 training complete: {exp_name}",
            text=(
                f"Finished {config.trainer.total_training_steps} "
                "training steps."
            ),
            level="INFO",
        )


if __name__ == "__main__":
    main()
