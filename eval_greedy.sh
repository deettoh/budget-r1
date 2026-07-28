#!/bin/bash
#SBATCH --job-name=eval_greedy
#SBATCH --gres=gpu:1
#SBATCH --output=logs/output_%j.txt
#SBATCH --error=logs/error_%j.txt

# Greedy evaluation of one checkpoint. Writes per-question records to
# outputs/premise_eval_<experiment_name>.json for the analysis scripts.
#
# Configure per condition: lora.adapter_path (actor/best, or drop it to
# evaluate the frozen base), data.val_files and budget_planner.enabled
# to match the arm, and a unique experiment_name.
#
# dump_val_text=true is required, it writes the extra_info.index that
# pairing depends on.
#
# val_data_num must be a multiple of val_batch_size, the loader drops
# the final partial batch so 1000 at batch 64 evaluates 960.
#
# Set trainer.val_do_sample=true for the sampled calibration pass.

mkdir -p logs
mkdir -p outputs

python -m verl.trainer.main_ppo \
    data.train_files=data/thesis_rl_budgetfirst/train.parquet \
    data.val_files=data/thesis_rl_budgetfirst/test.parquet \
    data.train_batch_size=64 \
    data.val_batch_size=64 \
    data.val_data_num=1024 \
    data.max_prompt_length=4096 \
    data.max_response_length=256 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=models/sr1-3b-it \
    actor_rollout_ref.model.lora.enabled=true \
    actor_rollout_ref.model.lora.quant_4bit=true \
    actor_rollout_ref.model.lora.adapter_path=verl_checkpoints/treatment_budget/actor/best \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=1 \
    actor_rollout_ref.actor.state_masking=true \
    actor_rollout_ref.actor.log_prob_chunk_size=512 \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    retriever.url=local \
    retriever.faiss_gpu=False \
    retriever.topk=3 \
    retriever.index_path=retrieval_data/e5_IVF.index \
    retriever.corpus_path=retrieval_data/wiki-18.jsonl \
    retriever.model_path=intfloat/e5-base-v2 \
    budget_planner.enabled=true \
    budget_planner.max_budget=5 \
    budget_planner.forced_exec.enabled=false \
    cost_reward.enabled=true \
    cost_reward.alpha=0.05 \
    cost_reward.beta=0.0001 \
    cost_reward.gamma=0.02 \
    cost_reward.delta=0.06 \
    cost_reward.answer_metric=em \
    cost_reward.cost_in_advantage=0.5 \
    cost_reward.grounding.enabled=true \
    cost_reward.grounding.lam=0.5 \
    max_turns=8 \
    trainer.logger=['console'] \
    +trainer.val_only=true \
    +trainer.val_before_train=true \
    trainer.dump_val_text=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.project_name=FYP \
    trainer.experiment_name=eval_budget_greedy \
    trainer.default_local_dir=verl_checkpoints/eval_budget_greedy
