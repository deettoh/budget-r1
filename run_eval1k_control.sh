#!/bin/bash
#SBATCH --job-name=ev1k_ctrl
#SBATCH --gres=gpu:1
#SBATCH --output=logs/output_%j.txt
#SBATCH --error=logs/error_%j.txt

# 1024-question greedy on control, actor/best is already f1-best

mkdir -p logs
mkdir -p outputs

python -m verl.trainer.main_ppo data.train_files=data/thesis_rl_v7_control/train.parquet data.val_files=data/thesis_rl_v7_control/test.parquet data.train_batch_size=64 data.val_batch_size=64 data.val_data_num=1024 data.max_prompt_length=4096 data.max_response_length=256 data.max_start_length=2048 data.max_obs_length=500 algorithm.adv_estimator=grpo algorithm.no_think_rl=false actor_rollout_ref.model.path=models/sr1-3b-it actor_rollout_ref.model.lora.enabled=true actor_rollout_ref.model.lora.quant_4bit=true actor_rollout_ref.model.lora.adapter_path=verl_checkpoints/control_sft/actor/best actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.actor.ppo_mini_batch_size=64 actor_rollout_ref.actor.ppo_micro_batch_size=1 actor_rollout_ref.actor.state_masking=true actor_rollout_ref.actor.log_prob_chunk_size=512 actor_rollout_ref.ref.log_prob_chunk_size=512 actor_rollout_ref.actor.fsdp_config.optimizer_offload=true actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.temperature=1 actor_rollout_ref.rollout.n_agent=1 actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.gpu_memory_utilization=0.4 actor_rollout_ref.rollout.log_prob_micro_batch_size=1 actor_rollout_ref.ref.log_prob_micro_batch_size=1 actor_rollout_ref.ref.fsdp_config.param_offload=True retriever.url=local retriever.faiss_gpu=False retriever.topk=3 retriever.index_path=retrieval_data/e5_IVF.index retriever.corpus_path=retrieval_data/wiki-18.jsonl retriever.model_path=intfloat/e5-base-v2 budget_planner.enabled=false cost_reward.enabled=false max_turns=8 trainer.logger=['console'] +trainer.val_only=true +trainer.val_before_train=true trainer.dump_val_text=true trainer.default_hdfs_dir=null trainer.n_gpus_per_node=1 trainer.nnodes=1 trainer.project_name=FYP trainer.experiment_name=eval1k_control_greedy trainer.default_local_dir=verl_checkpoints/eval1k_control_greedy
