#!/bin/bash
#SBATCH --job-name=control_lora
#SBATCH --gres=gpu:1
#SBATCH --output=logs/output_%j.txt
#SBATCH --error=logs/error_%j.txt

# Option C Control: binary-EM QLoRA on the released Search-R1 3B-it base
# (budget + cost OFF). 4-bit nf4 base (quant_4bit) clears the 24gb actor
# backward OOM; batch=64 + chunk=512 keep the step ~35-40min so ~40 steps
# fit one ~30h wall. val_before_train + test/save every 10 steps capture
# the curve and best-by-f1 checkpoint. Treatment (run.sh) must mirror this
# exact config with budget/cost ON.

mkdir -p logs
mkdir -p verl_checkpoints

python -m verl.trainer.main_ppo data.train_files=data/thesis_rl/train.parquet data.val_files=data/thesis_rl/test.parquet data.train_batch_size=64 data.val_batch_size=64 data.val_data_num=256 data.max_prompt_length=4096 data.max_response_length=256 data.max_start_length=2048 data.max_obs_length=500 data.shuffle_train_dataloader=True algorithm.adv_estimator=grpo algorithm.no_think_rl=false actor_rollout_ref.model.path=models/sr1-3b-it actor_rollout_ref.model.lora.enabled=true actor_rollout_ref.model.lora.quant_4bit=true actor_rollout_ref.model.enable_gradient_checkpointing=true actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.actor.optim.lr=1e-5 actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 actor_rollout_ref.actor.use_kl_loss=true actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.actor.ppo_mini_batch_size=64 actor_rollout_ref.actor.ppo_micro_batch_size=1 actor_rollout_ref.actor.state_masking=true actor_rollout_ref.actor.log_prob_chunk_size=512 actor_rollout_ref.actor.fsdp_config.param_offload=false actor_rollout_ref.actor.fsdp_config.grad_offload=false actor_rollout_ref.actor.fsdp_config.optimizer_offload=true actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.temperature=1 actor_rollout_ref.rollout.n_agent=5 actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.gpu_memory_utilization=0.5 actor_rollout_ref.rollout.log_prob_micro_batch_size=1 actor_rollout_ref.ref.log_prob_micro_batch_size=1 actor_rollout_ref.ref.fsdp_config.param_offload=True retriever.url=local retriever.faiss_gpu=False retriever.topk=3 retriever.index_path=retrieval_data/e5_IVF.index retriever.corpus_path=retrieval_data/wiki-18.jsonl retriever.model_path=intfloat/e5-base-v2 budget_planner.enabled=false cost_reward.enabled=false max_turns=8 trainer.logger=['console','wandb'] +trainer.val_only=false +trainer.val_before_train=true trainer.default_hdfs_dir=null trainer.n_gpus_per_node=1 trainer.nnodes=1 trainer.save_freq=10 trainer.test_freq=10 trainer.project_name=FYP trainer.experiment_name=control_lora trainer.total_epochs=15 trainer.total_training_steps=40 trainer.default_local_dir=verl_checkpoints/control_lora
