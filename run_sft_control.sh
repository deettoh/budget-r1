#!/bin/bash
#SBATCH --job-name=sft_control
#SBATCH --gres=gpu:1
#SBATCH --output=logs/output_%j.txt
#SBATCH --error=logs/error_%j.txt

# symmetric native-prompt sft so the primary contrast stays clean

mkdir -p logs
mkdir -p verl_checkpoints

python -m verl.trainer.fsdp_sft_trainer data.train_files=data/sft_native/train.parquet data.val_files=data/sft_native/test.parquet data.prompt_key=prompt data.response_key=response data.max_length=4096 data.truncation=right data.train_batch_size=32 data.micro_batch_size=1 data.mask_information_spans=true model.partial_pretrain=models/sr1-3b-it model.load_dtype=bfloat16 model.enable_gradient_checkpointing=true model.lora_rank=64 model.lora_alpha=128 'model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]' model.fsdp_config.cpu_offload=true model.fsdp_config.offload_params=true optim.lr=1e-4 trainer.default_local_dir=verl_checkpoints/sft_control trainer.default_hdfs_dir=null trainer.project_name=FYP trainer.experiment_name=sft_control trainer.total_epochs=2 trainer.validate_before_training=true trainer.logger=['console']
