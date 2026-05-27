#!/bin/bash
#SBATCH --job-name=searchr1-data
#SBATCH --gres=gpu:1
#SBATCH --output=logs/output_%j.txt
#SBATCH --error=logs/error_%j.txt

# server-side data prep, builds the thesis_rl parquets (CPU-only)

mkdir -p logs
mkdir -p data

srun python3 scripts/data_process/thesis_qa.py \
    --mode rl \
    --local_dir data/thesis_rl \
    --data_sources hotpotqa,2wikimultihopqa \
    --train_split train \
    --val_split dev

srun python3 scripts/data_process/thesis_qa.py \
    --mode rl \
    --require_budget \
    --local_dir data/thesis_rl_budget \
    --data_sources hotpotqa,2wikimultihopqa \
    --train_split train \
    --val_split dev
