#!/bin/bash
#SBATCH --job-name=build_v7_data
#SBATCH --gres=gpu:1
#SBATCH --output=logs/output_%j.txt
#SBATCH --error=logs/error_%j.txt

# v6 data with the think-first prompt, declaration still precedes search

mkdir -p logs
mkdir -p data/thesis_rl_v7_budget

python scripts/data_process/thesis_qa.py --mode rl --require_budget --local_dir data/thesis_rl_v7_budget --data_sources hotpotqa,2wikimultihopqa,musique --max_train_per_budget 10000 --seed 42

python scripts/verify_gold_titles.py --parquet data/thesis_rl_v7_budget/test.parquet

python scripts/verify_gold_titles.py --parquet data/thesis_rl_v7_budget/train.parquet
