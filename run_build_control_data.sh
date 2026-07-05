#!/bin/bash
#SBATCH --job-name=build_ctrl_data
#SBATCH --gres=gpu:1
#SBATCH --output=logs/output_%j.txt
#SBATCH --error=logs/error_%j.txt

# control-prompt parquet, same question mix and cap as v7

mkdir -p logs
mkdir -p data/thesis_rl_v7_control

python scripts/data_process/thesis_qa.py --mode rl --local_dir data/thesis_rl_v7_control --data_sources hotpotqa,2wikimultihopqa,musique --max_train_per_budget 10000 --seed 42

python scripts/verify_gold_titles.py --parquet data/thesis_rl_v7_control/test.parquet

python scripts/verify_gold_titles.py --parquet data/thesis_rl_v7_control/train.parquet
