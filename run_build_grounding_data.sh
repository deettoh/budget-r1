#!/bin/bash
#SBATCH --job-name=build_ground
#SBATCH --gres=gpu:1
#SBATCH --output=logs/output_%j.txt
#SBATCH --error=logs/error_%j.txt

# add gold_titles for grounding, additive so other runs are safe

mkdir -p logs
mkdir -p data/thesis_rl_musique_budget

python scripts/data_process/thesis_qa.py --mode rl --require_budget --local_dir data/thesis_rl_musique_budget --data_sources hotpotqa,2wikimultihopqa,musique

python scripts/verify_gold_titles.py --parquet data/thesis_rl_musique_budget/test.parquet
