#!/bin/bash
#SBATCH --job-name=nvcost_mq
#SBATCH --gres=gpu:1
#SBATCH --output=logs/output_%j.txt
#SBATCH --error=logs/error_%j.txt

# naive-rag ttc/ces on musique, counts prompt passages for parity

mkdir -p logs

python scripts/naiverag_cost.py --parquet data/thesis_naiverag_musique/test.parquet --eval_json outputs/premise_eval_baseline_naiverag_musique.json --model models/qwen2.5-3b-it
