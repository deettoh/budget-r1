#!/bin/bash
#SBATCH --job-name=bd_naive
#SBATCH --gres=gpu:1
#SBATCH --output=logs/output_%j.txt
#SBATCH --error=logs/error_%j.txt

# naive-rag parquet, top-3 at frozen nprobe 512, read then answer

mkdir -p logs
mkdir -p data/thesis_naiverag

python scripts/build_baseline_data.py --mode naiverag --src data/thesis_rl/test.parquet --out data/thesis_naiverag/test.parquet --nprobe 512
