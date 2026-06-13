#!/bin/bash
#SBATCH --job-name=bd_norag
#SBATCH --gres=gpu:1
#SBATCH --output=logs/output_%j.txt
#SBATCH --error=logs/error_%j.txt

# Build the no-RAG closed-book eval parquet from the existing test set.

mkdir -p logs
mkdir -p data/thesis_norag

python scripts/build_baseline_data.py --mode norag --src data/thesis_rl/test.parquet --out data/thesis_norag/test.parquet
