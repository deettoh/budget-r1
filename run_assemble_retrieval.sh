#!/bin/bash
#SBATCH --job-name=searchr1-assemble
#SBATCH --gres=gpu:1
#SBATCH --output=logs/output_%j.txt
#SBATCH --error=logs/error_%j.txt

# call python directly, this cluster forbids srun inside sbatch

mkdir -p logs

python scripts/assemble_retrieval.py --data_dir retrieval_data
