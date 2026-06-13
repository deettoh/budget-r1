#!/bin/bash
#SBATCH --job-name=set_nprobe
#SBATCH --gres=gpu:1
#SBATCH --output=logs/output_%j.txt
#SBATCH --error=logs/error_%j.txt

# freeze nprobe 512 so every condition retrieves at one operating point

mkdir -p logs

python scripts/set_nprobe.py --index retrieval_data/e5_IVF.index --nprobe 512
