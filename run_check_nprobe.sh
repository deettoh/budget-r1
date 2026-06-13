#!/bin/bash
#SBATCH --job-name=chk_nprobe
#SBATCH --gres=gpu:1
#SBATCH --output=logs/output_%j.txt
#SBATCH --error=logs/error_%j.txt

# read-only, report the nprobe persisted in the ivf index

mkdir -p logs

python scripts/set_nprobe.py --index retrieval_data/e5_IVF.index --nprobe 512 --dry_run
