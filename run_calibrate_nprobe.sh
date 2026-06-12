#!/bin/bash
#SBATCH --job-name=nprobe_calib
#SBATCH --gres=gpu:1
#SBATCH --output=logs/output_%j.txt
#SBATCH --error=logs/error_%j.txt

# ivfpq nprobe calibration, nprobe 4096 is the recall ceiling

mkdir -p logs
mkdir -p outputs

python scripts/calibrate_nprobe.py --ivf_index retrieval_data/e5_IVF.index --queries data/thesis_rl/test.parquet --retriever_name e5 --retriever_model intfloat/e5-base-v2 --topk 3 --num_queries 2000 --out outputs/nprobe_calibration.json
