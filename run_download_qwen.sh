#!/bin/bash
#SBATCH --job-name=qwen-dl
#SBATCH --gres=gpu:1
#SBATCH --output=logs/output_%j.txt
#SBATCH --error=logs/error_%j.txt

# fetch vanilla qwen2.5-3b-instruct for the no-rag baselines

mkdir -p logs
mkdir -p models

python download_qwen_it.py
