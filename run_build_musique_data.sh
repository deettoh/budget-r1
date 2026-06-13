#!/bin/bash
#SBATCH --job-name=bd_musique
#SBATCH --gres=gpu:1
#SBATCH --output=logs/output_%j.txt
#SBATCH --error=logs/error_%j.txt

# musique ood eval parquets, agentic then no-rag and naive-rag rewrites

mkdir -p logs
mkdir -p data/thesis_rl_musique
mkdir -p data/thesis_norag_musique
mkdir -p data/thesis_naiverag_musique

python scripts/data_process/thesis_qa.py --mode rl --local_dir data/thesis_rl_musique --data_sources musique --train_split dev --val_split dev

python scripts/build_baseline_data.py --mode norag --src data/thesis_rl_musique/test.parquet --out data/thesis_norag_musique/test.parquet

python scripts/build_baseline_data.py --mode naiverag --src data/thesis_rl_musique/test.parquet --out data/thesis_naiverag_musique/test.parquet --nprobe 512
