#!/bin/bash
#SBATCH --job-name=gold_recall
#SBATCH --gres=gpu:1
#SBATCH --output=logs/output_%j.txt
#SBATCH --error=logs/error_%j.txt

# Offline gold-passage recall attribution (no training, no rollout).
# Answers WHY budget-planning collapses to k=1 across 6679/6681/6682:
# is the second hop's gold passage even retrievable?
#   Tier A (oracle title query) low  -> corpus/index-limited
#   Tier A high, Tier B (question) low -> query/reasoning-limited
#   both high                          -> generation-limited (3B)
# E5 + wiki-18 IVF index, top-3, dev split, 200 q/dataset.

mkdir -p logs
mkdir -p outputs

python scripts/gold_recall_probe.py --num 200 --topk 3 --index_path retrieval_data/e5_IVF.index --corpus_path retrieval_data/wiki-18.jsonl --retriever_name e5 --retriever_model intfloat/e5-base-v2 --out outputs/gold_recall_probe.json
