#!/bin/bash
#SBATCH --job-name=gold_recall
#SBATCH --gres=gpu:1
#SBATCH --output=logs/output_%j.txt
#SBATCH --error=logs/error_%j.txt

# offline gold-recall attribution, splits unreachable from low-ranked

mkdir -p logs
mkdir -p outputs

python scripts/gold_recall_probe.py --num 200 --topks 3,5,10,20 --oracle_query title --index_path retrieval_data/e5_IVF.index --corpus_path retrieval_data/wiki-18.jsonl --retriever_name e5 --retriever_model intfloat/e5-base-v2 --out outputs/gold_recall_sweep_title.json

python scripts/gold_recall_probe.py --num 200 --topks 3,5,10,20 --oracle_query passage --index_path retrieval_data/e5_IVF.index --corpus_path retrieval_data/wiki-18.jsonl --retriever_name e5 --retriever_model intfloat/e5-base-v2 --out outputs/gold_recall_sweep_passage.json
