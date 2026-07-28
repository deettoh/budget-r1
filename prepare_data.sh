#!/bin/bash
#SBATCH --job-name=prep_data
#SBATCH --gres=gpu:1
#SBATCH --output=logs/output_%j.txt
#SBATCH --error=logs/error_%j.txt

# Builds both RL parquets from the same builder and seed, so the arms
# differ only in the prompt template and in whether gold_budget is set.
#
# --budget_template also accepts think_first, minimal, reason_first and
# soft. --max_train_per_budget caps each gold-budget stratum.
#
# CPU only, runs outside Slurm as `bash prepare_data.sh`.

mkdir -p logs
mkdir -p data

# budget-aware arm, declares <budget>k</budget> before reasoning
python scripts/data_process/thesis_qa.py \
    --mode rl \
    --require_budget \
    --budget_template budget_first \
    --local_dir data/thesis_rl_budgetfirst \
    --data_sources hotpotqa,2wikimultihopqa,musique \
    --max_train_per_budget 10000 \
    --seed 42

# control arm, native Search-R1 prompt, no budget declaration
python scripts/data_process/thesis_qa.py \
    --mode rl \
    --local_dir data/thesis_rl_control \
    --data_sources hotpotqa,2wikimultihopqa,musique \
    --max_train_per_budget 10000 \
    --seed 42

# guards the grounding reward, fails loudly if gold titles are missing
python scripts/verify_gold_titles.py --parquet data/thesis_rl_budgetfirst/train.parquet

python scripts/verify_gold_titles.py --parquet data/thesis_rl_budgetfirst/test.parquet
