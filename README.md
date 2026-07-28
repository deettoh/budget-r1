# Budget-Aware Agentic Retrieval-Augmented Generation: Reinforcement Learning of Self-Declared Search Budgets for Cost-Efficient Multi-hop Question Answering

An agentic retrieval-augmented generation system that learns to plan its own
retrieval budget. The model declares `<budget>k</budget>` as its first action,
the rollout environment enforces k as a hard cap on retrieval calls, and a
cost-aware GRPO objective trains that declaration to track question
difficulty. The work extends [Search-R1](SEARCH_R1_README.md), Jin et al. 2025,
on the vendored [veRL](VERL_README.md) trainer.

This is my final year project for a B.Eng in Artificial Intelligence (WIP).

On 1024 held-out multi-hop questions the system improves answer F1 over the
Search-R1 baseline, .341 against .324, while issuing 7% fewer retrieval calls
and spending 8% fewer tokens.

## Overview

Agentic RAG systems decide for themselves how many times to search. Left
unconstrained they over-retrieve, because a binary correctness reward never
charges for a wasted call. Retrieval dominates inference cost, so that
behaviour is expensive.

Two mechanisms address this.

- Learned budget planning. Before reasoning or searching, the model emits an
  integer k in `[0, 5]`. The rollout loop treats k as a hard cap and blocks any
  search beyond it, which turns an emergent behaviour into a declared,
  enforceable interface. A per-question budget is something a scalar training
  penalty cannot express.
- Cost-aware objective. Retrieval calls are z-scored within each GRPO group and
  enter the advantage, so a rollout is charged for being costlier than its
  siblings on the same question rather than for its absolute call count, and
  genuinely deep questions keep their depth. The scalar reward keeps two
  declaration terms, one for unused declared budget and one for declaring below
  the question's gold budget, plus a grounding term that credits a hop for
  retrieving a gold passage. A gate applies the cost signal only to rollouts
  that earned positive reward, so a group with no successful trajectory
  contributes no cost-only gradient.

## Results

Greedy decoding on 1024 held-out questions drawn from HotpotQA,
2WikiMultiHopQA, and MuSiQue. All three conditions answer the same questions,
so the comparisons are paired.

| Condition | EM | F1 | MRC | Generated | Retrieved | TTC | CES |
|---|---|---|---|---|---|---|---|
| Proposed | **.273** | **.341** | **1.87** | **296** | **925** | **1220** | **.280** |
| Search-R1 baseline, 4-bit | .260 | .324 | 2.00 | 331 | 993 | 1324 | .245 |
| Matched control | .237 | .295 | 3.02 | 497 | 1497 | 1994 | .148 |

MRC is mean retrieval calls per question, TTC is total token cost, and CES is
F1 x 1000 / TTC. Bold marks the best value per column, lower being better for
the four cost columns. The Search-R1 baseline is the published model with no
further training. The matched control is trained on the same questions and
recipe with both mechanisms disabled, so it isolates what the mechanisms add.
It uses the native Search-R1 prompt, since the budget prompt only exists once
the planner is enabled.

Against the Search-R1 baseline, F1 rises from .324 to .341 while retrieval
calls fall 7% and token cost falls 8%. Against the matched control, F1 rises
from .295 to .341 while retrieval calls fall 38% and token cost falls 39%.
Quality improves and cost drops together in both comparisons.

Declared budgets correlate with question difficulty at r = .479 against the
gold budget, measured under sampling on the same 1024 questions. Per dataset
the correlation is .519 on 2WikiMultiHopQA and .519 on MuSiQue. It is undefined
on HotpotQA, whose gold budget is uniformly 2, so a flat declaration there is
the correct behaviour.

Retrieval-free and single-shot baselines, measured separately on 256 questions,
place the agentic setting in context. Closed-book QA reaches F1 .205 and naive
top-3 RAG reaches .215, against .351 for the Search-R1 baseline at the same
quantization. Iterative retrieval, not the reward design, supplies most of the
answer quality.

## Method

Each rollout runs a generate, act, observe loop. The model emits one action per
turn, the response is truncated at the first closing tag, and the action is
parsed and dispatched.

- `<budget>k</budget>` declares the retrieval cap. It must come first when the
  planner is enabled, and any earlier action invalidates the rollout.
- `<search>query</search>` retrieves the top 3 passages, unless the declared
  budget is already exhausted, in which case the call is blocked and counted.
- `<answer>text</answer>` ends the episode.

Retrieved passages are masked out of the policy gradient, a Search-R1
invariant, and the same mask separates generated from retrieved tokens for the
token-cost term.

Both arms of the comparison receive a symmetric frozen-model self-distillation
SFT before RL, differing only in prompt template, so the contrast isolates the
two mechanisms rather than the warm start.

## Repository structure

Two co-located trees. `verl/` is a vendored copy of veRL supplying the GRPO
trainer, workers, and Hydra config system. The files below are the thesis
contribution.

```
Search-R1/
├── search_r1/
│   ├── budgeting.py              reward equation, single source of truth
│   ├── sft_masking.py            SFT loss mask over response spans
│   ├── lora_utils.py             QLoRA adapter resolution
│   ├── resume_utils.py           warm restart from the latest checkpoint
│   └── llm_agent/generation.py   rollout loop, budget parsing and cap
├── scripts/
│   ├── data_process/thesis_qa.py builds RL parquets and gold budgets
│   ├── build_sft_data.py         SFT parquets from frozen-model traces
│   ├── paired_significance.py    paired bootstrap CIs and McNemar
│   ├── build_baseline_data.py    no-RAG and naive-RAG prompt rewrites
│   └── analyze_*.py              per-dump diagnostics
├── verl/trainer/
│   ├── main_ppo.py               reward manager, the join point
│   ├── ppo/ray_trainer.py        validation, metrics, checkpoint selection
│   └── config/ppo_trainer.yaml   thesis config blocks, both off by default
├── tests/                        unit tests for the above
└── run_*.sh                      Slurm submission scripts
```

## Installation

Python 3.9 with a CUDA GPU, following the upstream Search-R1 environment. The
dependency pins in `requirements.txt` are what the trainer was validated
against. Training was run on a single 24 GB card using 4-bit QLoRA.

```bash
conda create -n searchr1 python=3.9
conda activate searchr1
pip install -r requirements.txt
pip install -e .
```

The retrieval server runs in its own Python 3.10 environment and needs a FAISS
index over the 2018 Wikipedia corpus. See the
[upstream README](SEARCH_R1_README.md) for the retriever setup. Data
preparation and the analysis scripts run on CPU and need only `datasets`,
`numpy`, and `pandas`.

## Reproducing

Commands run from this directory. Training targets a Slurm cluster, so each
stage is a submission script.

```bash
# build the RL parquets with the budget-aware prompt
python3 scripts/data_process/thesis_qa.py --mode rl --require_budget \
    --budget_template budget_first --local_dir data/thesis_rl_budgetfirst \
    --data_sources hotpotqa,2wikimultihopqa,musique --seed 42

# SFT cold-start, then GRPO with both mechanisms enabled
sbatch run_sft_budgetfirst.sh
sbatch run_treatment_v7bf_sft.sh

# matched control, same recipe with both mechanisms off
sbatch run_sft_control.sh
sbatch run_control_sft.sh

# greedy evaluation on 1024 questions, per condition
sbatch run_eval1k_proposed.sh
sbatch run_eval1k_frozen.sh
sbatch run_eval1k_control.sh
```

Each evaluation writes per-question records to
`outputs/premise_eval_<experiment>.json`. Compare two of them with

```bash
python3 scripts/paired_significance.py \
    --dump_a outputs/premise_eval_eval1k_proposed_greedy.json \
    --dump_b outputs/premise_eval_eval1k_frozen_greedy.json \
    --label_a Proposed --label_b Baseline
```

## Tests

```bash
python3 -m unittest discover -s tests
```

Reward math, prompt construction, dataset transforms, rollout accounting, and
the analysis scripts are covered. The advantage tests import PyTorch, so on a
machine without it run the subset that does not.

```bash
python3 -m unittest tests.test_budgeting tests.test_qa_metrics \
    tests.test_gold_recall tests.test_sft_masking tests.test_build_sft_data \
    tests.test_thesis_data tests.test_paired_significance \
    tests.test_resume_utils
```

## Experimental setup

Search-R1 3B-Instruct as the base, a 4-bit QLoRA adapter,
E5-base-v2 retrieval over a 2018 Wikipedia dump at top 3, GRPO group size 5,
learning rate 5e-6, KL coefficient 0.01, 8 turns maximum, and 40 training
steps. Fixed across every trained condition

Every condition selects its best checkpoint on validation F1, with ties broken
by the later checkpoint.

## Design notes

- Budget calibration is a property of the sampled declaration distribution.
  Greedy decoding takes the modal declaration, which collapses to 2 on most
  questions and hides the grading, so the correlation above is measured under
  sampling while the headline quality and cost figures are greedy.
- The declared budget is supervised by a cross-entropy term at the digit
  position, not by the scalar reward alone, because a scalar penalty moves 
  the mean declaration but does not grade it per question.
- Token cost is measured and reported but never charged. Once the cost signal
  moves into the advantage, the per-call and per-token reward penalties are
  both zeroed, so only retrieval calls are priced and the token saving is a
  downstream effect of issuing fewer of them.

## Limitations

- One seed per condition, so none of the differences above are separated from
  seed-to-seed variance.
- Checkpoints are selected on the same split that is reported, because FlashRAG
  exposes no public test split with answers for these datasets.
- TTC and CES are not comparable across prompt formats, since the budget prompt
  adds reasoning tokens that enter the token count. Cross-format cost claims
  lead with MRC, which counts retrieval actions and is prompt-invariant.
- Part of the margin over the matched control comes from the control degrading
  through over-retrieval rather than from the proposed system improving.
- Results are for a single 3B model family at 4-bit quantization. The 4-bit base
  costs about .029 F1 against bf16.
