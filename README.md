# Budget-Aware Agentic Retrieval-Augmented Generation: Reinforcement Learning of Self-Declared Search Budgets for Cost-Efficient Multi-hop Question Answering

A reinforcement learning framework that trains an agentic retrieval-augmented
generation policy to plan its own retrieval budget. The model declares
`<budget>k</budget>` as its first action, the rollout environment enforces `k`
as a hard cap on retrieval calls, and a cost-aware GRPO objective trains that
declaration to track question difficulty. The work extends
[Search-R1](SEARCH_R1_README.md), Jin et al. 2025, on the vendored
[veRL](VERL_README.md) trainer.

This is my final year project for a B.Eng in Artificial Intelligence, and it
remains a work in progress, so the repository will keep changing.

On 1024 held-out multi-hop questions the framework improves answer F1 over the
Search-R1 baseline (*.341* against *.324*) while issuing *7%* fewer retrieval
calls and spending *8%* fewer tokens.

## Overview

An agentic RAG system decides for itself how many times to search, and under a
binary correctness reward that never charges for a wasted call it
over-retrieves. Retrieval dominates inference cost, so that behaviour is
expensive.

This problem is addressed with the following mechanisms:

- Learned budget planning

  Before reasoning or searching, the model emits an integer `k` in `[0, 5]`.
  The rollout loop treats `k` as a hard cap and blocks any search beyond it,
  which turns an emergent behaviour into a declared, enforceable interface. A
  per-question budget is something a scalar training penalty cannot express.

- Cost-aware objective

  Retrieval calls are z-scored within each GRPO group and subtracted from the
  advantage, so a rollout is charged for being costlier than its siblings on
  the same question rather than for its absolute call count, and a question
  that needs more hops is not penalised for taking them. The scalar reward
  keeps two declaration terms, one for unused declared budget and one for
  declaring below the question's gold budget, plus a grounding term that
  credits a hop for retrieving a gold passage. A gate applies the cost signal
  only to rollouts that earned positive reward, so a group with no successful
  trajectory contributes no cost-only gradient.

## Results

All values come from greedy decoding on 1024 held-out questions drawn from
HotpotQA, 2WikiMultiHopQA and MuSiQue, and every condition answers the same
questions, so the comparisons are paired.

| Condition | EM ↑ | F1 ↑ | MRC ↓ | Generated ↓ | Retrieved ↓ | TTC ↓ | CES ↑ |
|---|---|---|---|---|---|---|---|
| Proposed | **.273** | **.341** | **1.87** | **296** | **925** | **1220** | **.280** |
| Search-R1 baseline, 4-bit | .260 | .324 | 2.00 | 331 | 993 | 1324 | .245 |
| Matched control | .237 | .295 | 3.02 | 497 | 1497 | 1994 | .148 |

MRC is mean retrieval calls per question, TTC is total token cost, and CES is
F1 x 1000 / TTC. Arrows mark the direction of improvement, and bold marks the
best value in each column. The Search-R1 baseline is the published model with
no further
training, and the matched control is trained on the same questions and settings
with both mechanisms disabled, so it isolates what the mechanisms add. The
control uses the native Search-R1 prompt, since the budget prompt only exists
once the planner is enabled.

Against the Search-R1 baseline, F1 rises from *.324* to *.341* while retrieval
calls fall *7%* and token cost falls *8%*. Against the matched control, F1
rises from *.295* to *.341* while retrieval calls fall *38%* and token cost
falls *39%*. Quality improves and cost drops together in both comparisons.

Declared budgets correlate with question difficulty at r = *.479* against the
gold budget, measured under sampling on the same 1024 questions. Per dataset
the correlation is *.519* on 2WikiMultiHopQA and *.519* on MuSiQue. It is
undefined on HotpotQA, whose gold budget is uniformly 2, so a flat declaration
there is the correct behaviour.

Retrieval-free and single-shot baselines, measured separately on 256 questions,
place the agentic setting in context. Closed-book QA reaches F1 *.205* and
naive top-3 RAG reaches *.215*, against *.351* for the Search-R1 baseline at
the same quantization. Iterative retrieval, not the reward design, supplies
most of the answer quality.

## Method

Each rollout runs a generate, act, observe loop in which the model emits one
action per turn, the response is truncated at the first closing tag, and the
action is parsed and dispatched.

- `<budget>k</budget>` declares the retrieval cap. It must come first when the
  planner is enabled, and any earlier action invalidates the rollout.
- `<search>query</search>` retrieves the top 3 passages, unless the declared
  budget is already exhausted, in which case the call is blocked and counted.
- `<answer>text</answer>` ends the episode.

Retrieved passages are masked out of the policy gradient, which is a Search-R1
invariant, and the same mask separates generated from retrieved tokens for the
token-cost accounting.

Both arms of the comparison receive a symmetric frozen-model self-distillation
SFT before RL, differing only in prompt template, so the contrast isolates the
two mechanisms rather than the warm start.

### Objective

The scalar reward is

```
R = R_answer + λ·recall
    − γ·max(0, k − N)
    − δ·max(0, k_gold − k)
```

implemented in `search_r1/budgeting.py`, where `R_answer` is the answer reward,
`N` is the number of retrieval calls a rollout made, `k` is the budget it
declared, `k_gold` is the question's reference budget, and `recall` is the
fraction of gold passages its retrievals returned. Both penalties tie the
declaration to behaviour, since one charges for budget declared and left unused
and the other for declaring below the question's gold budget.

The retrieval cost is group-normalised and subtracted from the GRPO advantage.
For rollout `i` in group `g`,

```
Â_i  = z_g(R_i) − c · z_g(N_i) · 1[R_i > 0]

z_g(x) = (x − μ_g) / (σ_g + ε)
```

implemented in `verl/trainer/ppo/core_algos.py`, where `Â_i` is the advantage
of rollout `i`, `z_g` is the z-score across its group with mean `μ_g` and
standard deviation `σ_g`, `ε` prevents division by zero when a group has no
variance, and `1[·]` is the indicator function. Normalising within the group
charges a rollout for being costlier than its siblings on the same question
rather than for its absolute call count, so a question that genuinely needs
more hops is not penalised for taking them. The indicator acts as a gate, so a
group in which no rollout earned positive reward contributes no cost-only
gradient, which prevents the policy from learning to stop searching entirely.

The proposed framework uses these values.

| term | meaning | value |
|---|---|---|
| γ | declared budget left unused | .02 |
| δ | declared below gold budget | .06 |
| λ | gold-passage grounding | .5 |
| c | cost weight in the advantage | .5 |

The declared digit also carries an auxiliary cross-entropy loss toward the gold
budget, weighted .05. Each ablation cell disables one mechanism, listed in
`train_grpo_budget.sh`.

## Repository structure

`verl/` is a vendored copy of veRL supplying the GRPO trainer, workers and
Hydra config system, and the files below are the thesis contribution.

```
Search-R1/
├── search_r1/
│   ├── budgeting.py              reward equation
│   ├── sft_masking.py            SFT loss mask over response spans
│   ├── lora_utils.py             QLoRA adapter resolution
│   ├── resume_utils.py           warm restart from the latest checkpoint
│   └── llm_agent/generation.py   rollout loop, budget parsing and cap
├── scripts/
│   ├── data_process/thesis_qa.py builds RL parquets and gold budgets
│   ├── build_sft_data.py         SFT parquets from frozen-model traces
│   ├── paired_significance.py    paired bootstrap CIs and McNemar
│   ├── build_baseline_data.py    no-RAG and naive-RAG prompt rewrites
│   └── analyze_baselines.py      dump to metric table
├── verl/trainer/
│   ├── main_ppo.py               reward manager, the join point
│   ├── ppo/ray_trainer.py        validation, metrics, checkpoint selection
│   └── config/ppo_trainer.yaml   thesis config blocks, both off by default
├── tests/                        unit tests for the above
├── prepare_data.sh               builds both RL parquets
├── train_sft.sh                  self-distillation warm start
├── train_grpo_budget.sh          proposed framework
├── train_grpo_control.sh         matched control
└── eval_greedy.sh                evaluation, writes the per-question dump
```

The five shell scripts are Slurm templates, each holding one documented command
with a comment block naming the overrides worth changing.

## Installation

Python 3.9 with a CUDA GPU, following the upstream Search-R1 environment.
Training was run on a single 24 GB card using 4-bit QLoRA, which needs `peft`
and `bitsandbytes`.

```bash
conda create -n searchr1 python=3.9
conda activate searchr1
pip install -r requirements.txt
pip install -e .
```

The reported runs used torch 2.4.0 with CUDA 12.1, transformers 4.47.1,
peft 0.19.1, bitsandbytes 0.43.3, vllm 0.6.3, flash-attn 2.6.3 and
tensordict 0.12.4. Install torch separately for your CUDA version.

The retrieval server runs in its own Python 3.10 environment and needs a FAISS
index over the 2018 Wikipedia corpus. See the
[upstream README](SEARCH_R1_README.md) for the retriever setup. Data
preparation and the analysis scripts run on CPU and need only `datasets`,
`numpy`, and `pandas`.

## Reproducing

Commands run from this directory, and since training targets a Slurm cluster
each stage is a submission script. Drop `sbatch` and run them with `bash`
instead.

```bash
# build both RL parquets, budget-aware and native, from one seed
sbatch prepare_data.sh

# warm start, then GRPO with both mechanisms enabled
sbatch train_sft.sh
sbatch train_grpo_budget.sh

# matched control, same settings with both mechanisms off
# edit train_sft.sh to point at the control parquet first
sbatch train_sft.sh
sbatch train_grpo_control.sh

# evaluate a checkpoint, once per condition
sbatch eval_greedy.sh
```

Each condition needs its own edit of `eval_greedy.sh`, changing
`lora.adapter_path` to that condition's `actor/best`, `data.val_files` to the
matching parquet, and `experiment_name`. Ablations work the same way against
`train_grpo_budget.sh`, where each
cell is a single flag.

Every evaluation writes per-question records to
`outputs/premise_eval_<experiment_name>.json`. Build the comparison table from
any number of dumps with

```bash
python3 scripts/analyze_baselines.py --per_source --runs \
    proposed=outputs/premise_eval_eval_budget_greedy.json \
    control=outputs/premise_eval_eval_control_greedy.json
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

Every trained condition holds the same setup, with Search-R1 3B-Instruct as the
base, a 4-bit QLoRA adapter, E5-base-v2 retrieval over a 2018 Wikipedia dump at
top 3, GRPO group size 5, learning rate 5e-6, KL coefficient 0.01, 8 turns
maximum, and 40 training steps.

Every condition selects its best checkpoint on validation F1, with ties broken
by the later checkpoint.

## Design notes

- Budget calibration is a property of the sampled declaration distribution,
  since greedy decoding takes the modal declaration, which collapses to 2 on
  most questions and hides the grading. The correlation above is therefore
  measured under sampling while the reported quality and cost figures are
  greedy.
- The declared budget is supervised by a cross-entropy term at the digit
  position, not by the scalar reward alone, because a scalar penalty moves
  the mean declaration but does not grade it per question.
- Token cost is measured and reported but never charged, since only retrieval
  calls are priced, and the token saving is a downstream effect of issuing
  fewer of them.

## Limitations

- One seed per condition, so none of the differences above are separated from
  seed-to-seed variance.
- Checkpoints are selected on the same split that is reported, because FlashRAG
  exposes no public test split with answers for these datasets.
- TTC and CES are not comparable across prompt formats, since the budget prompt
  adds reasoning tokens to the total. Cross-format cost claims
  lead with MRC, which counts retrieval actions and is prompt-invariant.
- Part of the margin over the matched control comes from the control degrading
  through over-retrieval rather than from the proposed framework improving.
- Results are for a single 3B model family at 4-bit quantization. The 4-bit base
  costs about *.029* F1 against bf16.

## License and attribution

Apache License 2.0. See [LICENSE](LICENSE) for the terms and
[NOTICE](NOTICE) for attribution and the list of modified files.

The repository vendors two upstream projects, both Apache 2.0. `verl/` is
[veRL](https://github.com/volcengine/verl) by Bytedance, supplying the GRPO
trainer, the distributed workers, and the Hydra config system. `search_r1/`,
`example/`, and the upstream `train_ppo.sh` and `train_grpo.sh` come from
[Search-R1](https://github.com/PeterGriffinJin/Search-R1) by Jin et al.,
supplying the agentic rollout and the retriever.

The contribution of this work is the budget mechanism and the cost-aware
objective, implemented in `search_r1/budgeting.py`, the budget parsing and cap
enforcement in `search_r1/llm_agent/generation.py`, the reward manager in
`verl/trainer/main_ppo.py`, the group-normalised cost advantage in
`verl/trainer/ppo/core_algos.py`, and the dataset, SFT, and evaluation tooling
under `scripts/`.
