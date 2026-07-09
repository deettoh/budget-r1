"""Confirm two RL budget parquets differ only in the prompt wording.

Guards the budget-first CE ablation, which trains the v7 recipe on the
v6 parquet. Rows compare position-aligned, valid because both were
built with the same sources, cap and seed.
"""

import sys

import pandas as pd

V6 = "data/thesis_rl_v6_budget/train.parquet"
V7 = "data/thesis_rl_v7_budget/train.parquet"

a = pd.read_parquet(V6)
b = pd.read_parquet(V7)

print(f"v6 rows={len(a)}  v7 rows={len(b)}")
if len(a) != len(b):
    sys.exit("ABORT: row counts differ; not row-aligned")


def question_of(row):
    return row["prompt"][0]["content"].split("Question:")[-1].strip()


def gold_of(row):
    return int(row["extra_info"]["gold_budget"])


q_mismatch = 0
g_mismatch = 0
prompt_same = 0
v6_marker = 0
v7_marker = 0
for i in range(len(a)):
    ra, rb = a.iloc[i], b.iloc[i]
    if question_of(ra) != question_of(rb):
        q_mismatch += 1
    if gold_of(ra) != gold_of(rb):
        g_mismatch += 1
    ca = ra["prompt"][0]["content"]
    cb = rb["prompt"][0]["content"]
    if ca == cb:
        prompt_same += 1
    if "Before reasoning or searching" in ca:
        v6_marker += 1
    if "First think" in cb:
        v7_marker += 1

print(f"question mismatches:     {q_mismatch}")
print(f"gold_budget mismatches:  {g_mismatch}")
print(f"identical prompts:       {prompt_same}")
print(f"v6 budget-first marker:  {v6_marker}/{len(a)}")
print(f"v7 think-first marker:   {v7_marker}/{len(b)}")

ok = (
    q_mismatch == 0
    and g_mismatch == 0
    and prompt_same == 0
    and v6_marker == len(a)
    and v7_marker == len(b)
)
print("\nCLEAN single-variable (prompt only): " + ("YES" if ok else "NO"))
