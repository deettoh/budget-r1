"""Build SFT parquets from frozen-native trace dumps.

Self-distillation: keep EM-correct native rollouts, then emit two
symmetric arms. Treatment carries the budget_first prompt with a
declared <budget>k</budget> (k = the trace's own search count) so the
declaration matches the behavior that follows. Control keeps the native
prompt and response verbatim. The same train/val partition feeds both
arms so a question lands in the same split on each side.
"""

import argparse
import json
import os
import random
import re
import sys

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_process"),
)
import thesis_qa  # noqa: E402

make_search_prefix = thesis_qa.make_search_prefix

MAX_BUDGET = 5
VAL_FRACTION = 0.05
ASSISTANT_MARKER = "<|im_start|>assistant\n"
_QUESTION_RE = re.compile(
    r"Question:\s*(.*?)\s*(?:<\|im_end\|>|$)", re.DOTALL
)
_TRAILING_SPECIAL = ("<|im_end|>", "<|endoftext|>")


def split_prompt_response(text: str) -> tuple[str, str]:
    """Split a decoded trace at the assistant marker.

    Raises:
        ValueError: If the assistant marker is absent.
    """
    if ASSISTANT_MARKER not in text:
        raise ValueError("assistant marker not found in trace")
    prompt, response = text.rsplit(ASSISTANT_MARKER, 1)
    return prompt, response


def extract_question(prompt_part: str) -> str:
    """Return the question from a native prompt segment.

    Raises:
        ValueError: If no non-empty question follows ``Question:``.
    """
    match = _QUESTION_RE.search(prompt_part)
    if match is None or not match.group(1).strip():
        raise ValueError("no question found in prompt segment")
    return match.group(1).strip()


def clean_response(response_part: str) -> str:
    """Strip trailing chat/eos markers and surrounding whitespace."""
    text = response_part
    changed = True
    while changed:
        changed = False
        text = text.rstrip()
        for token in _TRAILING_SPECIAL:
            if text.endswith(token):
                text = text[: -len(token)]
                changed = True
    return text.strip()


def clamp_budget(calls: int, max_budget: int = MAX_BUDGET) -> int:
    """Return the search count clamped to [0, max_budget]."""
    return max(0, min(int(calls), max_budget))


def treatment_response(k: int, response: str) -> str:
    """Prepend the declared budget tag to a native response."""
    return f"<budget>{k}</budget>\n{response}"


def build_rows(records: list) -> tuple[list, list]:
    """Return (treatment_rows, control_rows) from EM-correct traces.

    Raises:
        ValueError: If a kept record lacks sequences_str.
    """
    treatment, control = [], []
    for rec in records:
        if float(rec.get("em", 0.0)) != 1.0:
            continue
        if "sequences_str" not in rec:
            raise ValueError(
                "trace dump missing sequences_str; rerun with "
                "trainer.dump_val_text=true"
            )
        prompt_part, response_part = split_prompt_response(
            rec["sequences_str"]
        )
        question = extract_question(prompt_part)
        response = clean_response(response_part)
        source = rec.get("data_source", "unknown")
        k = clamp_budget(rec.get("valid_search_calls", 0))
        treatment.append({
            "prompt": make_search_prefix(
                question, require_budget=True, max_budget=MAX_BUDGET,
                budget_template="budget_first",
            ),
            "response": treatment_response(k, response),
            "data_source": source,
        })
        control.append({
            "prompt": make_search_prefix(question, require_budget=False),
            "response": response,
            "data_source": source,
        })
    return treatment, control


def split_train_val(n: int, seed: int) -> tuple[list, list]:
    """Return (train_idx, val_idx) with VAL_FRACTION held out."""
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_val = max(1, int(n * VAL_FRACTION)) if n > 1 else 0
    return idx[n_val:], idx[:n_val]


def _write_parquet(rows: list, path: str) -> None:
    """Write rows to a parquet file, creating parent dirs."""
    import pandas as pd

    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_parquet(path)


def _per_source_counts(rows: list) -> dict:
    """Return a count of rows per data_source."""
    counts: dict = {}
    for row in rows:
        counts[row["data_source"]] = counts.get(row["data_source"], 0) + 1
    return counts


def main() -> None:
    """Build the budgetfirst and native SFT parquets from a dump."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace_dump", required=True)
    parser.add_argument("--budgetfirst_dir", default="data/sft_budgetfirst")
    parser.add_argument("--native_dir", default="data/sft_native")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.trace_dump) as f:
        records = json.load(f)
    treatment, control = build_rows(records)
    if not treatment:
        raise ValueError("no EM-correct traces to build SFT data")

    print(f"[sft] kept {len(treatment)} EM-correct traces")
    print(f"[sft] per-source: {_per_source_counts(treatment)}")

    train_idx, val_idx = split_train_val(len(treatment), args.seed)
    for rows, out_dir in (
        (treatment, args.budgetfirst_dir),
        (control, args.native_dir),
    ):
        _write_parquet(
            [rows[i] for i in train_idx],
            os.path.join(out_dir, "train.parquet"),
        )
        _write_parquet(
            [rows[i] for i in val_idx],
            os.path.join(out_dir, "test.parquet"),
        )
        print(f"[sft] wrote {out_dir} "
              f"(train {len(train_idx)}, val {len(val_idx)})")


if __name__ == "__main__":
    main()
