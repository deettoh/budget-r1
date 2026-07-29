"""Build SFT parquets from frozen-native trace dumps.

Keeps EM-correct rollouts and emits two symmetric arms. Treatment
splices a declared <budget>k</budget> into the trace, control keeps the
native response verbatim. Both arms share one train/val partition so a
question lands in the same split on each side.

Attributes:
    MAX_BUDGET: Upper clamp on a spliced declaration.
    VAL_FRACTION: Share of questions held out for validation.
    SFT_BUDGET_TEMPLATES: Accepted --budget_template values.
    ASSISTANT_MARKER: Token marking where a trace response begins.

Typical usage example:

  python3 scripts/build_sft_data.py \
    --trace_dump outputs/frozen_native_dump.json
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
SFT_BUDGET_TEMPLATES = ("budget_first", "think_first")
ASSISTANT_MARKER = "<|im_start|>assistant\n"
_QUESTION_RE = re.compile(
    r"Question:\s*(.*?)\s*(?:<\|im_end\|>|$)", re.DOTALL
)
_TRAILING_SPECIAL = ("<|im_end|>", "<|endoftext|>")


def split_prompt_response(text: str) -> tuple[str, str]:
    """Split a decoded trace at the assistant marker.

    Args:
        text: One decoded rollout trace.

    Returns:
        The prompt and response halves.

    Raises:
        ValueError: If the assistant marker is absent.
    """
    if ASSISTANT_MARKER not in text:
        raise ValueError("assistant marker not found in trace")
    prompt, response = text.rsplit(ASSISTANT_MARKER, 1)
    return prompt, response


def extract_question(prompt_part: str) -> str:
    """Return the question from a native prompt segment.

    Args:
        prompt_part: Prompt half of a split trace.

    Returns:
        The question text, stripped.

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


def treatment_response(
    k: int, response: str, budget_template: str = "budget_first"
) -> str:
    """Return the trace response with <budget>k</budget> spliced in.

    Args:
        k: Budget to declare.
        response: Native trace response to splice into.
        budget_template: budget_first prepends the tag, think_first
            inserts it after the pre-search reasoning, before the
            first action tag.

    Returns:
        The response carrying the declaration.

    Raises:
        ValueError: On unknown template, or for think_first when the
            response has neither a <search> nor an <answer> tag.
    """
    if budget_template == "budget_first":
        return f"<budget>{k}</budget>\n{response}"
    if budget_template == "think_first":
        # native traces have no <think> tags, declare at first action
        positions = [
            pos for pos in (
                response.find(tag) for tag in ("<search>", "<answer>")
            )
            if pos != -1
        ]
        if not positions:
            raise ValueError(
                "think_first splice needs a <search> or <answer> tag "
                "in the trace response"
            )
        first_action = min(positions)
        return (
            f"{response[:first_action]}<budget>{k}</budget>\n"
            f"{response[first_action:]}"
        )
    raise ValueError(
        f"unknown budget_template {budget_template!r}; "
        f"expected one of {SFT_BUDGET_TEMPLATES}"
    )


def _resolve_budget(rec: dict, budget_label: str) -> int:
    """Return the k to declare for one record under budget_label.

    Args:
        rec: One trace record from the dump.
        budget_label: used takes the trace's own search count, gold
            takes the extra-info gold budget. gold fails loud on a
            missing or negative value so a bad dump cannot silently
            produce mislabeled data.

    Returns:
        The budget to declare, clamped to the valid range.

    Raises:
        ValueError: On unknown budget_label or unusable gold_budget.
    """
    if budget_label == "used":
        return clamp_budget(rec.get("valid_search_calls", 0))
    if budget_label == "gold":
        gold = rec.get("gold_budget")
        if gold is None or int(gold) < 0:
            raise ValueError(
                "budget_label=gold needs a non-negative gold_budget "
                "on every kept record"
            )
        return max(1, clamp_budget(int(gold)))
    raise ValueError(f"unknown budget_label {budget_label!r}")


def filter_em_correct(records: list) -> list:
    """Return only the EM-correct trace records, order kept."""
    return [
        rec for rec in records if float(rec.get("em", 0.0)) == 1.0
    ]


def upsample_records(
    records: list, budget_label: str, max_factor: int, seed: int
) -> list:
    """Return records with minority (source, k) groups duplicated.

    Train partition only, so duplicates never straddle the val split.

    Args:
        records: Records to upsample, read but not mutated.
        budget_label: Selector passed to _resolve_budget for grouping.
        max_factor: Cap on how far a group may grow, as a multiple of
            its own size. Non-positive returns the input unchanged.
        seed: Seed for the per-group shuffle.

    Returns:
        A new list where smaller budget groups are duplicated toward
        the source's largest.
    """
    if max_factor <= 0:
        return records

    by_source: dict = {}
    for position, rec in enumerate(records):
        source = rec.get("data_source", "unknown")
        k = _resolve_budget(rec, budget_label)
        by_source.setdefault(source, {}).setdefault(k, []).append(
            position
        )

    out = list(records)
    for source in sorted(by_source):
        groups = by_source[source]
        majority = max(len(v) for v in groups.values())
        for k in sorted(groups):
            positions = groups[k]
            target = min(majority, len(positions) * max_factor)
            extra = target - len(positions)
            if extra <= 0:
                continue
            pool = list(positions)
            random.Random(f"{seed}:{source}:{k}").shuffle(pool)
            out.extend(
                records[pool[i % len(pool)]] for i in range(extra)
            )
    return out


def balance_records(
    records: list, budget_label: str, cap: int, seed: int
) -> list:
    """Return EM-correct records capped per (source, budget) group.

    Args:
        records: Records to cap, read but not mutated.
        budget_label: Selector passed to _resolve_budget for grouping.
        cap: Maximum rows per group. Non-positive returns the input
            unchanged.
        seed: Seed for the per-group shuffle that picks survivors.

    Returns:
        A new list in the original order. Non-EM-correct records are
        dropped when the cap is active, since their budget label is
        meaningless.
    """
    if cap <= 0:
        return records

    by_group: dict = {}
    for position, rec in enumerate(records):
        if float(rec.get("em", 0.0)) != 1.0:
            continue
        key = (
            rec.get("data_source", "unknown"),
            _resolve_budget(rec, budget_label),
        )
        by_group.setdefault(key, []).append(position)

    keep: set = set()
    for key in sorted(by_group, key=str):
        positions = list(by_group[key])
        random.Random(f"{seed}:{key[0]}:{key[1]}").shuffle(positions)
        keep.update(positions[:cap])

    return [rec for i, rec in enumerate(records) if i in keep]


def build_rows(
    records: list,
    budget_label: str = "used",
    budget_template: str = "budget_first",
) -> tuple[list, list]:
    """Return (treatment_rows, control_rows) from EM-correct traces.

    Args:
        records: Trace records to build from.
        budget_label: Selector passed to _resolve_budget.
        budget_template: Splice style for the treatment arm.

    Returns:
        The two arms, where treatment carries a spliced declaration
        and control keeps the native response verbatim.

    Raises:
        ValueError: On unknown budget_template, or if a kept record
            lacks sequences_str or a usable budget for
            ``budget_label``.
    """
    if budget_template not in SFT_BUDGET_TEMPLATES:
        raise ValueError(
            f"unknown budget_template {budget_template!r}; "
            f"expected one of {SFT_BUDGET_TEMPLATES}"
        )
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
        k = _resolve_budget(rec, budget_label)
        treatment.append({
            "prompt": make_search_prefix(
                question, require_budget=True, max_budget=MAX_BUDGET,
                budget_template=budget_template,
            ),
            "response": treatment_response(
                k, response, budget_template=budget_template
            ),
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
    parser.add_argument(
        "--budget_label", choices=("used", "gold"), default="used",
        help="declared k source: trace search count or gold_budget",
    )
    parser.add_argument(
        "--budget_template", choices=SFT_BUDGET_TEMPLATES,
        default="budget_first",
        help="declaration placement: prepend or after reasoning",
    )
    parser.add_argument(
        "--balance_budget", type=int, default=0,
        help="cap kept traces per (source, k) group; 0 = off",
    )
    parser.add_argument(
        "--upsample_budget", type=int, default=0,
        help="duplicate minority (source, k) train rows toward the "
             "source majority, at most this factor; 0 = off",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.balance_budget > 0 and args.upsample_budget > 0:
        raise ValueError(
            "use --balance_budget or --upsample_budget, not both"
        )

    with open(args.trace_dump) as f:
        records = json.load(f)
    records = filter_em_correct(records)
    if not records:
        raise ValueError("no EM-correct traces to build SFT data")
    records = balance_records(
        records, args.budget_label, args.balance_budget, args.seed
    )

    print(f"[sft] kept {len(records)} EM-correct traces")

    # split before upsampling so duplicates never cross the val split
    train_idx, val_idx = split_train_val(len(records), args.seed)
    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]
    train_records = upsample_records(
        train_records, args.budget_label, args.upsample_budget,
        args.seed,
    )

    for split_records, filename in (
        (train_records, "train.parquet"),
        (val_records, "test.parquet"),
    ):
        treatment, control = build_rows(
            split_records,
            budget_label=args.budget_label,
            budget_template=args.budget_template,
        )
        print(f"[sft] {filename} per-source: "
              f"{_per_source_counts(treatment)}")
        _write_parquet(
            treatment, os.path.join(args.budgetfirst_dir, filename)
        )
        _write_parquet(
            control, os.path.join(args.native_dir, filename)
        )
    print(f"[sft] wrote {args.budgetfirst_dir} and {args.native_dir} "
          f"(train {len(train_records)}, val {len(val_records)})")


if __name__ == "__main__":
    main()
