"""EM-flip diagnostic over a val_only per-sample dump.

Splits samples into single-call and multi-call bands, reports EM/F1
per band and per data_source, and compares the EM gap to the
break-even flip rate. Observational and selection-biased, a small gap
signals Branch B, a large gap only justifies the Branch-A causal test.
"""

import argparse
import json
import os
import sys

# repo root on path so a direct script run can import search_r1
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from search_r1.budgeting import BudgetRewardConfig  # noqa: E402

# binary EM tops out at 1.0, so a flip is worth +1.0 of answer reward
_ANSWER_REWARD_MAX = 1.0
_SINGLE_CALLS = 1  # the single-call band is exactly one retrieval
_MULTI_CALLS = 2   # the multi-call band is two or more retrievals


def _mean(values) -> float:
    """Return the mean, or 0.0 for an empty sequence."""
    return sum(values) / len(values) if values else 0.0


def split_by_calls(
    records: list, low: int = _SINGLE_CALLS, high: int = _MULTI_CALLS
) -> tuple:
    """Split records into single-call (==low) and multi-call (>=high).

    Zero-call samples answer from parametric memory and belong to
    neither band, so they are dropped from this comparison.
    """
    single = [
        r for r in records if int(r["valid_search_calls"]) == low
    ]
    multi = [
        r for r in records if int(r["valid_search_calls"]) >= high
    ]
    return single, multi


def _break_even(records: list, cfg: BudgetRewardConfig) -> float:
    """Return the flip rate at which one extra call breaks even.

    Marginal cost of going 1 -> 2 calls is one extra retrieval:
    alpha for the call plus beta times the tokens it returns. The
    token cost uses the measured mean retrieved tokens per call so it
    is data-driven, not assumed.
    """
    searched = [r for r in records if int(r["valid_search_calls"]) > 0]
    total_calls = sum(int(r["valid_search_calls"]) for r in searched)
    total_tokens = sum(int(r["retrieved_tokens"]) for r in searched)
    tokens_per_call = total_tokens / total_calls if total_calls else 0.0
    marginal_cost = cfg.alpha + cfg.beta * tokens_per_call
    return marginal_cost / _ANSWER_REWARD_MAX


def compute_emflip(
    records: list, cfg: BudgetRewardConfig
) -> dict:
    """Return the single-vs-multi-call EM comparison for samples."""
    if not records:
        raise ValueError("no records to summarize")
    single, multi = split_by_calls(records)
    em_single = _mean([r["em"] for r in single])
    em_multi = _mean([r["em"] for r in multi])
    return {
        "n_single": len(single),
        "n_multi": len(multi),
        "em_single": em_single,
        "em_multi": em_multi,
        "em_delta": em_multi - em_single,
        "f1_single": _mean([r["f1"] for r in single]),
        "f1_multi": _mean([r["f1"] for r in multi]),
        "break_even": _break_even(records, cfg),
    }


def _verdict(m: dict) -> str:
    """Return the observational Branch-A / Branch-B read."""
    if m["em_delta"] > m["break_even"]:
        return "SIGNAL -> Branch A (optimization failure)"
    return "NO SIGNAL -> Branch B (k=1 reward-optimal)"


def _format_row(label: str, m: dict) -> str:
    """Return one fixed-width table row for an EM-flip dict."""
    return (
        f"{label:<22} {m['n_single']:>5} {m['n_multi']:>5} "
        f"{m['em_single']:>7.3f} {m['em_multi']:>7.3f} "
        f"{m['em_delta']:>+7.3f} {m['break_even']:>7.3f}"
    )


def _by_source(records: list) -> dict:
    """Return records grouped by their data_source field."""
    groups: dict = {}
    for r in records:
        groups.setdefault(r["data_source"], []).append(r)
    return groups


def main() -> None:
    """Print the EM-flip table and verdict from one dump file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dump", required=True,
        help="path to a premise_eval_*.json val_only dump",
    )
    parser.add_argument("--per_source", action="store_true")
    args = parser.parse_args()

    with open(args.dump) as f:
        records = json.load(f)
    cfg = BudgetRewardConfig()

    header = (
        f"{'band':<22} {'n=1':>5} {'n>=2':>5} {'EM1':>7} {'EM2+':>7} "
        f"{'dEM':>7} {'brkevn':>7}"
    )
    print(header)
    print("-" * len(header))
    overall = compute_emflip(records, cfg)
    print(_format_row("overall", overall))
    if args.per_source:
        for source, group in sorted(_by_source(records).items()):
            print(_format_row(source, compute_emflip(group, cfg)))
    print()
    print(f"verdict (observational): {_verdict(overall)}")
    print(
        "NOTE selection-biased upward; necessary-not-sufficient. "
        "Confirm Branch A with the forced-execution causal test."
    )


if __name__ == "__main__":
    main()
