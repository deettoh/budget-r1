"""Summarize val_only per-sample dumps into the thesis metric table.

Reads premise_eval_*.json (a list with one dict per eval sample) and
reports EM, F1, has_answer, MRC (+ 0/1/2/3/4+ call distribution), TTC,
and CES=F1*1000/TTC, overall and per data_source. Validated to
reproduce the premise-check numbers (MRC 1.99 / CES 0.29).

Typical usage example:

  python3 scripts/analyze_baselines.py --per_source \
    --runs frozen=outputs/premise_eval_frozen.json \
    norag=outputs/premise_eval_norag.json
"""

import argparse
import json

_MRC_CAP = 4  # bucket 4+ together in the call distribution


def _mean(values) -> float:
    """Return the mean, or 0.0 for an empty sequence."""
    return sum(values) / len(values) if values else 0.0


def compute_metrics(records: list) -> dict:
    """Return the aggregate metric dict for one set of samples.

    Args:
        records: Per-sample eval records for one group.

    Returns:
        Counts, EM, F1, has_answer, MRC with its call distribution,
        TTC and CES.

    Raises:
        ValueError: Empty records, which would leave every mean and
            the CES ratio undefined.
    """
    if not records:
        raise ValueError("no records to summarize")
    f1 = _mean([r["f1"] for r in records])
    ttc = _mean(
        [r["generated_tokens"] + r["retrieved_tokens"] for r in records]
    )
    dist = {k: 0 for k in range(_MRC_CAP + 1)}
    for r in records:
        dist[min(int(r["valid_search_calls"]), _MRC_CAP)] += 1
    return {
        "n": len(records),
        "em": _mean([r["em"] for r in records]),
        "f1": f1,
        "has_answer": _mean([r["has_answer"] for r in records]),
        "mrc": _mean([r["valid_search_calls"] for r in records]),
        "ttc": ttc,
        "ces": (f1 * 1000 / ttc) if ttc else 0.0,
        "mrc_dist": dist,
    }


def _format_row(label: str, m: dict) -> str:
    """Return one fixed-width table row for a metric dict."""
    dist = "/".join(str(m["mrc_dist"][k]) for k in range(_MRC_CAP + 1))
    return (
        f"{label:<22} {m['n']:>4} {m['em']:>6.3f} {m['f1']:>6.3f} "
        f"{m['has_answer']:>6.3f} {m['mrc']:>6.3f} {m['ttc']:>8.1f} "
        f"{m['ces']:>6.3f}  {dist}"
    )


def _by_source(records: list) -> dict:
    """Return records grouped by their data_source field."""
    groups: dict = {}
    for r in records:
        groups.setdefault(r["data_source"], []).append(r)
    return groups


def main() -> None:
    """Print the comparable baseline table from the dump files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", nargs="+", required=True,
        help="label=path entries for each condition dump",
    )
    parser.add_argument("--per_source", action="store_true")
    args = parser.parse_args()

    header = (
        f"{'condition':<22} {'n':>4} {'EM':>6} {'F1':>6} "
        f"{'hasA':>6} {'MRC':>6} {'TTC':>8} {'CES':>6}  0/1/2/3/4+"
    )
    print(header)
    print("-" * len(header))
    for entry in args.runs:
        label, path = entry.split("=", 1)
        records = json.load(open(path))
        print(_format_row(label, compute_metrics(records)))
        if args.per_source:
            for source, group in sorted(_by_source(records).items()):
                print(_format_row(f"  {source}", compute_metrics(group)))


if __name__ == "__main__":
    main()
