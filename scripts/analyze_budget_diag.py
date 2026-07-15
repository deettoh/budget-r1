"""Item-1 frozen budget-prompt diagnostic over a val_only dump.

Attributes per-source F1 gaps on two axes, format and cap behaviour.
Pass --native_dump for the well-formed-vs-native F1 column.
"""

import argparse
import json

_MAX_BUDGET = 5  # declared-k range upper bound for the distribution


def _mean(values) -> float:
    """Return the mean, or 0.0 for an empty sequence."""
    return sum(values) / len(values) if values else 0.0


def _by_source(records: list) -> dict:
    """Return records grouped by their data_source field."""
    groups: dict = {}
    for r in records:
        groups.setdefault(r["data_source"], []).append(r)
    return groups


def has_blocked_field(records: list) -> bool:
    """Return True iff any record carries blocked_search_calls."""
    return any("blocked_search_calls" in r for r in records)


def split_well_formed(records: list) -> tuple:
    """Split records into (well_formed, malformed) on declared_budget.

    Malformed rows carry -1, so the sign is the well-formed flag.
    """
    well = [r for r in records if int(r["declared_budget"]) >= 0]
    mal = [r for r in records if int(r["declared_budget"]) < 0]
    return well, mal


def declared_k_distribution(records: list) -> dict:
    """Return well-formed counts per declared k in 0..max_budget."""
    well, _ = split_well_formed(records)
    dist = {k: 0 for k in range(_MAX_BUDGET + 1)}
    for r in well:
        k = int(r["declared_budget"])
        if k in dist:
            dist[k] += 1
    return dist


def f1_by_k(records: list) -> dict:
    """Return mean F1 per declared k over well-formed records."""
    well, _ = split_well_formed(records)
    groups: dict = {}
    for r in well:
        groups.setdefault(int(r["declared_budget"]), []).append(r["f1"])
    return {k: _mean(v) for k, v in groups.items()}


def _blocked_stats(well_formed: list):
    """Return (rate_with_blocked, mean_blocked) over well-formed.

    (None, None) when the dump predates blocked_search_calls, so the
    caller flags the axis unavailable instead of a silent zero.
    """
    if not has_blocked_field(well_formed):
        return None, None
    blocked = [int(r.get("blocked_search_calls", 0)) for r in well_formed]
    rate = _mean([1.0 if b > 0 else 0.0 for b in blocked])
    return rate, _mean(blocked)


def compute_axes(records: list) -> dict:
    """Return the format + cap-behavioral axis summary for records."""
    if not records:
        raise ValueError("no records to summarize")
    well, mal = split_well_formed(records)
    f1_well = _mean([r["f1"] for r in well])
    f1_mal = _mean([r["f1"] for r in mal])
    blocked_rate, blocked_mean = _blocked_stats(well)
    return {
        "n": len(records),
        "n_well_formed": len(well),
        "n_malformed": len(mal),
        "well_formed_rate": len(well) / len(records),
        "f1_well_formed": f1_well,
        "f1_malformed": f1_mal,
        "f1_gap": f1_well - f1_mal,
        "mean_valid_calls": _mean(
            [r["valid_search_calls"] for r in records]
        ),
        "blocked_rate_wf": blocked_rate,
        "mean_blocked_wf": blocked_mean,
    }


def native_f1_by_source(records: list) -> dict:
    """Return mean F1 per data_source for a frozen-native dump."""
    return {
        source: _mean([r["f1"] for r in group])
        for source, group in _by_source(records).items()
    }


def native_f1_overall(records: list):
    """Return sample-weighted native F1, or None when empty.

    Weight by record, not by source. Averaging per-source means
    misweights uneven splits.
    """
    if not records:
        return None
    return _mean([r["f1"] for r in records])


def _fmt_opt(value, spec: str) -> str:
    """Format an optional float, or 'n/a' when it is None."""
    return "n/a".rjust(len(format(0.0, spec))) if value is None \
        else format(value, spec)


def _format_row(label: str, m: dict, native: float = None) -> str:
    """Return one fixed-width axis-table row for one group."""
    nat = _fmt_opt(native, ">7.3f")
    delta = _fmt_opt(
        None if native is None else m["f1_well_formed"] - native, ">+7.3f"
    )
    return (
        f"{label:<20} {m['n']:>4} {m['well_formed_rate']:>6.3f} "
        f"{m['f1_well_formed']:>7.3f} {m['f1_malformed']:>7.3f} "
        f"{m['f1_gap']:>+7.3f} {nat} {delta} "
        f"{m['mean_valid_calls']:>6.3f} "
        f"{_fmt_opt(m['blocked_rate_wf'], '>7.3f')} "
        f"{_fmt_opt(m['mean_blocked_wf'], '>7.3f')}"
    )


def _print_axis_table(
    records: list, native_by_src: dict, native_overall=None
) -> None:
    """Print the per-source format + cap-behavioral axis table."""
    header = (
        f"{'data_source':<20} {'n':>4} {'wfrate':>6} {'F1wf':>7} "
        f"{'F1mal':>7} {'gap':>7} {'F1nat':>7} {'wf-nat':>7} "
        f"{'MRC':>6} {'blkrt':>7} {'blkmn':>7}"
    )
    print(header)
    print("-" * len(header))
    print(_format_row("OVERALL", compute_axes(records), native_overall))
    for source, group in sorted(_by_source(records).items()):
        print(_format_row(
            source, compute_axes(group), native_by_src.get(source)
        ))


def _print_k_tables(records: list) -> None:
    """Print declared-k distribution and F1-vs-k per data_source."""
    print("\ndeclared-k distribution and F1-vs-k (well-formed only)")
    for source, group in sorted(_by_source(records).items()):
        dist = declared_k_distribution(group)
        by_k = f1_by_k(group)
        counts = " ".join(
            f"k{k}={dist[k]}" for k in range(_MAX_BUDGET + 1)
        )
        f1s = " ".join(f"k{k}={by_k[k]:.3f}" for k in sorted(by_k))
        print(f"  {source:<18} {counts}")
        print(f"  {'':<18} F1: {f1s}")


def main() -> None:
    """Print the item-1 axis attribution from the dump file(s)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dump", required=True,
        help="frozen budget-prompt premise_eval_*.json val_only dump",
    )
    parser.add_argument(
        "--native_dump", default=None,
        help="frozen-native dump for the well-formed-vs-native F1 column",
    )
    args = parser.parse_args()

    with open(args.dump) as f:
        records = json.load(f)
    native_by_src = {}
    native_overall = None
    if args.native_dump:
        with open(args.native_dump) as f:
            native_records = json.load(f)
        native_by_src = native_f1_by_source(native_records)
        native_overall = native_f1_overall(native_records)

    _print_axis_table(records, native_by_src, native_overall)
    _print_k_tables(records)

    if not has_blocked_field(records):
        print(
            "\n[WARN] blocked_search_calls absent -> cap-behavioral axis "
            "unavailable. Re-run val_only after the main_ppo field add."
        )
    print(
        "\nNOTE optimization axis (within-group reward std, advantage "
        "norm, KL) is training-time; pull from wandb/console, not here."
    )


if __name__ == "__main__":
    main()
