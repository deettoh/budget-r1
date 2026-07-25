"""Paired significance tests between two val_only dumps.

Pairs on (data_source, index), then reports a paired bootstrap CI per
metric and McNemar's exact test for EM. Both dumps need
trainer.dump_val_text=true, which is what writes extra_info.index.
"""

import argparse
import importlib.util
import json
import os
from dataclasses import dataclass
from math import comb
from typing import Callable

import numpy as np

# import ces by path, the verl chain pulls torch
_CES_SPEC = importlib.util.spec_from_file_location(
    "qa_metrics",
    os.path.join(
        os.path.dirname(__file__), "..", "verl", "utils",
        "reward_score", "qa_metrics.py"),
)
_qa_metrics = importlib.util.module_from_spec(_CES_SPEC)
_CES_SPEC.loader.exec_module(_qa_metrics)
cost_efficiency_score = _qa_metrics.cost_efficiency_score

_N_RESAMPLES = 10000
_SEED = 42
_ALPHA = 0.05
_METRICS = ("em", "f1", "mrc", "ttc")

Metric = Callable[[dict], float]


@dataclass(frozen=True)
class BootstrapResult:
    """One paired bootstrap comparison. Frozen, do not mutate."""

    diff: float
    lo: float
    hi: float

    @property
    def is_significant(self) -> bool:
        """Return True when the CI excludes zero."""
        return self.lo > 0.0 or self.hi < 0.0


@dataclass(frozen=True)
class McNemarResult:
    """Discordant pair counts and the exact two-sided p-value."""

    b: int
    c: int
    p_value: float

    @property
    def is_significant(self) -> bool:
        """Return True at the conventional 5% level."""
        return self.p_value < _ALPHA


def _key(record: dict) -> tuple:
    """Return the (data_source, index) pairing key for one record."""
    if "index" not in record:
        raise ValueError(
            "record has no 'index'; the run needs "
            "trainer.dump_val_text=true to emit it"
        )
    return (record.get("data_source", "unknown"), int(record["index"]))


def _index_by_key(records: list[dict]) -> dict:
    """Return records keyed by (data_source, index), rejecting dupes."""
    keyed: dict = {}
    for record in records:
        key = _key(record)
        if key in keyed:
            raise ValueError(f"duplicate pairing key {key}")
        keyed[key] = record
    return keyed


def align_dumps(
    dump_a: list[dict], dump_b: list[dict]
) -> tuple[list[dict], list[dict]]:
    """Return both dumps cut to their shared keys, in a matching order.

    Order follows dump_a so the two lists line up positionally.
    """
    keyed_a = _index_by_key(dump_a)
    keyed_b = _index_by_key(dump_b)
    shared = [key for key in keyed_a if key in keyed_b]
    return (
        [keyed_a[key] for key in shared],
        [keyed_b[key] for key in shared],
    )


def to_columns(records: list[dict]) -> dict:
    """Return the metric columns as float arrays.

    mrc aliases valid_search_calls and ttc sums the token fields.
    """
    return {
        "em": np.array([float(r["em"]) for r in records]),
        "f1": np.array([float(r["f1"]) for r in records]),
        "mrc": np.array(
            [float(r["valid_search_calls"]) for r in records]
        ),
        "ttc": np.array(
            [
                float(r["generated_tokens"] + r["retrieved_tokens"])
                for r in records
            ]
        ),
    }


def mean_of(field: str) -> Metric:
    """Return a metric taking the mean of one column."""

    def metric(columns: dict) -> float:
        return float(columns[field].mean())

    return metric


def ces_of(columns: dict) -> float:
    """Return CES over a column set, as a ratio of the two means.

    Averaging per-question CES would weight cheap rollouts far too
    heavily, so this matches how _validate reports it.
    """
    return cost_efficiency_score(
        float(columns["f1"].mean()), float(columns["ttc"].mean())
    )


def paired_bootstrap_ci(
    records_a: list[dict],
    records_b: list[dict],
    metric: Metric,
    n_resamples: int = _N_RESAMPLES,
    seed: int = _SEED,
    alpha: float = _ALPHA,
) -> BootstrapResult:
    """Return the a-minus-b difference with a paired bootstrap CI.

    One index draw is applied to both conditions per resample, which
    is what makes the test paired.
    """
    if len(records_a) != len(records_b):
        raise ValueError(
            f"unpaired inputs: {len(records_a)} vs {len(records_b)}"
        )
    if not records_a:
        raise ValueError("no paired records to compare")

    cols_a = to_columns(records_a)
    cols_b = to_columns(records_b)
    observed = metric(cols_a) - metric(cols_b)

    rng = np.random.default_rng(seed)
    size = len(records_a)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        draw = rng.integers(0, size, size)
        resampled_a = {k: v[draw] for k, v in cols_a.items()}
        resampled_b = {k: v[draw] for k, v in cols_b.items()}
        diffs[i] = metric(resampled_a) - metric(resampled_b)

    lo, hi = np.percentile(
        diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)]
    )
    return BootstrapResult(float(observed), float(lo), float(hi))


def mcnemar(
    records_a: list[dict], records_b: list[dict], field: str = "em"
) -> McNemarResult:
    """Return McNemar's exact test on a binary field.

    Agreements carry no information about which side is better, so
    only the discordant pairs enter the binomial.
    """
    if len(records_a) != len(records_b):
        raise ValueError(
            f"unpaired inputs: {len(records_a)} vs {len(records_b)}"
        )

    wins = sum(
        1
        for ra, rb in zip(records_a, records_b)
        if float(ra[field]) > float(rb[field])
    )
    losses = sum(
        1
        for ra, rb in zip(records_a, records_b)
        if float(ra[field]) < float(rb[field])
    )

    total = wins + losses
    if total == 0:
        return McNemarResult(wins, losses, 1.0)

    tail = sum(
        comb(total, i) for i in range(min(wins, losses) + 1)
    )
    p_value = min(1.0, 2.0 * tail * 0.5 ** total)
    return McNemarResult(wins, losses, p_value)


def load_dump(path: str) -> list[dict]:
    """Return the per-sample records of a val_only dump."""
    with open(path) as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"{path} is not a list of records")
    return records


def compare(
    records_a: list[dict],
    records_b: list[dict],
    n_resamples: int = _N_RESAMPLES,
    seed: int = _SEED,
) -> dict:
    """Return every metric CI plus McNemar for one aligned pair."""
    results = {
        name: paired_bootstrap_ci(
            records_a, records_b, mean_of(name), n_resamples, seed
        )
        for name in _METRICS
    }
    results["ces"] = paired_bootstrap_ci(
        records_a, records_b, ces_of, n_resamples, seed
    )
    results["mcnemar_em"] = mcnemar(records_a, records_b)
    return results


def _print_report(
    label_a: str, label_b: str, n: int, results: dict
) -> None:
    """Print the comparison table for one pair of conditions."""
    print(f"\n=== {label_a} vs {label_b}  (n={n} paired) ===")
    print(f"{'metric':<8}{'diff':>12}  {'95% CI':>26}  verdict")
    for name in (*_METRICS, "ces"):
        r = results[name]
        ci = f"[{r.lo:+.4f}, {r.hi:+.4f}]"
        verdict = "SIG" if r.is_significant else "n.s."
        print(f"{name:<8}{r.diff:>+12.4f}  {ci:>26}  {verdict}")

    m = results["mcnemar_em"]
    verdict = "SIG" if m.is_significant else "n.s."
    print(
        f"mcnemar em: b={m.b} c={m.c} p={m.p_value:.5f}  {verdict}"
        f"   (b = {label_a} right / {label_b} wrong)"
    )


def main() -> None:
    """Run the paired comparison between two dumps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump_a", required=True)
    parser.add_argument("--dump_b", required=True)
    parser.add_argument("--label_a", default="A")
    parser.add_argument("--label_b", default="B")
    parser.add_argument("--n_resamples", type=int, default=_N_RESAMPLES)
    parser.add_argument("--seed", type=int, default=_SEED)
    args = parser.parse_args()

    records_a, records_b = align_dumps(
        load_dump(args.dump_a), load_dump(args.dump_b)
    )
    if not records_a:
        raise ValueError(
            "no shared (data_source, index) keys; check both runs "
            "used the same val draw and dump_val_text=true"
        )

    results = compare(
        records_a, records_b, args.n_resamples, args.seed
    )
    _print_report(
        args.label_a, args.label_b, len(records_a), results
    )


if __name__ == "__main__":
    main()
