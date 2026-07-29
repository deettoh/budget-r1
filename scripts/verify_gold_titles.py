"""Print whether a built RL parquet carries ``extra_info.gold_titles``.

Sanity check before a grounding run, which needs gold titles present.

Typical usage example:

  python3 scripts/verify_gold_titles.py \
    --parquet data/thesis_rl_budget/train.parquet
"""

import argparse


def main() -> None:
    """Print the row count and the first row's extra_info keys."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True)
    args = parser.parse_args()

    import pandas as pd

    frame = pd.read_parquet(args.parquet)
    first = frame.iloc[0]["extra_info"]
    keys = sorted(first.keys()) if isinstance(first, dict) else type(first)
    print(f"rows={len(frame)} first_extra_info_keys={keys}")

    # a whole-file count hid musique's empty gold_titles
    per_source: dict[str, list[int]] = {}
    for source, extra in zip(frame["data_source"], frame["extra_info"]):
        counts = per_source.setdefault(str(source), [0, 0])
        counts[1] += 1
        if not isinstance(extra, dict):
            continue
        gold = extra.get("gold_titles")
        if gold is not None and len(gold) > 0:
            counts[0] += 1

    empty_sources = []
    for source in sorted(per_source):
        non_empty, total = per_source[source]
        print(f"  {source}: non-empty gold_titles {non_empty}/{total}")
        if non_empty == 0:
            empty_sources.append(source)

    if isinstance(first, dict) and first.get("gold_titles") is not None:
        print(f"sample gold_titles: {list(first['gold_titles'])}")
    else:
        print("WARNING: gold_titles MISSING from first row")

    if empty_sources:
        raise SystemExit(
            f"FAIL: zero gold_titles coverage for {empty_sources}"
        )


if __name__ == "__main__":
    main()
