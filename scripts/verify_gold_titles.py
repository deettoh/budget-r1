"""Print whether a built RL parquet carries ``extra_info.gold_titles``.

Sanity check before a grounding run, which needs gold titles present.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True)
    args = parser.parse_args()

    import pandas as pd

    frame = pd.read_parquet(args.parquet)
    first = frame.iloc[0]["extra_info"]
    keys = sorted(first.keys()) if isinstance(first, dict) else type(first)
    print(f"rows={len(frame)} first_extra_info_keys={keys}")

    non_empty = 0
    for extra in frame["extra_info"]:
        if not isinstance(extra, dict):
            continue
        gold = extra.get("gold_titles")
        if gold is not None and len(gold) > 0:
            non_empty += 1
    print(f"rows with non-empty gold_titles: {non_empty}/{len(frame)}")

    if isinstance(first, dict) and first.get("gold_titles") is not None:
        print(f"sample gold_titles: {list(first['gold_titles'])}")
    else:
        print("WARNING: gold_titles MISSING from first row")


if __name__ == "__main__":
    main()
