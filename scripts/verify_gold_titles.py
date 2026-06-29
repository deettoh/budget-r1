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
    non_empty = sum(
        1
        for extra in frame["extra_info"]
        if isinstance(extra, dict) and len(extra.get("gold_titles") or []) > 0
    )
    print(f"rows={len(frame)} first_extra_info_keys={keys}")
    print(f"rows with non-empty gold_titles: {non_empty}/{len(frame)}")
    if isinstance(first, dict) and first.get("gold_titles") is not None:
        print(f"sample gold_titles: {list(first['gold_titles'])}")
    else:
        print("WARNING: gold_titles MISSING from first row")


if __name__ == "__main__":
    main()
