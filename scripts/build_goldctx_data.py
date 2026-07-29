"""Build the gold-context (oracle-RAG) eval parquet.

Injects each question's gold passages into the naive-RAG prompt for a
max_turns=0 read. A high score means the reader converts handed
evidence (query-limited collapse), a low score generation-limited.
Gold comes from the dataset, no GPU or retriever.

Typical usage example:

  python3 scripts/build_goldctx_data.py --num 200 \
    --out data/goldctx/test.parquet
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "data_process"))
sys.path.insert(0, _HERE)

import build_baseline_data  # noqa: E402
import thesis_qa  # noqa: E402

make_naiverag_prefix = build_baseline_data.make_naiverag_prefix


def format_gold_passages(pairs: list[tuple[str, str]]) -> str:
    """Format gold title and body pairs as the naive-RAG block."""
    out = ""
    for idx, (title, body) in enumerate(pairs):
        out += f"Doc {idx + 1}(Title: {title}) {body}\n"
    return out


def make_goldctx_record(
    example: dict, idx: int, data_source: str, split: str
) -> dict | None:
    """Return one gold-context eval record, or None if no gold passages.

    Args:
        example: Source dataset row.
        idx: Row index, stored in extra_info for pairing.
        data_source: Dataset name driving gold-passage extraction.
        split: Split name recorded on the record.

    Returns:
        A record using the same prompt wording as the naive-RAG
        baseline, so the only difference is gold-vs-retrieved
        passages. None when the row exposes no gold passages.
    """
    pairs = thesis_qa.extract_gold_passages(example, data_source)
    if not pairs:
        return None
    question = str(example.get("question", "")).strip()
    answer = thesis_qa.extract_answer(example)
    prompt = make_naiverag_prefix(question, format_gold_passages(pairs))
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": prompt}],
        "ability": "fact-reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": {"target": answer},
        },
        "extra_info": {"split": split, "index": idx},
    }


def main() -> None:
    """Build the gold-context eval parquet across data sources."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_sources", default="hotpotqa,2wikimultihopqa"
    )
    parser.add_argument("--split", default="dev")
    parser.add_argument("--num", type=int, default=200)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import pandas as pd

    records = []
    for raw_source in args.data_sources.split(","):
        data_source = raw_source.strip()
        dataset = thesis_qa.load_named_dataset(data_source)
        if args.split not in dataset:
            raise ValueError(
                f"Split {args.split!r} missing for {data_source!r}. "
                f"Available: {list(dataset.keys())}"
            )
        count = 0
        for idx, example in enumerate(dataset[args.split]):
            record = make_goldctx_record(
                example, idx, data_source, args.split
            )
            if record is None:
                continue
            records.append(record)
            count += 1
            if count >= args.num:
                break
        print(f"{data_source}: {count} gold-context records")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(records).to_parquet(args.out)
    print(f"wrote {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()
