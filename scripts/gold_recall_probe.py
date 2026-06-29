"""Gold-passage recall probe (offline retriever attribution).

Attributes the k=1 budget collapse to a retriever vs generation limit
by measuring how often the gold supporting passages are retrievable.

Two tiers per question (recall@topk of gold supporting titles):
  A (oracle): query the index with each gold title's own text -> is
     that passage retrievable at all? (corpus/index coverage)
  B (question): query the index with the question -> single-shot recall.

Decision rule:
  A low            -> corpus/index-limited (no reward can fix it)
  A high, B low    -> query/multi-hop-reasoning-limited
  A high, B high   -> generation-limited (evidence is there, unused)

No training and no model rollout: a retriever + index pass only.
"""

import argparse
import json
import os
import re
import sys

# thesis_qa is a standalone module (run elsewhere as a plain script,
# not an installed package), so import it by adding its dir to the path
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_process")
)

import thesis_qa  # noqa: E402

extract_gold_titles = thesis_qa.extract_gold_titles
load_named_dataset = thesis_qa.load_named_dataset
normalize_question = thesis_qa.normalize_question

_WHITESPACE = re.compile(r"\s+")


def normalize_title(title: str) -> str:
    """Lowercase and collapse whitespace for lenient title matching."""
    return _WHITESPACE.sub(" ", title.lower()).strip()


def title_from_contents(contents: str) -> str:
    """Return the title line (first line) of a retrieved passage."""
    return contents.split("\n", 1)[0].strip()


def recall_at_k(retrieved_titles, gold_titles) -> float:
    """Return the fraction of gold titles matched in retrieved titles.

    A gold title counts as hit when it is contained in (or contains) any
    retrieved title after normalization, tolerating minor format drift.
    """
    if not gold_titles:
        return 0.0
    retrieved_norm = [normalize_title(t) for t in retrieved_titles if t]
    hits = 0
    for gold in gold_titles:
        gold_norm = normalize_title(gold)
        if any(
            gold_norm == r or gold_norm in r or r in gold_norm
            for r in retrieved_norm
        ):
            hits += 1
    return hits / len(gold_titles)


def _collect_examples(data_source: str, split: str, num: int) -> list[dict]:
    """Return up to ``num`` examples that carry gold titles."""
    dataset = load_named_dataset(data_source)
    if split not in dataset:
        raise ValueError(
            f"Split '{split}' missing for '{data_source}'. "
            f"Available: {list(dataset.keys())}"
        )
    collected = []
    for example in dataset[split]:
        titles = extract_gold_titles(example, data_source)
        if not titles:
            continue
        collected.append(
            {
                "question": normalize_question(
                    str(example.get("question", ""))
                ),
                "gold_titles": titles,
            }
        )
        if len(collected) >= num:
            break
    return collected


def _build_retriever(args):
    """Return a local FlashRAG retriever (mirrors the rollout config)."""
    from search_r1.search.retrieval_server import Config, get_retriever

    config = Config(
        retrieval_method=args.retriever_name,
        retrieval_topk=args.topk,
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        faiss_gpu=args.faiss_gpu,
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method="mean",
        retrieval_query_max_length=64,
        retrieval_use_fp16=True,
        retrieval_batch_size=args.batch_size,
    )
    return get_retriever(config)


def _retrieve_titles(retriever, queries, topk) -> list[list[str]]:
    """Return per-query lists of retrieved passage titles."""
    results, _ = retriever.batch_search(
        query_list=queries, num=topk, return_score=True
    )
    return [
        [title_from_contents(doc["contents"]) for doc in single]
        for single in results
    ]


def _probe_dataset(retriever, examples, topk) -> dict:
    """Return mean Tier-A and Tier-B recall over ``examples``."""
    questions = [ex["question"] for ex in examples]
    question_titles = _retrieve_titles(retriever, questions, topk)
    tier_b = [
        recall_at_k(retrieved, ex["gold_titles"])
        for ex, retrieved in zip(examples, question_titles)
    ]

    oracle_queries, owner = [], []
    for i, ex in enumerate(examples):
        for gold in ex["gold_titles"]:
            oracle_queries.append(gold)
            owner.append((i, gold))
    oracle_titles = _retrieve_titles(retriever, oracle_queries, topk)
    oracle_hits = [0] * len(examples)
    oracle_total = [0] * len(examples)
    for (idx, gold), retrieved in zip(owner, oracle_titles):
        oracle_total[idx] += 1
        if recall_at_k(retrieved, [gold]) > 0:
            oracle_hits[idx] += 1
    tier_a = [
        oracle_hits[i] / oracle_total[i]
        for i in range(len(examples))
        if oracle_total[i]
    ]

    return {
        "n": len(examples),
        "tier_a_oracle_recall": sum(tier_a) / len(tier_a) if tier_a else 0.0,
        "tier_b_question_recall": (
            sum(tier_b) / len(tier_b) if tier_b else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_sources", default="hotpotqa,2wikimultihopqa,musique"
    )
    parser.add_argument("--split", default="dev")
    parser.add_argument("--num", type=int, default=200)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--index_path", default="retrieval_data/e5_IVF.index")
    parser.add_argument("--corpus_path", default="retrieval_data/wiki-18.jsonl")
    parser.add_argument("--retriever_name", default="e5")
    parser.add_argument("--retriever_model", default="intfloat/e5-base-v2")
    parser.add_argument("--faiss_gpu", action="store_true")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--out", default="outputs/gold_recall_probe.json")
    args = parser.parse_args()

    retriever = _build_retriever(args)
    report = {}
    for raw_source in args.data_sources.split(","):
        data_source = raw_source.strip()
        examples = _collect_examples(data_source, args.split, args.num)
        row = _probe_dataset(retriever, examples, args.topk)
        report[data_source] = row
        print(
            f"{data_source}: n={row['n']} "
            f"tierA(oracle)={row['tier_a_oracle_recall']:.3f} "
            f"tierB(question)={row['tier_b_question_recall']:.3f}"
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
