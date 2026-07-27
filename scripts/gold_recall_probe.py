"""Gold-passage recall probe, offline retriever attribution.

Measures gold recall@k under an oracle query (tier A, is the passage
reachable) and the question (tier B), swept over --topks. Low A means
corpus-limited, high A with low B query-limited, both high means
generation-limited. Retriever and index pass only, no training.
"""

import argparse
import json
import os
import sys

# thesis_qa is not an installed package, import it by path
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_process")
)

import thesis_qa  # noqa: E402
from search_r1.budgeting import (  # noqa: E402,F401
    normalize_title,
    title_recall,
)

extract_gold_titles = thesis_qa.extract_gold_titles
extract_gold_passages = thesis_qa.extract_gold_passages
load_named_dataset = thesis_qa.load_named_dataset
normalize_question = thesis_qa.normalize_question

# title matching is shared with the grounding reward (single source)
recall_at_k = title_recall


def title_from_contents(contents: str) -> str:
    """Return the title line (first line) of a retrieved passage."""
    return contents.split("\n", 1)[0].strip()


def mean_recall_by_k(
    retrieved_lists: list[list[str]],
    gold_lists: list[list[str]],
    topks: list[int],
) -> dict[int, float]:
    """Return mean recall@k over aligned retrieved and gold lists."""
    result: dict[int, float] = {}
    for k in topks:
        if not gold_lists:
            result[k] = 0.0
            continue
        per = [
            recall_at_k(retrieved[:k], gold)
            for retrieved, gold in zip(retrieved_lists, gold_lists)
        ]
        result[k] = sum(per) / len(per)
    return result


def _oracle_recall_at_k(
    owner: list[tuple[int, str]],
    oracle_titles: list[list[str]],
    num_examples: int,
    k: int,
) -> float:
    """Return mean per-example oracle recall@k.

    Each oracle query targets one gold title; per example = fraction of
    its gold passages whose title is in that query's top-k retrieved.
    """
    hits = [0] * num_examples
    total = [0] * num_examples
    for (idx, gold), retrieved in zip(owner, oracle_titles):
        total[idx] += 1
        if recall_at_k(retrieved[:k], [gold]) > 0:
            hits[idx] += 1
    per = [
        hits[i] / total[i] for i in range(num_examples) if total[i]
    ]
    return sum(per) / len(per) if per else 0.0


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
                "gold_passages": extract_gold_passages(
                    example, data_source
                ),
            }
        )
        if len(collected) >= num:
            break
    return collected


def _build_retriever(args, retrieval_topk: int):
    """Return a local FlashRAG retriever, mirroring the rollout."""
    from search_r1.search.retrieval_server import Config, get_retriever

    config = Config(
        retrieval_method=args.retriever_name,
        retrieval_topk=retrieval_topk,
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


def _oracle_queries(examples, oracle_query):
    """Return (query_text, owner) pairs for the Tier-A oracle probe.

    ``passage`` queries with the gold passage body, ``title`` with the
    gold title; both check whether the gold title is retrieved.
    """
    queries, owner = [], []
    for i, ex in enumerate(examples):
        if oracle_query == "passage":
            for title, body in ex["gold_passages"]:
                queries.append(body)
                owner.append((i, title))
        else:
            for title in ex["gold_titles"]:
                queries.append(title)
                owner.append((i, title))
    return queries, owner


def _probe_dataset(retriever, examples, topks, oracle_query) -> dict:
    """Return Tier-A and Tier-B recall@k sweeps over ``examples``."""
    max_k = max(topks)
    questions = [ex["question"] for ex in examples]
    question_titles = _retrieve_titles(retriever, questions, max_k)
    gold_lists = [ex["gold_titles"] for ex in examples]
    tier_b = mean_recall_by_k(question_titles, gold_lists, topks)

    queries, owner = _oracle_queries(examples, oracle_query)
    oracle_titles = (
        _retrieve_titles(retriever, queries, max_k) if queries else []
    )
    tier_a = {
        k: _oracle_recall_at_k(owner, oracle_titles, len(examples), k)
        for k in topks
    }

    return {
        "n": len(examples),
        "n_oracle_queries": len(queries),
        "oracle_query": oracle_query,
        "topks": list(topks),
        "tier_a_oracle_recall": tier_a,
        "tier_b_question_recall": tier_b,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_sources", default="hotpotqa,2wikimultihopqa,musique"
    )
    parser.add_argument("--split", default="dev")
    parser.add_argument("--num", type=int, default=200)
    parser.add_argument("--topks", default="3")
    parser.add_argument(
        "--oracle_query", default="title", choices=["title", "passage"]
    )
    parser.add_argument("--index_path", default="retrieval_data/e5_IVF.index")
    parser.add_argument(
        "--corpus_path", default="retrieval_data/wiki-18.jsonl"
    )
    parser.add_argument("--retriever_name", default="e5")
    parser.add_argument("--retriever_model", default="intfloat/e5-base-v2")
    parser.add_argument("--faiss_gpu", action="store_true")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--out", default="outputs/gold_recall_probe.json")
    args = parser.parse_args()

    topks = sorted({int(x) for x in args.topks.split(",") if x.strip()})
    if not topks or topks[0] < 1:
        parser.error("--topks must list positive integers, e.g. 3,5,10,20")

    retriever = _build_retriever(args, max(topks))
    report = {}
    for raw_source in args.data_sources.split(","):
        data_source = raw_source.strip()
        examples = _collect_examples(data_source, args.split, args.num)
        row = _probe_dataset(retriever, examples, topks, args.oracle_query)
        report[data_source] = row
        sweep_a = " ".join(
            f"@{k}={row['tier_a_oracle_recall'][k]:.3f}" for k in topks
        )
        sweep_b = " ".join(
            f"@{k}={row['tier_b_question_recall'][k]:.3f}" for k in topks
        )
        print(
            f"{data_source}: n={row['n']} "
            f"oracle={args.oracle_query}({row['n_oracle_queries']}) "
            f"tierA[{sweep_a}] tierB[{sweep_b}]"
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
