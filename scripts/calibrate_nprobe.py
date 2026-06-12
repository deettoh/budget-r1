"""Calibrate IVFPQ nprobe against the exact Flat index.

Sweeps nprobe and picks the smallest recovering ~99% of the ceiling,
the exact Flat index or exhaustive IVF when Flat is absent. Heavy deps
load lazily so recall_at_k stays importable for unit tests.
"""

import argparse
import json
import os
import sys

# fully-exhaustive IVF search = the PQ-only recall ceiling
_EXHAUSTIVE_NPROBE = 4096
_DEFAULT_NPROBES = (32, 64, 128, 256, 512, 1024, _EXHAUSTIVE_NPROBE)


def recall_at_k(pred_ids, gold_ids) -> float:
    """Return mean fraction of each query's gold ids found in pred.

    Args:
        pred_ids: Per-query retrieved id lists (the approximate run).
        gold_ids: Per-query ground-truth id lists (same length).
    """
    if len(pred_ids) != len(gold_ids):
        raise ValueError(
            f"pred/gold length mismatch: {len(pred_ids)} vs "
            f"{len(gold_ids)}")
    if not gold_ids:
        raise ValueError("no queries to score")
    total = 0.0
    for pred, gold in zip(pred_ids, gold_ids):
        gold_set = set(gold)
        if not gold_set:
            continue
        hits = sum(1 for i in pred if i in gold_set)
        total += hits / len(gold_set)
    return total / len(gold_ids)


def extract_question(prompt_content: str) -> str:
    """Return the raw question from a Search-R1 templated prompt.

    The template embeds the question after a final "Question: "
    marker; recover it so the encoder sees a natural query rather
    than the instruction boilerplate.
    """
    return prompt_content.rsplit("Question: ", 1)[-1].strip()


def knee_nprobe(recalls: dict, tol: float = 0.99) -> int:
    """Return the smallest nprobe recovering tol of the ceiling.

    Args:
        recalls: Map nprobe -> recall@k vs Flat ground truth.
        tol: Fraction of the exhaustive-nprobe recall to recover.
    """
    ceiling = recalls[_EXHAUSTIVE_NPROBE]
    target = tol * ceiling
    for nprobe in sorted(recalls):
        if recalls[nprobe] >= target:
            return nprobe
    return _EXHAUSTIVE_NPROBE


def _load_queries(path: str, limit: int) -> list:
    """Return up to limit raw questions from a parquet eval set."""
    import pandas as pd

    frame = pd.read_parquet(path)
    questions = [
        extract_question(prompt[0]["content"])
        for prompt in frame["prompt"].tolist()
    ]
    return questions[:limit]


def _search_ids(index, query_emb, topk: int) -> list:
    """Return per-query top-k id lists from a faiss index."""
    _scores, idxs = index.search(query_emb, k=topk)
    return [[int(i) for i in row if i >= 0] for row in idxs]


def _build_encoder(args):
    """Build the e5 Encoder used by the live retrieval server."""
    sys.path.insert(
        0, os.path.join(os.path.dirname(__file__), "..", "search_r1",
                        "search"))
    from retrieval_server import Encoder

    return Encoder(
        model_name=args.retriever_name,
        model_path=args.retriever_model,
        pooling_method="mean",
        max_length=256,
        use_fp16=True,
    )


def main() -> None:
    """Run the nprobe sweep and print the recall decomposition."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flat_index", default=None,
                        help="exact ground truth; omit if not on disk")
    parser.add_argument("--ivf_index", required=True)
    parser.add_argument("--queries", required=True,
                        help="parquet eval set with a prompt column")
    parser.add_argument("--retriever_name", default="e5")
    parser.add_argument("--retriever_model",
                        default="intfloat/e5-base-v2")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--num_queries", type=int, default=2000)
    parser.add_argument("--nprobes", type=int, nargs="+",
                        default=list(_DEFAULT_NPROBES))
    parser.add_argument("--out", default=None,
                        help="optional json path for the results")
    args = parser.parse_args()

    import faiss

    encoder = _build_encoder(args)
    queries = _load_queries(args.queries, args.num_queries)
    query_emb = encoder.encode(queries)

    ivf = faiss.read_index(args.ivf_index)
    nprobes = sorted(set(args.nprobes) | {_EXHAUSTIVE_NPROBE})

    if args.flat_index:
        flat = faiss.read_index(args.flat_index)
        gold = _search_ids(flat, query_emb, args.topk)
        del flat
        gold_source = "flat"
    else:
        # no Flat on disk so exhaustive IVF is the ceiling
        faiss.ParameterSpace().set_index_parameter(
            ivf, "nprobe", _EXHAUSTIVE_NPROBE)
        gold = _search_ids(ivf, query_emb, args.topk)
        gold_source = "ivf_exhaustive"

    recalls = {}
    for nprobe in nprobes:
        if (gold_source == "ivf_exhaustive"
                and nprobe == _EXHAUSTIVE_NPROBE):
            # exhaustive vs itself, skip the costly repeat search
            recalls[nprobe] = 1.0
            continue
        faiss.ParameterSpace().set_index_parameter(
            ivf, "nprobe", nprobe)
        pred = _search_ids(ivf, query_emb, args.topk)
        recalls[nprobe] = recall_at_k(pred, gold)

    ceiling = recalls[_EXHAUSTIVE_NPROBE]
    pq_loss = (1.0 - ceiling) if gold_source == "flat" else None
    chosen = knee_nprobe(recalls)
    ivf_loss = ceiling - recalls[chosen]

    print(f"queries: {len(queries)}  topk: {args.topk}  "
          f"gold: {gold_source}")
    print(f"{'nprobe':>8}  {'recall@k':>9}  {'%ceiling':>9}")
    for nprobe in nprobes:
        pct = recalls[nprobe] / ceiling if ceiling else 0.0
        print(f"{nprobe:>8}  {recalls[nprobe]:>9.4f}  {pct:>9.4f}")
    if pq_loss is None:
        print("PQ loss vs Flat: unavailable (no Flat index)")
    else:
        print(f"PQ loss (1 - ceiling vs Flat): {pq_loss:.4f}")
    print(f"chosen nprobe: {chosen}  residual IVF loss: {ivf_loss:.4f}")

    if args.out:
        payload = {
            "gold_source": gold_source,
            "recalls": recalls,
            "ceiling": ceiling,
            "pq_loss": pq_loss,
            "chosen_nprobe": chosen,
            "ivf_loss_at_chosen": ivf_loss,
        }
        with open(args.out, "w") as handle:
            json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
