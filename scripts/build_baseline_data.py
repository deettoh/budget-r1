"""Rewrite the eval parquet prompts for the no-RAG / naive-RAG baselines.

Reads an existing Search-R1 RL test parquet, extracts the raw
question from each agentic prompt, and rebuilds the prompt as either
a closed-book prompt (no retrieval) or a single retrieve-then-read
prompt with the top-k passages injected up front. Wording mirrors
the Search-R1 template so the only change vs the frozen baseline is
the retrieval regime. The naive-RAG mode retrieves once per question
through the same IVF index used by the agentic runs.
"""

import argparse


def make_closedbook_prefix(question: str) -> str:
    """Return the no-RAG closed-book prompt for one question."""
    return (
        "Answer the given question. "
        "You must conduct reasoning inside <think> and </think> "
        "first. After reasoning, provide the answer inside <answer> "
        "and </answer>, without detailed illustrations. For example, "
        f"<answer> Beijing </answer>. Question: {question}\n"
    )


def make_naiverag_prefix(question: str, passages: str) -> str:
    """Return the naive-RAG prompt with one injected info block."""
    return (
        "Answer the given question based on the given information. "
        "The top search results are provided between <information> "
        "and </information>.\n"
        f"<information>{passages}</information>\n"
        "You must conduct reasoning inside <think> and </think> "
        "first. After reasoning, provide the answer inside <answer> "
        "and </answer>, without detailed illustrations. For example, "
        f"<answer> Beijing </answer>. Question: {question}\n"
    )


def extract_question(prompt_content: str) -> str:
    """Return the raw question from a Search-R1 templated prompt."""
    return prompt_content.rsplit("Question: ", 1)[-1].strip()


def format_passages(docs: list) -> str:
    """Format retrieved docs as the harness _passages2string does."""
    out = ""
    for idx, doc_item in enumerate(docs):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        out += f"Doc {idx + 1}(Title: {title}) {text}\n"
    return out


def _retrieve_passages(questions: list, args) -> list:
    """Return one formatted passage block per question via the IVF."""
    from search_r1.search.retrieval_server import Config, get_retriever

    retriever = get_retriever(Config(
        retrieval_method=args.retriever_name,
        retrieval_topk=args.topk,
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        faiss_gpu=False,
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method="mean",
        retrieval_query_max_length=256,
        retrieval_use_fp16=True,
        retrieval_batch_size=args.batch_size,
        faiss_nprobe=args.nprobe,
    ))
    results, scores = retriever.batch_search(
        query_list=questions, num=args.topk, return_score=True)
    blocks = []
    for docs, doc_scores in zip(results, scores):
        wrapped = [
            {"document": doc, "score": score}
            for doc, score in zip(docs, doc_scores)
        ]
        blocks.append(format_passages(wrapped))
    return blocks


def main() -> None:
    """Build the rewritten baseline parquet for the chosen mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True,
                        choices=["norag", "naiverag"])
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--index_path",
                        default="retrieval_data/e5_IVF.index")
    parser.add_argument("--corpus_path",
                        default="retrieval_data/wiki-18.jsonl")
    parser.add_argument("--retriever_name", default="e5")
    parser.add_argument("--retriever_model",
                        default="intfloat/e5-base-v2")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--nprobe", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()

    import pandas as pd

    frame = pd.read_parquet(args.src)
    questions = [
        extract_question(prompt[0]["content"])
        for prompt in frame["prompt"].tolist()
    ]
    if args.mode == "naiverag":
        blocks = _retrieve_passages(questions, args)
        prompts = [
            make_naiverag_prefix(q, b)
            for q, b in zip(questions, blocks)
        ]
    else:
        prompts = [make_closedbook_prefix(q) for q in questions]

    frame["prompt"] = [
        [{"role": "user", "content": text}] for text in prompts
    ]
    frame.to_parquet(args.out)
    print(f"wrote {len(frame)} {args.mode} records to {args.out}")


if __name__ == "__main__":
    main()
