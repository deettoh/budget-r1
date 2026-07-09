"""Build Search-R1 RL parquet datasets, one record per QA example.

--require_budget switches to the budget prompt. extra_info always
carries gold_budget and gold_titles regardless.
"""

import argparse
import os
from typing import Any, Iterable


SUPPORTED_DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
SUPPORTED_MODES = ("rl",)


def normalize_question(question: str) -> str:
    """Return ``question`` stripped, ending with ``?``."""
    question = question.strip()
    if question and question[-1] != "?":
        question += "?"
    return question


BUDGET_TEMPLATES = (
    "think_first", "minimal", "reason_first", "soft", "budget_first",
)


def make_search_prefix(
    question: str,
    require_budget: bool = False,
    max_budget: int = 5,
    budget_template: str = "think_first",
) -> str:
    """Return the rollout user-prompt prefix.

    With require_budget the model declares <budget>k</budget> before
    its first search. budget_template picks the wording.

    Raises:
        ValueError: If ``budget_template`` is unknown.
    """
    if budget_template not in BUDGET_TEMPLATES:
        raise ValueError(
            f"Unknown budget_template {budget_template!r}; "
            f"expected one of {BUDGET_TEMPLATES}"
        )
    question = normalize_question(question)
    if require_budget and budget_template == "minimal":
        return (
            "Answer the given question. "
            "You must conduct reasoning inside <think> and </think> first every time you get new information. "
            "Before your first search, declare your retrieval budget as <budget>k</budget>, "
            f"where k is an integer in [0, {max_budget}]. "
            "After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> "
            "and it will return the top searched results between <information> and </information>. "
            "You can search as many times as your want, up to your budget k. "
            "If you find no further external knowledge needed, you can directly provide the answer inside <answer> and "
            f"</answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"
        )
    if require_budget and budget_template == "reason_first":
        return (
            "Answer the given question. First, think step by step inside <think> and </think> about what the "
            "question is asking and what information you will need. Based on that reasoning, decide how many "
            f"searches you will need and state it as <budget>k</budget>, an integer from 0 to {max_budget}. "
            "Then, if you find you lack some knowledge, you can call a search engine by <search> query </search> "
            "and it will return the top searched results between <information> and </information>. "
            "Continue reasoning inside <think> and </think> after each result. You may search up to k times. "
            "When you have enough, provide the final answer inside <answer> and </answer>, without detailed "
            f"illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"
        )
    if require_budget and budget_template == "budget_first":
        return (
            "Answer the given question. Before reasoning or searching, you must first output exactly one retrieval budget "
            f"as <budget>k</budget>, where k is an integer in [0, {max_budget}]. "
            "Use k as the maximum number of search calls needed. After the budget, "
            "you must conduct reasoning inside <think> and </think> first every time you get new information. "
            "If you find you lack some knowledge, you can call a search engine by <search> query </search>; "
            "and it will return the top searched results between <information> and </information>. "
            "You can search as many times as you want up to the budget k. "
            "If you find no further external knowledge needed, you can directly provide the answer inside <answer> and "
            f"</answer> without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"
        )
    if require_budget and budget_template == "soft":
        return (
            "Answer the given question. "
            "You must conduct reasoning inside <think> and </think> first every time you get new information. "
            "As a rough plan, first note about how many searches you expect to need as <budget>k</budget>, "
            f"an integer from 0 to {max_budget}; this is just a guide. "
            "If you find you lack some knowledge, you can call a search engine by <search> query </search> "
            "and it will return the top searched results between <information> and </information>. "
            "You can search as many times as needed, up to k. "
            "If you find no further external knowledge needed, you can directly provide the answer inside <answer> and "
            f"</answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"
        )
    if require_budget:
        return (
            "Answer the given question. First think inside <think> and </think> about what the question asks "
            "and how many search calls it likely needs. Then, before any search, output exactly one retrieval budget "
            f"as <budget>k</budget>, where k is an integer in [0, {max_budget}]. "
            "Use k as the maximum number of search calls needed. After the budget, "
            "you must conduct reasoning inside <think> and </think> first every time you get new information. "
            "If you find you lack some knowledge, you can call a search engine by <search> query </search>; "
            "and it will return the top searched results between <information> and </information>. "
            "You can search as many times as you want up to the budget k. "
            "If you find no further external knowledge needed, you can directly provide the answer inside <answer> and "
            f"</answer> without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"
        )

    return (
        "Answer the given question. "
        "You must conduct reasoning inside <think> and </think> first every time you get new information. "
        "After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> "
        "and it will return the top searched results between <information> and </information>. "
        "You can search as many times as your want. "
        "If you find no further external knowledge needed, you can directly provide the answer inside <answer> and "
        f"</answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"
    )


def _as_list(value: Any) -> list[Any]:
    """Coerce ``value`` to a list (``None`` -> ``[]``, else wrap)."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def extract_answer(example: dict[str, Any]) -> Any:
    """Return the first non-empty answer field.

    Tries ``golden_answers``, ``answers``, ``answer`` in order.

    Raises:
        KeyError: If none are present or non-empty.
    """
    for key in ("golden_answers", "answers", "answer"):
        if key in example and example[key] not in (None, ""):
            return example[key]
    raise KeyError("Could not find an answer field in the example.")


MAX_GOLD_BUDGET = 5


def _titles_from_supporting_facts(supporting_facts: Any) -> set[str]:
    """Return distinct titles from any supporting-facts shape.

    Handles the columnar-dict, [title, sent_idx] pairs, and dict forms.
    """
    titles: set[str] = set()
    if isinstance(supporting_facts, dict):
        for title in _as_list(supporting_facts.get("title")):
            titles.add(str(title))
    elif isinstance(supporting_facts, (list, tuple)):
        for fact in supporting_facts:
            if isinstance(fact, (list, tuple)) and fact:
                titles.add(str(fact[0]))
            elif isinstance(fact, dict):
                title = (
                    fact.get("title") or fact.get("page") or fact.get("paragraph_id")
                )
                if title is not None:
                    titles.add(str(title))
    return titles


def derive_gold_budget(example: dict[str, Any], data_source: str) -> int:
    """Return gold budget clamped to ``[1, MAX_GOLD_BUDGET]``.

    Tries supporting-fact titles, supporting-paragraph ids, then the
    MuSiQue question-decomposition fallback.

    Raises:
        ValueError: If no usable signal is present.
    """
    metadata = example.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}

    supporting_facts = example.get("supporting_facts") or metadata.get(
        "supporting_facts"
    )
    titles = _titles_from_supporting_facts(supporting_facts)
    if titles:
        return min(max(len(titles), 1), MAX_GOLD_BUDGET)

    paragraphs = (
        example.get("paragraphs")
        or example.get("supporting_paragraphs")
        or metadata.get("paragraphs")
        or metadata.get("supporting_paragraphs")
    )
    if paragraphs:
        ids = set()
        for paragraph in paragraphs:
            if isinstance(paragraph, dict) and paragraph.get("is_supporting"):
                pid = (
                    paragraph.get("idx")
                    or paragraph.get("id")
                    or paragraph.get("paragraph_id")
                    or paragraph.get("title")
                )
                if pid is not None:
                    ids.add(str(pid))
        if ids:
            return min(max(len(ids), 1), MAX_GOLD_BUDGET)

    if data_source == "musique":
        decompositions = _as_list(
            example.get("question_decomposition")
            or metadata.get("question_decomposition")
        )
        if decompositions:
            return min(max(len(decompositions), 1), MAX_GOLD_BUDGET)

    raise ValueError(
        f"Could not derive gold budget for data_source={data_source!r}: "
        "no supporting_facts, supporting paragraphs, or question decomposition "
        f"found. top-level keys={sorted(example.keys())}, "
        f"metadata keys={sorted(metadata.keys())}"
    )


def extract_gold_titles(
    example: dict[str, Any], data_source: str
) -> list[str]:
    """Return distinct gold supporting-passage titles, sorted.

    Reuses the title parser plus MuSiQue paragraphs, empty when none.
    """
    metadata = example.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}

    supporting_facts = example.get("supporting_facts") or metadata.get(
        "supporting_facts"
    )
    titles = _titles_from_supporting_facts(supporting_facts)

    paragraphs = (
        example.get("paragraphs")
        or example.get("supporting_paragraphs")
        or metadata.get("paragraphs")
        or metadata.get("supporting_paragraphs")
    )
    for paragraph in _as_list(paragraphs):
        if (
            isinstance(paragraph, dict)
            and paragraph.get("is_supporting")
            and paragraph.get("title")
        ):
            titles.add(str(paragraph["title"]))

    # flashrag musique titles live only in the decomposition
    decomposition = example.get("question_decomposition") or metadata.get(
        "question_decomposition"
    )
    for step in _as_list(decomposition):
        if not isinstance(step, dict):
            continue
        support = step.get("support_paragraph")
        if isinstance(support, dict) and support.get("title"):
            titles.add(str(support["title"]))

    return sorted(titles)


def _join_sentences(value: Any) -> str:
    """Join a sentence group (list or string) into one body string."""
    if isinstance(value, str):
        return value.strip()
    return " ".join(str(s) for s in _as_list(value)).strip()


def _context_title_to_body(context: Any) -> dict[str, str]:
    """Map passage title -> joined body text from a context field.

    Handles the columnar-dict and list-of-pairs/dict forms.
    """
    bodies: dict[str, str] = {}
    if isinstance(context, dict):
        sentences = (
            context.get("content")
            or context.get("sentences")
            or context.get("text")
        )
        for title, group in zip(
            _as_list(context.get("title")), _as_list(sentences)
        ):
            bodies[str(title)] = _join_sentences(group)
    elif isinstance(context, (list, tuple)):
        for item in context:
            if isinstance(item, dict) and item.get("title") is not None:
                body = (
                    item.get("sentences")
                    or item.get("content")
                    or item.get("text")
                    or item.get("paragraph_text")
                )
                bodies[str(item["title"])] = _join_sentences(body)
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                bodies[str(item[0])] = _join_sentences(item[1])
    return bodies


def extract_gold_passages(
    example: dict[str, Any], data_source: str
) -> list[tuple[str, str]]:
    """Return ``(title, body)`` for each gold supporting passage.

    Body is the passage text for the recall-probe oracle query.
    Passages whose body cannot be recovered are skipped.
    """
    gold_titles = set(extract_gold_titles(example, data_source))
    if not gold_titles:
        return []

    metadata = example.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}

    bodies: dict[str, str] = {}
    paragraphs = (
        example.get("paragraphs")
        or example.get("supporting_paragraphs")
        or metadata.get("paragraphs")
        or metadata.get("supporting_paragraphs")
    )
    for paragraph in _as_list(paragraphs):
        if (
            isinstance(paragraph, dict)
            and paragraph.get("title") in gold_titles
        ):
            body = _join_sentences(
                paragraph.get("paragraph_text")
                or paragraph.get("text")
                or paragraph.get("sentences")
            )
            if body:
                bodies.setdefault(str(paragraph["title"]), body)

    context = example.get("context") or metadata.get("context")
    for title, body in _context_title_to_body(context).items():
        if title in gold_titles and body:
            bodies.setdefault(title, body)

    return [(title, bodies[title]) for title in sorted(bodies)]


def make_rl_record(
    example: dict[str, Any],
    idx: int,
    data_source: str,
    split: str,
    require_budget: bool,
    max_budget: int,
    budget_template: str = "think_first",
) -> dict[str, Any]:
    """Return one RL parquet record.

    require_budget controls only the prompt wording, extra_info always
    carries gold_budget and gold_titles for the cap and grounding.
    """
    question = make_search_prefix(
        example["question"],
        require_budget=require_budget,
        max_budget=max_budget,
        budget_template=budget_template,
    )
    answer = extract_answer(example)
    extra_info = {
        "split": split,
        "index": idx,
        "gold_budget": derive_gold_budget(example, data_source),
        "gold_titles": extract_gold_titles(example, data_source),
    }

    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": question}],
        "ability": "fact-reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": {"target": answer},
        },
        "extra_info": extra_info,
    }


def cap_records_per_budget(
    records: list[dict[str, Any]], cap: int, seed: int = 42
) -> list[dict[str, Any]]:
    """Return records with at most ``cap`` rows per gold_budget value.

    Rebalances hop variance via a seeded per-group shuffle, kept in
    original order. A non-positive cap returns the input unchanged.

    Raises:
        ValueError: If any record lacks ``extra_info.gold_budget``.
    """
    if cap <= 0:
        return records

    import random

    by_budget: dict[int, list[int]] = {}
    for position, record in enumerate(records):
        extra = record.get("extra_info") or {}
        if "gold_budget" not in extra:
            raise ValueError(
                "cap_records_per_budget needs extra_info.gold_budget "
                "on every record"
            )
        by_budget.setdefault(int(extra["gold_budget"]), []).append(
            position
        )

    keep: set[int] = set()
    for budget in sorted(by_budget):
        positions = list(by_budget[budget])
        random.Random(seed + budget).shuffle(positions)
        keep.update(positions[:cap])

    return [r for i, r in enumerate(records) if i in keep]


def load_named_dataset(data_source: str):
    """Return the FlashRAG HuggingFace dataset for ``data_source``."""
    import datasets

    return datasets.load_dataset("RUC-NLPIR/FlashRAG_datasets", data_source)


def build_records(
    data_sources: Iterable[str],
    mode: str,
    split: str,
    require_budget: bool,
    max_budget: int,
    budget_template: str = "think_first",
) -> list[dict[str, Any]]:
    """Return RL records for every example in ``split``.

    Raises:
        ValueError: If ``split`` is missing for some dataset or
            ``mode`` is not in ``SUPPORTED_MODES``.
    """
    records = []
    for data_source in data_sources:
        dataset = load_named_dataset(data_source)
        if split not in dataset:
            available = list(dataset.keys())
            raise ValueError(
                f"Split '{split}' not found for dataset '{data_source}'. Available splits: {available}"
            )
        for idx, example in enumerate(dataset[split]):
            if mode == "rl":
                records.append(
                    make_rl_record(
                        example, idx, data_source, split,
                        require_budget, max_budget,
                        budget_template=budget_template,
                    )
                )
            else:
                raise ValueError(f"Unsupported mode: {mode}")
    return records


def main() -> None:
    """Write train/test parquets under ``--local_dir``."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/thesis_search")
    parser.add_argument("--data_sources", default="hotpotqa,2wikimultihopqa")
    parser.add_argument("--mode", choices=SUPPORTED_MODES, default="rl")
    parser.add_argument("--require_budget", action="store_true")
    parser.add_argument(
        "--budget_template",
        choices=BUDGET_TEMPLATES,
        default="think_first",
    )
    parser.add_argument("--max_budget", type=int, default=5)
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="dev")
    parser.add_argument(
        "--max_train_per_budget",
        type=int,
        default=0,
        help="cap train rows per gold_budget value (0 = off)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_sources = [
        source.strip() for source in args.data_sources.split(",") if source.strip()
    ]
    unsupported = sorted(set(data_sources) - set(SUPPORTED_DATASETS))
    if unsupported:
        raise ValueError(f"Unsupported thesis datasets: {unsupported}")

    os.makedirs(args.local_dir, exist_ok=True)
    train_records = build_records(
        data_sources, args.mode, args.train_split,
        args.require_budget, args.max_budget,
        budget_template=args.budget_template,
    )
    train_records = cap_records_per_budget(
        train_records, args.max_train_per_budget, args.seed
    )
    val_records = build_records(
        data_sources, args.mode, args.val_split,
        args.require_budget, args.max_budget,
        budget_template=args.budget_template,
    )
    import datasets

    datasets.Dataset.from_list(train_records).to_parquet(
        os.path.join(args.local_dir, "train.parquet")
    )
    datasets.Dataset.from_list(val_records).to_parquet(
        os.path.join(args.local_dir, "test.parquet")
    )


if __name__ == "__main__":
    main()
