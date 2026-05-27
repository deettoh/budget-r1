"""SQuAD-style QA scoring, normalization, EM, F1, and CES.

MRC and TTC aggregate in the trainer, they need rollout state not the
answer string, so they live in RayPPOTrainer._validate.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable


_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", flags=re.DOTALL)
_PUNCTUATION = set(string.punctuation)


def normalize_answer(text: str) -> str:
    """Lowercase; strip articles, punctuation, and extra whitespace."""
    text = text.lower()
    text = _ARTICLES.sub(" ", text)
    text = "".join(ch for ch in text if ch not in _PUNCTUATION)
    return " ".join(text.split())


def extract_solution(solution_str: str) -> str | None:
    """Return the last ``<answer>...</answer>`` span, or None.

    The prompt carries a literal example span, so the real answer is
    the last one. None when fewer than two spans are present.
    """
    matches = list(_ANSWER_TAG.finditer(solution_str))
    if len(matches) <= 1:
        return None
    return matches[-1].group(1).strip()


def _as_gold_list(ground_truth) -> list[str]:
    """Coerce the ground-truth field into a list of strings."""
    if isinstance(ground_truth, dict):
        target = ground_truth.get("target", "")
    else:
        target = ground_truth
    if target is None:
        return []
    if isinstance(target, str):
        return [target]
    if isinstance(target, Iterable):
        return [str(t) for t in target]
    return [str(target)]


def em_score(prediction: str, golden_answers) -> float:
    """Return 1.0 iff any normalized gold matches the prediction."""
    norm_pred = normalize_answer(prediction)
    for gold in _as_gold_list(golden_answers):
        if normalize_answer(gold) == norm_pred:
            return 1.0
    return 0.0


def _f1_against_single_gold(pred_tokens: list[str], gold_tokens: list[str]) -> float:
    """Return token-overlap F1 against one gold answer.

    SQuAD empty-input convention, 1.0 only when both sides are empty.
    """
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2.0 * precision * recall / (precision + recall)


def f1_score(prediction: str, golden_answers) -> float:
    """Return max token-level F1 across the gold answer strings."""
    pred_tokens = normalize_answer(prediction).split()
    best = 0.0
    for gold in _as_gold_list(golden_answers):
        gold_tokens = normalize_answer(gold).split()
        score = _f1_against_single_gold(pred_tokens, gold_tokens)
        if score > best:
            best = score
    return best


def compute_qa_metrics(solution_str: str, ground_truth) -> dict:
    """Return {em, f1, has_answer} for a single rollout output.

    has_answer False when no span parses, then em and f1 are 0.
    """
    answer = extract_solution(solution_str)
    if answer is None:
        return {"em": 0.0, "f1": 0.0, "has_answer": False}
    return {
        "em": em_score(answer, ground_truth),
        "f1": f1_score(answer, ground_truth),
        "has_answer": True,
    }


def cost_efficiency_score(f1: float, total_token_cost: float) -> float:
    """Return CES = F1 * 1000 / TTC, or 0 when TTC <= 0.

    The 1000 scaling reads CES as F1 points per kilotoken.
    """
    if total_token_cost <= 0:
        return 0.0
    return float(f1) * 1000.0 / float(total_token_cost)
