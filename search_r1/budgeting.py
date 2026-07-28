"""Thesis budget reward.

R = R_answer + lam*G - gamma*max(0, k-N) - delta*max(0, gold_k-k),
where G is gold-passage grounding recall. The per-call retrieval
cost is subtracted from the GRPO advantage in ``core_algos``.
"""

import re
from dataclasses import dataclass
from typing import Optional, Sequence


_BUDGET_PATTERN = re.compile(r"^\s*<budget>\s*(\d+)\s*</budget>\s*", re.DOTALL)
_TITLE_PATTERN = re.compile(r"Doc\s+\d+\(Title:\s*(.*?)\)")
_WHITESPACE = re.compile(r"\s+")


def normalize_title(title: str) -> str:
    """Lowercase and collapse whitespace for lenient title matching."""
    return _WHITESPACE.sub(" ", title.lower()).strip()


def title_recall(
    retrieved_titles: Sequence[str], gold_titles: Sequence[str]
) -> float:
    """Return the fraction of gold titles matched in retrieved titles.

    Matches on equality or substring either way after normalization.
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


def parse_retrieved_titles(text: str) -> list[str]:
    """Return the titles of the ``Doc i(Title: ...)`` blocks."""
    return [m.strip() for m in _TITLE_PATTERN.findall(text or "")]


def compute_grounding_reward(
    retrieved_titles: Sequence[str],
    gold_titles: Sequence[str],
    lam: float,
) -> float:
    """Return lam times gold-passage recall, answer-independent."""
    return lam * title_recall(retrieved_titles, gold_titles)


@dataclass(frozen=True)
class BudgetRewardConfig:
    """Frozen declaration-coupling coefficients.

    gamma charges budget declared and left unused, delta charges
    declaring below the gold budget.
    """

    gamma: float = 0.01
    delta: float = 0.0


def validate_cost_reward_config(config: Optional[dict]) -> None:
    """Raise if the cost reward is on but charges no retrieval cost."""
    cfg = config or {}
    if not cfg.get("enabled", False):
        return
    if float(cfg.get("cost_in_advantage", 0.0)) <= 0:
        raise ValueError(
            "cost_reward.enabled=true requires "
            "cost_reward.cost_in_advantage>0, the per-call cost is "
            "applied only in the GRPO advantage"
        )


def parse_budget_declaration(text: str, max_budget: int = 5) -> Optional[int]:
    """Return k if ``text`` opens with a valid ``<budget>k</budget>``.

    None when malformed or k outside [0, max_budget].
    """
    match = _BUDGET_PATTERN.match(text or "")
    if match is None:
        return None

    budget = int(match.group(1))
    if 0 <= budget <= max_budget:
        return budget
    return None


def find_budget_digit_position(
    response_ids: Sequence[int],
    budget_mask: Sequence[int],
    digit_token_ids: Sequence[int],
) -> Optional[int]:
    """Return the index of the declared digit inside the budget span.

    Used by the budget-CE aux loss. None when the span has no digit.
    """
    digit_set = set(digit_token_ids)
    for position, (token, masked) in enumerate(
        zip(response_ids, budget_mask)
    ):
        if masked and token in digit_set:
            return position
    return None


def build_budget_mask(
    response_ids: Sequence[int], budget_ids: Sequence[int]
) -> list[int]:
    """Return a 0/1 mask over the first <budget>k</budget> span.

    All zeros when the declaration is empty or absent.
    """
    mask = [0] * len(response_ids)
    span = len(budget_ids)
    if span == 0 or span > len(response_ids):
        return mask

    target = list(budget_ids)
    for start in range(len(response_ids) - span + 1):
        if list(response_ids[start:start + span]) == target:
            for offset in range(span):
                mask[start + offset] = 1
            return mask
    return mask


def select_answer_reward(em: float, f1: float, metric: str = "em") -> float:
    """Return the answer reward under metric em, f1, or em_f1.

    em keeps parity with the Search-R1 baseline.
    """
    if metric == "em":
        return float(em)
    if metric == "f1":
        return float(f1)
    if metric == "em_f1":
        return 0.5 * float(em) + 0.5 * float(f1)
    raise ValueError(f"unknown answer_metric: {metric!r}")


def should_force_search(
    declared_budget: int,
    search_count: int,
    force_active: bool,
    min_searches: int = 0,
) -> bool:
    """Return True if an early answer must be blocked to force search.

    Bootstrap scaffolding, off in every reported run.
    """
    target = max(declared_budget if declared_budget >= 0 else 0, min_searches)
    return force_active and target >= 1 and search_count < target


def curriculum_gamma(base_gamma: float, force_active: bool) -> float:
    """Return 0 while forcing is active, else ``base_gamma``.

    Suppresses the unused-budget penalty only in the forcing window.
    """
    return 0.0 if force_active else base_gamma


def compute_budget_reward(
    answer_score: float,
    valid_search_calls: int,
    declared_budget: int,
    config: BudgetRewardConfig,
    gold_budget: Optional[int] = None,
    grounding_reward: float = 0.0,
) -> tuple[float, dict[str, float]]:
    """Return ``(score, parts)`` for one rollout.

    parts contains each component for logging. The under-declaration
    penalty applies only when gold_budget is given.
    """
    unused_budget_penalty = config.gamma * max(
        0, declared_budget - valid_search_calls
    )
    under_declaration_penalty = 0.0
    if gold_budget is not None:
        under_declaration_penalty = config.delta * max(
            0, gold_budget - declared_budget
        )
    score = (
        answer_score
        + grounding_reward
        - unused_budget_penalty
        - under_declaration_penalty
    )

    return score, {
        "answer": answer_score,
        "grounding_reward": grounding_reward,
        "unused_budget_penalty": unused_budget_penalty,
        "under_declaration_penalty": under_declaration_penalty,
    }
