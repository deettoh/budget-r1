"""Thesis cost-penalized GRPO reward.

Implements ``R = R_answer - alpha*N - beta*T - gamma*max(0, k-N)``.
Single source of truth for alpha/beta/gamma. The trainer pulls
coefficients from Hydra and calls in. Math is not duplicated.
"""

import re
from dataclasses import dataclass
from typing import Optional


_BUDGET_PATTERN = re.compile(r"^\s*<budget>\s*(\d+)\s*</budget>\s*", re.DOTALL)


@dataclass(frozen=True)
class BudgetRewardConfig:
    """Frozen alpha/beta/gamma, ``gamma < alpha`` enforced at config layer."""

    alpha: float = 0.05
    beta: float = 0.0001
    gamma: float = 0.01


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


def should_force_search(
    declared_budget: int, search_count: int, force_active: bool
) -> bool:
    """Return True if an early answer must be blocked to force search.

    Bootstrap-only scaffolding for the budget planner: while forcing is
    active, a sample that declared ``k >= 1`` may not answer until it has
    used its declared calls, so declaring k binds the cap and reopens the
    k->N->R_answer gradient. Pure and config-free; callers gate it on
    ``enable_budget_planner`` and the train-vs-eval split.
    """
    return force_active and declared_budget >= 1 and search_count < declared_budget


def curriculum_gamma(base_gamma: float, force_active: bool) -> float:
    """Return 0 while forcing is active, else ``base_gamma``.

    Suppresses the unused-budget penalty only in the forcing window.
    """
    return 0.0 if force_active else base_gamma


def compute_budget_reward(
    answer_score: float,
    valid_search_calls: int,
    generated_tokens: int,
    retrieved_tokens: int,
    declared_budget: int,
    config: BudgetRewardConfig,
) -> tuple[float, dict[str, float]]:
    """Return ``(score, parts)`` for one rollout.

    ``parts`` carries each penalty component for logging.
    """
    total_tokens = generated_tokens + retrieved_tokens
    retrieval_penalty = config.alpha * valid_search_calls
    token_penalty = config.beta * total_tokens
    unused_budget_penalty = config.gamma * max(
        0, declared_budget - valid_search_calls
    )
    score = (
        answer_score
        - retrieval_penalty
        - token_penalty
        - unused_budget_penalty
    )

    return score, {
        "answer": answer_score,
        "retrieval_penalty": retrieval_penalty,
        "token_penalty": token_penalty,
        "unused_budget_penalty": unused_budget_penalty,
        "total_tokens": total_tokens,
    }
