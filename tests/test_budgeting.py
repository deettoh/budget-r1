"""Unit tests for the reward formula and <budget>k</budget> parser."""

import math
import unittest

from search_r1.budgeting import (
    BudgetRewardConfig,
    compute_budget_reward,
    curriculum_gamma,
    parse_budget_declaration,
    should_force_search,
)


class BudgetingTest(unittest.TestCase):
    def test_parse_budget_declaration_accepts_bounded_integer(self):
        self.assertEqual(parse_budget_declaration("<budget>3</budget>"), 3)
        self.assertEqual(parse_budget_declaration("  <budget> 0 </budget>\n"), 0)
        self.assertEqual(parse_budget_declaration("<budget>5</budget>"), 5)

    def test_parse_budget_declaration_rejects_missing_or_out_of_range_budget(self):
        self.assertIsNone(parse_budget_declaration("<think>search first</think>"))
        self.assertIsNone(parse_budget_declaration("<budget>6</budget>"))
        self.assertIsNone(parse_budget_declaration("<budget>two</budget>"))

    def test_compute_budget_reward_matches_thesis_formula(self):
        score, parts = compute_budget_reward(
            answer_score=1.0,
            valid_search_calls=2,
            generated_tokens=400,
            retrieved_tokens=700,
            declared_budget=3,
            config=BudgetRewardConfig(alpha=0.05, beta=0.0001, gamma=0.01),
        )

        self.assertTrue(math.isclose(score, 0.78))
        self.assertEqual(parts["answer"], 1.0)
        self.assertEqual(parts["retrieval_penalty"], 0.10)
        self.assertEqual(parts["token_penalty"], 0.11)
        self.assertEqual(parts["unused_budget_penalty"], 0.01)

    def test_compute_budget_reward_does_not_penalize_under_declaration_directly(self):
        score, parts = compute_budget_reward(
            answer_score=0.0,
            valid_search_calls=3,
            generated_tokens=100,
            retrieved_tokens=200,
            declared_budget=1,
            config=BudgetRewardConfig(alpha=0.05, beta=0.0001, gamma=0.01),
        )

        self.assertTrue(math.isclose(score, -0.18))
        self.assertEqual(parts["unused_budget_penalty"], 0.0)


class ForceSearchTest(unittest.TestCase):
    def test_no_force_when_bootstrap_inactive(self):
        for declared in (-1, 0, 3):
            self.assertFalse(should_force_search(declared, 0, False))

    def test_no_force_without_a_declaration(self):
        self.assertFalse(should_force_search(-1, 0, True))

    def test_no_force_when_declared_zero(self):
        self.assertFalse(should_force_search(0, 0, True))

    def test_force_until_declared_calls_are_used(self):
        self.assertTrue(should_force_search(3, 0, True))
        self.assertTrue(should_force_search(3, 2, True))

    def test_no_force_once_declared_budget_reached(self):
        self.assertFalse(should_force_search(3, 3, True))
        self.assertFalse(should_force_search(3, 4, True))


class CurriculumGammaTest(unittest.TestCase):
    def test_gamma_passes_through_when_not_forcing(self):
        self.assertEqual(curriculum_gamma(0.01, False), 0.01)

    def test_gamma_zeroed_while_forcing(self):
        self.assertEqual(curriculum_gamma(0.01, True), 0.0)


if __name__ == "__main__":
    unittest.main()
