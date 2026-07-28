"""Unit tests for the reward formula and <budget>k</budget> parser."""

import math
import unittest

from search_r1.budgeting import (
    BudgetRewardConfig,
    build_budget_mask,
    compute_budget_reward,
    compute_grounding_reward,
    curriculum_gamma,
    find_budget_digit_position,
    normalize_title,
    parse_budget_declaration,
    parse_retrieved_titles,
    select_answer_reward,
    should_force_search,
    title_recall,
    validate_cost_reward_config,
)


class BudgetingTest(unittest.TestCase):
    def test_parse_budget_declaration_accepts_bounded_integer(self):
        self.assertEqual(parse_budget_declaration("<budget>3</budget>"), 3)
        padded = "  <budget> 0 </budget>\n"
        self.assertEqual(parse_budget_declaration(padded), 0)
        self.assertEqual(parse_budget_declaration("<budget>5</budget>"), 5)

    def test_parse_budget_declaration_rejects_missing_or_out_of_range(self):
        self.assertIsNone(parse_budget_declaration("<think>search</think>"))
        self.assertIsNone(parse_budget_declaration("<budget>6</budget>"))
        self.assertIsNone(parse_budget_declaration("<budget>two</budget>"))

    def test_compute_budget_reward_matches_thesis_formula(self):
        score, parts = compute_budget_reward(
            answer_score=1.0,
            valid_search_calls=2,
            declared_budget=3,
            config=BudgetRewardConfig(gamma=0.01),
        )

        self.assertTrue(math.isclose(score, 0.99))
        self.assertEqual(parts["answer"], 1.0)
        self.assertEqual(parts["unused_budget_penalty"], 0.01)

    def test_compute_budget_reward_ignores_under_declaration(self):
        score, parts = compute_budget_reward(
            answer_score=0.0,
            valid_search_calls=3,
            declared_budget=1,
            config=BudgetRewardConfig(gamma=0.01),
        )

        self.assertTrue(math.isclose(score, 0.0))
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


class AnswerRewardTest(unittest.TestCase):
    def test_em_metric_returns_binary_em(self):
        self.assertEqual(select_answer_reward(1.0, 0.5, "em"), 1.0)
        self.assertEqual(select_answer_reward(0.0, 0.7, "em"), 0.0)

    def test_f1_metric_returns_f1(self):
        self.assertEqual(select_answer_reward(0.0, 0.7, "f1"), 0.7)
        self.assertEqual(select_answer_reward(1.0, 1.0, "f1"), 1.0)

    def test_em_f1_metric_averages(self):
        self.assertTrue(
            math.isclose(select_answer_reward(0.0, 0.6, "em_f1"), 0.3)
        )
        self.assertTrue(
            math.isclose(select_answer_reward(1.0, 1.0, "em_f1"), 1.0)
        )

    def test_unknown_metric_raises(self):
        with self.assertRaises(ValueError):
            select_answer_reward(1.0, 1.0, "bleu")


class DeclarationFloorTest(unittest.TestCase):
    def test_penalizes_under_declaration_toward_gold(self):
        score, parts = compute_budget_reward(
            answer_score=1.0,
            valid_search_calls=1,
            declared_budget=1,
            config=BudgetRewardConfig(gamma=0.01, delta=0.02),
            gold_budget=4,
        )

        self.assertTrue(math.isclose(score, 0.94))
        self.assertTrue(
            math.isclose(parts["under_declaration_penalty"], 0.06)
        )

    def test_no_penalty_when_delta_zero(self):
        _, parts = compute_budget_reward(
            answer_score=1.0,
            valid_search_calls=1,
            declared_budget=1,
            config=BudgetRewardConfig(),
            gold_budget=4,
        )

        self.assertEqual(parts["under_declaration_penalty"], 0.0)

    def test_no_penalty_when_gold_absent_or_met(self):
        _, parts_none = compute_budget_reward(
            answer_score=0.0,
            valid_search_calls=1,
            declared_budget=1,
            config=BudgetRewardConfig(delta=0.1),
            gold_budget=None,
        )
        _, parts_met = compute_budget_reward(
            answer_score=0.0,
            valid_search_calls=4,
            declared_budget=5,
            config=BudgetRewardConfig(delta=0.1),
            gold_budget=4,
        )

        self.assertEqual(parts_none["under_declaration_penalty"], 0.0)
        self.assertEqual(parts_met["under_declaration_penalty"], 0.0)


class BuildBudgetMaskTest(unittest.TestCase):
    # budget_ids stand in for tokenized "<budget>k</budget>"
    def test_marks_span_at_start_of_response(self):
        response_ids = [10, 3, 11, 40, 41]
        budget_ids = [10, 3, 11]
        self.assertEqual(
            build_budget_mask(response_ids, budget_ids),
            [1, 1, 1, 0, 0],
        )

    def test_marks_span_after_leading_tokens(self):
        response_ids = [99, 10, 3, 11, 40]
        budget_ids = [10, 3, 11]
        self.assertEqual(
            build_budget_mask(response_ids, budget_ids),
            [0, 1, 1, 1, 0],
        )

    def test_marks_only_first_occurrence(self):
        response_ids = [10, 3, 11, 10, 3, 11]
        budget_ids = [10, 3, 11]
        self.assertEqual(
            build_budget_mask(response_ids, budget_ids),
            [1, 1, 1, 0, 0, 0],
        )

    def test_all_zeros_when_declaration_absent(self):
        self.assertEqual(
            build_budget_mask([40, 41, 42], [10, 3, 11]),
            [0, 0, 0],
        )

    def test_all_zeros_when_budget_ids_empty(self):
        self.assertEqual(build_budget_mask([1, 2, 3], []), [0, 0, 0])


class MinSearchFloorTest(unittest.TestCase):
    def test_floor_forces_beyond_declared(self):
        # declared 1, used 1: no force normally, but a floor of 2 forces
        self.assertFalse(should_force_search(1, 1, True))
        self.assertTrue(should_force_search(1, 1, True, min_searches=2))

    def test_floor_forces_even_when_declared_zero(self):
        self.assertTrue(should_force_search(0, 0, True, min_searches=2))

    def test_floor_stops_once_reached(self):
        self.assertFalse(should_force_search(1, 2, True, min_searches=2))

    def test_no_force_when_inactive(self):
        self.assertFalse(should_force_search(1, 0, False, min_searches=2))

    def test_default_floor_preserves_declared_behavior(self):
        self.assertTrue(should_force_search(3, 1, True))
        self.assertFalse(should_force_search(0, 0, True))


class GroundingRewardTest(unittest.TestCase):
    def test_normalize_title_lowers_and_collapses_space(self):
        self.assertEqual(normalize_title("  The   Movie "), "the movie")

    def test_title_recall_counts_matched_gold_by_containment(self):
        self.assertEqual(title_recall(["a", "C", "D"], ["A", "B"]), 0.5)

    def test_title_recall_zero_when_no_gold(self):
        self.assertEqual(title_recall(["x"], []), 0.0)

    def test_parse_retrieved_titles_extracts_doc_titles(self):
        text = (
            "stuff <information>Doc 1(Title: Alpha Beta) body one\n"
            "Doc 2(Title: Gamma) body two\n</information> tail"
        )
        self.assertEqual(
            parse_retrieved_titles(text), ["Alpha Beta", "Gamma"]
        )

    def test_parse_retrieved_titles_empty_when_no_docs(self):
        self.assertEqual(parse_retrieved_titles("no docs here"), [])

    def test_compute_grounding_reward_scales_recall_by_lambda(self):
        self.assertTrue(
            math.isclose(
                compute_grounding_reward(
                    ["Alpha", "Zeta"], ["Alpha", "Beta"], 0.3
                ),
                0.15,
            )
        )

    def test_compute_grounding_reward_zero_without_gold(self):
        self.assertEqual(
            compute_grounding_reward(["Alpha"], [], 0.3), 0.0
        )

    def test_compute_budget_reward_adds_grounding_term(self):
        cfg = BudgetRewardConfig(gamma=0.01)
        base, _ = compute_budget_reward(
            answer_score=0.0,
            valid_search_calls=2,
            declared_budget=2,
            config=cfg,
        )
        withg, parts = compute_budget_reward(
            answer_score=0.0,
            valid_search_calls=2,
            declared_budget=2,
            config=cfg,
            grounding_reward=0.3,
        )
        self.assertTrue(math.isclose(withg - base, 0.3))
        self.assertEqual(parts["grounding_reward"], 0.3)

    def test_correct_beats_incorrect_when_lambda_bounded(self):
        cfg = BudgetRewardConfig(gamma=0.01)
        worst_correct, _ = compute_budget_reward(
            answer_score=1.0,
            valid_search_calls=5,
            declared_budget=5,
            config=cfg,
            grounding_reward=0.0,
        )
        best_incorrect, _ = compute_budget_reward(
            answer_score=0.0,
            valid_search_calls=0,
            declared_budget=0,
            config=cfg,
            grounding_reward=0.3,
        )
        self.assertGreater(worst_correct, best_incorrect)


class FindBudgetDigitPositionTest(unittest.TestCase):
    # token ids: 10..19 stand in for digits "0".."9"
    DIGITS = list(range(10, 20))

    def test_finds_digit_inside_budget_span(self):
        ids = [50, 51, 12, 52, 99]
        mask = [0, 1, 1, 1, 0]
        self.assertEqual(
            find_budget_digit_position(ids, mask, self.DIGITS), 2
        )

    def test_ignores_digit_outside_span(self):
        ids = [12, 51, 52]
        mask = [0, 1, 1]
        self.assertIsNone(
            find_budget_digit_position(ids, mask, self.DIGITS)
        )

    def test_none_when_span_absent(self):
        ids = [50, 51, 52]
        mask = [0, 0, 0]
        self.assertIsNone(
            find_budget_digit_position(ids, mask, self.DIGITS)
        )


class ValidateCostRewardConfigTest(unittest.TestCase):
    def test_accepts_disabled_config_whatever_the_coeff(self):
        validate_cost_reward_config(None)
        validate_cost_reward_config({"enabled": False})
        validate_cost_reward_config(
            {"enabled": False, "cost_in_advantage": 0.0}
        )

    def test_accepts_enabled_with_positive_coeff(self):
        validate_cost_reward_config(
            {"enabled": True, "cost_in_advantage": 0.5}
        )

    def test_rejects_enabled_without_a_cost_signal(self):
        for cfg in ({"enabled": True}, {"enabled": True,
                                        "cost_in_advantage": 0.0}):
            with self.assertRaises(ValueError):
                validate_cost_reward_config(cfg)


class CostInAdvantageRewardTest(unittest.TestCase):
    """The scalar reward carries the couplings, never the call cost.

    Absolute retrieval and token cost belong to the advantage, so the
    reward must vary only with the planning couplings: gamma (unused
    budget) and delta (declaration floor toward gold).
    """

    def test_reward_is_flat_in_calls_at_a_fixed_declaration(self):
        cfg = BudgetRewardConfig(gamma=0.0, delta=0.0)
        common = dict(
            answer_score=1.0, declared_budget=0, config=cfg
        )
        cheap, _ = compute_budget_reward(valid_search_calls=0, **common)
        pricey, _ = compute_budget_reward(valid_search_calls=5, **common)
        self.assertEqual(cheap, pricey)

    def test_couplings_still_apply(self):
        cfg = BudgetRewardConfig(gamma=0.01, delta=0.02)
        score, _ = compute_budget_reward(
            answer_score=1.0,
            valid_search_calls=1,
            declared_budget=4,
            config=cfg,
            gold_budget=2,
            grounding_reward=0.25,
        )
        # gamma*max(0, 4-1) = .03, delta idle since declared > gold
        self.assertTrue(math.isclose(score, 1.0 + 0.25 - 0.03))

    def test_declare_down_escape_costs_more_than_gamma_saves(self):
        # delta > gamma, so under-declaring saves gamma but pays delta
        cfg = BudgetRewardConfig(gamma=0.01, delta=0.02)
        common = dict(
            answer_score=0.0,
            valid_search_calls=1,
            config=cfg,
            gold_budget=3,
            grounding_reward=0.0,
        )
        at_gold, _ = compute_budget_reward(declared_budget=3, **common)
        below_gold, _ = compute_budget_reward(
            declared_budget=2, **common
        )
        self.assertGreater(at_gold, below_gold)


if __name__ == "__main__":
    unittest.main()
