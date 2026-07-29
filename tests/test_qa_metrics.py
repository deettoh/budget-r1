"""Unit tests for verl.utils.reward_score.qa_metrics.

Covers answer normalization, answer-span extraction, EM, token-level
F1 and CES. MRC and TTC are plain aggregates with no logic to
unit-test here.

Typical usage example:

  python3 -m unittest tests.test_qa_metrics
"""

from __future__ import annotations

import importlib.util
import pathlib
import unittest


# import directly, verl/__init__ drags in numpy/torch
_HERE = pathlib.Path(__file__).resolve().parent
_SCORE_DIR = _HERE.parent / "verl" / "utils" / "reward_score"
_QA_METRICS_PATH = _SCORE_DIR / "qa_metrics.py"
_spec = importlib.util.spec_from_file_location("qa_metrics", _QA_METRICS_PATH)
qa_metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qa_metrics)


# mirrors the literal example in the rollout prompt
_PROMPT_EXAMPLE = "<answer> Beijing </answer>"


def _wrap(answer: str) -> str:
    """Return a decoded solution holding the example, then answer."""
    return f"... {_PROMPT_EXAMPLE} ... <answer>{answer}</answer>"


class NormalizeAnswerTest(unittest.TestCase):
    def test_lowercases_and_strips_punctuation(self):
        self.assertEqual(qa_metrics.normalize_answer("  The Eiffel Tower!  "),
                         "eiffel tower")

    def test_removes_articles_only_at_word_boundary(self):
        # "Thea" keeps its "a" articles are word-bound
        self.assertEqual(qa_metrics.normalize_answer("Thea"), "thea")


class ExtractSolutionTest(unittest.TestCase):
    def test_returns_last_answer_span(self):
        solution = _wrap("Paris")
        self.assertEqual(qa_metrics.extract_solution(solution), "Paris")

    def test_returns_none_when_only_prompt_example(self):
        # only the example <answer> tag, no real model answer
        text = f"... {_PROMPT_EXAMPLE} ..."
        self.assertIsNone(qa_metrics.extract_solution(text))

    def test_returns_none_when_no_answer_tag(self):
        self.assertIsNone(qa_metrics.extract_solution("no tags here"))


class EmScoreTest(unittest.TestCase):
    def test_exact_match_after_normalization(self):
        self.assertEqual(qa_metrics.em_score("Paris", "paris"), 1.0)
        self.assertEqual(qa_metrics.em_score("The Eiffel Tower",
                                             ["eiffel tower"]), 1.0)

    def test_no_match(self):
        self.assertEqual(qa_metrics.em_score("Paris", "London"), 0.0)

    def test_accepts_dict_ground_truth(self):
        gold = {"target": ["paris", "lyon"]}
        self.assertEqual(qa_metrics.em_score("Paris", gold), 1.0)


class F1ScoreTest(unittest.TestCase):
    def test_full_overlap_returns_one(self):
        self.assertAlmostEqual(qa_metrics.f1_score("Paris", "paris"), 1.0)

    def test_partial_overlap_strict_between_zero_and_one(self):
        score = qa_metrics.f1_score("Eiffel Tower Paris", "eiffel tower")
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_no_overlap_returns_zero(self):
        self.assertEqual(qa_metrics.f1_score("Paris", "Madrid"), 0.0)

    def test_takes_max_across_gold_list(self):
        # worse gold first, perfect gold second
        self.assertAlmostEqual(
            qa_metrics.f1_score("Paris", ["Madrid", "paris"]),
            1.0,
        )


class ComputeQaMetricsTest(unittest.TestCase):
    def test_returns_zeros_when_no_answer_extracted(self):
        result = qa_metrics.compute_qa_metrics(_PROMPT_EXAMPLE, "paris")
        self.assertEqual(result, {"em": 0.0, "f1": 0.0, "has_answer": False})

    def test_returns_em_and_f1_when_answer_present(self):
        gold = {"target": "paris"}
        result = qa_metrics.compute_qa_metrics(_wrap("Paris"), gold)
        self.assertEqual(result["has_answer"], True)
        self.assertEqual(result["em"], 1.0)
        self.assertAlmostEqual(result["f1"], 1.0)


class CesTest(unittest.TestCase):
    def test_ces_formula(self):
        # CES = F1*1000/TTC, here 0.5*1000/1000 = 0.5
        ces = qa_metrics.cost_efficiency_score(0.5, 1000)
        self.assertAlmostEqual(ces, 0.5)

    def test_ces_zero_ttc_returns_zero(self):
        self.assertEqual(qa_metrics.cost_efficiency_score(0.9, 0), 0.0)

    def test_ces_negative_ttc_returns_zero(self):
        self.assertEqual(qa_metrics.cost_efficiency_score(0.9, -5), 0.0)


if __name__ == "__main__":
    unittest.main()
