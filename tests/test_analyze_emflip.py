"""Unit tests for the EM-flip diagnostic aggregation.

Covers splitting records by retrieval-call count and the win, loss
and tie counts between two conditions.

Typical usage example:

  python3 -m unittest tests.test_analyze_emflip
"""

import importlib.util
import os
import unittest

from search_r1.budgeting import BudgetRewardConfig

_SPEC = importlib.util.spec_from_file_location(
    "analyze_emflip",
    os.path.join(
        os.path.dirname(__file__), "..", "scripts",
        "analyze_emflip.py"),
)
analyze_emflip = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(analyze_emflip)

split_by_calls = analyze_emflip.split_by_calls
compute_emflip = analyze_emflip.compute_emflip


def _rec(em, calls, retrieved=300, f1=None, source="hotpotqa"):
    return {
        "data_source": source,
        "em": em,
        "f1": em if f1 is None else f1,
        "has_answer": 1,
        "generated_tokens": 100,
        "retrieved_tokens": retrieved,
        "valid_search_calls": calls,
        "declared_budget": -1,
    }


class SplitByCallsTest(unittest.TestCase):
    def test_partitions_single_and_multi(self):
        recs = [_rec(0, 0), _rec(1, 1), _rec(0, 1), _rec(1, 2),
                _rec(1, 4)]
        single, multi = split_by_calls(recs)
        # zero-call samples belong to neither band
        self.assertEqual([r["valid_search_calls"] for r in single],
                         [1, 1])
        self.assertEqual([r["valid_search_calls"] for r in multi],
                         [2, 4])


class ComputeEmflipTest(unittest.TestCase):
    def setUp(self):
        self.cfg = BudgetRewardConfig()

    def test_delta_is_multi_minus_single(self):
        recs = [_rec(0, 1), _rec(0, 1), _rec(1, 2), _rec(1, 3)]
        m = compute_emflip(recs, self.cfg)
        self.assertEqual(m["n_single"], 2)
        self.assertEqual(m["n_multi"], 2)
        self.assertAlmostEqual(m["em_single"], 0.0)
        self.assertAlmostEqual(m["em_multi"], 1.0)
        self.assertAlmostEqual(m["em_delta"], 1.0)

    def test_break_even_from_alpha_beta_and_tokens(self):
        # avg tokens/call = (400+600)/2 = 500 -> 0.05 + 1e-4*500 = 0.10
        recs = [_rec(0, 1, retrieved=400), _rec(1, 1, retrieved=600)]
        m = compute_emflip(recs, self.cfg)
        self.assertAlmostEqual(m["break_even"], 0.10)

    def test_empty_band_does_not_crash(self):
        recs = [_rec(1, 1), _rec(0, 1)]
        m = compute_emflip(recs, self.cfg)
        self.assertEqual(m["n_multi"], 0)
        self.assertAlmostEqual(m["em_multi"], 0.0)


if __name__ == "__main__":
    unittest.main()
