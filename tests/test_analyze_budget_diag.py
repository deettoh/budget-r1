"""Unit tests for the item-1 budget-prompt diagnostic aggregation."""

import importlib.util
import os
import unittest

_SPEC = importlib.util.spec_from_file_location(
    "analyze_budget_diag",
    os.path.join(
        os.path.dirname(__file__), "..", "scripts",
        "analyze_budget_diag.py"),
)
diag = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(diag)


def _rec(f1, declared, calls=1, blocked=0, source="hotpotqa",
         em=None, with_blocked=True):
    rec = {
        "data_source": source,
        "em": f1 if em is None else em,
        "f1": f1,
        "has_answer": 1,
        "generated_tokens": 100,
        "retrieved_tokens": 300,
        "valid_search_calls": calls,
        "declared_budget": declared,
    }
    if with_blocked:
        rec["blocked_search_calls"] = blocked
    return rec


class SplitWellFormedTest(unittest.TestCase):
    def test_splits_on_declared_budget_sign(self):
        recs = [_rec(0.5, 2), _rec(0.1, -1), _rec(0.9, 0)]
        well, mal = diag.split_well_formed(recs)
        self.assertEqual([r["declared_budget"] for r in well], [2, 0])
        self.assertEqual([r["declared_budget"] for r in mal], [-1])


class DeclaredKDistributionTest(unittest.TestCase):
    def test_counts_per_k_over_well_formed_with_zeros(self):
        recs = [_rec(0.5, 2), _rec(0.5, 2), _rec(0.5, 0), _rec(0.5, -1)]
        dist = diag.declared_k_distribution(recs)
        self.assertEqual(dist[2], 2)
        self.assertEqual(dist[0], 1)
        self.assertEqual(dist[5], 0)
        # malformed excluded from the k-distribution
        self.assertEqual(sum(dist.values()), 3)


class F1ByKTest(unittest.TestCase):
    def test_mean_f1_per_declared_k(self):
        recs = [_rec(0.2, 2), _rec(0.4, 2), _rec(0.9, 0)]
        by_k = diag.f1_by_k(recs)
        self.assertAlmostEqual(by_k[2], 0.3)
        self.assertAlmostEqual(by_k[0], 0.9)
        self.assertNotIn(1, by_k)  # only observed k appear


class ComputeAxesTest(unittest.TestCase):
    def test_format_axis_and_valid_calls(self):
        recs = [_rec(0.8, 2, calls=2), _rec(0.6, 0, calls=0),
                _rec(0.1, -1, calls=0)]
        m = diag.compute_axes(recs)
        self.assertEqual(m["n"], 3)
        self.assertAlmostEqual(m["well_formed_rate"], 2 / 3)
        self.assertAlmostEqual(m["f1_well_formed"], 0.7)
        self.assertAlmostEqual(m["f1_malformed"], 0.1)
        self.assertAlmostEqual(m["f1_gap"], 0.6)
        self.assertAlmostEqual(m["mean_valid_calls"], 2 / 3)

    def test_blocked_conditioned_on_well_formed(self):
        recs = [_rec(0.8, 2, blocked=0), _rec(0.6, 3, blocked=2),
                _rec(0.1, -1, blocked=5)]
        m = diag.compute_axes(recs)
        # malformed sample's blocked count is excluded
        self.assertAlmostEqual(m["blocked_rate_wf"], 0.5)
        self.assertAlmostEqual(m["mean_blocked_wf"], 1.0)

    def test_blocked_none_when_field_absent(self):
        recs = [_rec(0.8, 2, with_blocked=False),
                _rec(0.1, -1, with_blocked=False)]
        m = diag.compute_axes(recs)
        self.assertIsNone(m["blocked_rate_wf"])
        self.assertIsNone(m["mean_blocked_wf"])

    def test_empty_records_raise(self):
        with self.assertRaises(ValueError):
            diag.compute_axes([])


class HasBlockedFieldTest(unittest.TestCase):
    def test_detects_presence(self):
        self.assertTrue(diag.has_blocked_field([_rec(0.5, 2)]))
        self.assertFalse(
            diag.has_blocked_field([_rec(0.5, 2, with_blocked=False)])
        )


class NativeF1BySourceTest(unittest.TestCase):
    def test_groups_native_f1(self):
        native = [_rec(0.4, -1, source="2wikimultihopqa"),
                  _rec(0.6, -1, source="2wikimultihopqa"),
                  _rec(0.3, -1, source="hotpotqa")]
        by_src = diag.native_f1_by_source(native)
        self.assertAlmostEqual(by_src["2wikimultihopqa"], 0.5)
        self.assertAlmostEqual(by_src["hotpotqa"], 0.3)


if __name__ == "__main__":
    unittest.main()
