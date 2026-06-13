"""Unit tests for the baseline metric aggregation."""

import importlib.util
import os
import unittest

_SPEC = importlib.util.spec_from_file_location(
    "analyze_baselines",
    os.path.join(
        os.path.dirname(__file__), "..", "scripts",
        "analyze_baselines.py"),
)
analyze_baselines = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(analyze_baselines)

compute_metrics = analyze_baselines.compute_metrics


def _rec(em, f1, ha, gen, ret, calls, source="hotpotqa"):
    return {
        "data_source": source,
        "em": em,
        "f1": f1,
        "has_answer": ha,
        "generated_tokens": gen,
        "retrieved_tokens": ret,
        "valid_search_calls": calls,
        "declared_budget": -1,
    }


class ComputeMetricsTest(unittest.TestCase):
    def test_basic_aggregates(self):
        recs = [
            _rec(1, 1.0, 1, 100, 300, 2),
            _rec(0, 0.0, 1, 100, 100, 0),
        ]
        m = compute_metrics(recs)
        self.assertEqual(m["n"], 2)
        self.assertAlmostEqual(m["em"], 0.5)
        self.assertAlmostEqual(m["f1"], 0.5)
        self.assertAlmostEqual(m["mrc"], 1.0)
        self.assertAlmostEqual(m["ttc"], 300.0)
        self.assertAlmostEqual(m["ces"], 0.5 * 1000 / 300.0)

    def test_mrc_distribution_buckets_4plus(self):
        recs = [_rec(0, 0.0, 1, 10, 0, c) for c in (0, 1, 2, 3, 5, 9)]
        m = compute_metrics(recs)
        self.assertEqual(
            m["mrc_dist"], {0: 1, 1: 1, 2: 1, 3: 1, 4: 2})

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            compute_metrics([])


if __name__ == "__main__":
    unittest.main()
