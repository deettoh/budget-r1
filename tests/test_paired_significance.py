"""Unit tests for the paired significance tests over val dumps."""

import importlib.util
import os
import unittest

_SPEC = importlib.util.spec_from_file_location(
    "paired_significance",
    os.path.join(
        os.path.dirname(__file__), "..", "scripts",
        "paired_significance.py"),
)
sig = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sig)


def _rec(index, em=0.0, f1=0.0, calls=1, gen=100, ret=200,
         source="hotpotqa"):
    return {
        "data_source": source,
        "index": index,
        "em": em,
        "f1": f1,
        "valid_search_calls": calls,
        "generated_tokens": gen,
        "retrieved_tokens": ret,
    }


class AlignDumpsTest(unittest.TestCase):
    def test_keeps_only_shared_keys_in_matching_order(self):
        # Arrange
        a = [_rec(1), _rec(2), _rec(3)]
        b = [_rec(3), _rec(1), _rec(9)]

        # Act
        left, right = sig.align_dumps(a, b)

        # Assert
        self.assertEqual([r["index"] for r in left], [1, 3])
        self.assertEqual([r["index"] for r in right], [1, 3])

    def test_same_index_different_source_is_not_paired(self):
        a = [_rec(1, source="hotpotqa")]
        b = [_rec(1, source="musique")]

        left, right = sig.align_dumps(a, b)

        self.assertEqual(left, [])
        self.assertEqual(right, [])

    def test_raises_on_duplicate_key(self):
        a = [_rec(1), _rec(1)]
        b = [_rec(1)]

        with self.assertRaises(ValueError):
            sig.align_dumps(a, b)

    def test_raises_when_index_missing(self):
        a = [{"data_source": "hotpotqa", "em": 1.0}]
        b = [_rec(1)]

        with self.assertRaises(ValueError):
            sig.align_dumps(a, b)


class ColumnsTest(unittest.TestCase):
    def test_ttc_is_generated_plus_retrieved(self):
        cols = sig.to_columns([_rec(1, gen=300, ret=900)])

        self.assertAlmostEqual(float(cols["ttc"][0]), 1200.0)

    def test_mrc_aliases_valid_search_calls(self):
        cols = sig.to_columns([_rec(1, calls=3)])

        self.assertAlmostEqual(float(cols["mrc"][0]), 3.0)

    def test_ces_uses_ratio_of_means_not_mean_of_ratios(self):
        # thesis CES 0.5*1000/300 = 1.667, per-question averages 1.0
        records = [
            _rec(1, f1=1.0, gen=100, ret=400),
            _rec(2, f1=0.0, gen=50, ret=50),
        ]
        cols = sig.to_columns(records)

        # Act
        value = sig.ces_of(cols)

        # Assert
        self.assertAlmostEqual(value, 5.0 / 3.0)
        self.assertNotAlmostEqual(value, 1.0)


class PairedBootstrapTest(unittest.TestCase):
    def test_identical_inputs_give_zero_diff_and_ci_covering_zero(self):
        a = [_rec(i, f1=0.3 + 0.001 * i) for i in range(64)]

        result = sig.paired_bootstrap_ci(
            a, list(a), sig.mean_of("f1"), n_resamples=200, seed=42
        )

        self.assertAlmostEqual(result.diff, 0.0)
        self.assertLessEqual(result.lo, 0.0)
        self.assertGreaterEqual(result.hi, 0.0)
        self.assertFalse(result.is_significant)

    def test_constant_uplift_is_significant(self):
        a = [_rec(i, f1=0.5) for i in range(64)]
        b = [_rec(i, f1=0.2) for i in range(64)]

        result = sig.paired_bootstrap_ci(
            a, b, sig.mean_of("f1"), n_resamples=200, seed=42
        )

        self.assertAlmostEqual(result.diff, 0.3)
        self.assertTrue(result.is_significant)
        self.assertGreater(result.lo, 0.0)

    def test_is_deterministic_for_a_fixed_seed(self):
        a = [_rec(i, f1=(i % 7) / 10.0) for i in range(64)]
        b = [_rec(i, f1=(i % 5) / 10.0) for i in range(64)]

        first = sig.paired_bootstrap_ci(
            a, b, sig.mean_of("f1"), n_resamples=200, seed=7)
        second = sig.paired_bootstrap_ci(
            a, b, sig.mean_of("f1"), n_resamples=200, seed=7)

        self.assertEqual(first.lo, second.lo)
        self.assertEqual(first.hi, second.hi)

    def test_raises_on_length_mismatch(self):
        with self.assertRaises(ValueError):
            sig.paired_bootstrap_ci(
                [_rec(1)], [_rec(1), _rec(2)], sig.mean_of("f1"))

    def test_raises_on_empty_input(self):
        with self.assertRaises(ValueError):
            sig.paired_bootstrap_ci([], [], sig.mean_of("f1"))


class McNemarTest(unittest.TestCase):
    def test_balanced_disagreements_are_not_significant(self):
        a = [_rec(1, em=1.0), _rec(2, em=0.0)]
        b = [_rec(1, em=0.0), _rec(2, em=1.0)]

        result = sig.mcnemar(a, b)

        self.assertEqual((result.b, result.c), (1, 1))
        self.assertAlmostEqual(result.p_value, 1.0)

    def test_one_sided_disagreements_give_small_p(self):
        # Arrange: 10 wins, 0 losses -> 2 * 0.5**10
        a = [_rec(i, em=1.0) for i in range(10)]
        b = [_rec(i, em=0.0) for i in range(10)]

        result = sig.mcnemar(a, b)

        self.assertEqual((result.b, result.c), (10, 0))
        self.assertAlmostEqual(result.p_value, 2 * 0.5 ** 10)

    def test_agreements_are_discarded(self):
        a = [_rec(1, em=1.0), _rec(2, em=0.0), _rec(3, em=1.0)]
        b = [_rec(1, em=1.0), _rec(2, em=0.0), _rec(3, em=0.0)]

        result = sig.mcnemar(a, b)

        self.assertEqual((result.b, result.c), (1, 0))

    def test_no_disagreement_gives_p_of_one(self):
        a = [_rec(1, em=1.0)]
        b = [_rec(1, em=1.0)]

        result = sig.mcnemar(a, b)

        self.assertEqual((result.b, result.c), (0, 0))
        self.assertAlmostEqual(result.p_value, 1.0)

    def test_p_value_never_exceeds_one(self):
        a = [_rec(i, em=float(i % 2)) for i in range(20)]
        b = [_rec(i, em=float((i + 1) % 2)) for i in range(20)]

        result = sig.mcnemar(a, b)

        self.assertLessEqual(result.p_value, 1.0)


if __name__ == "__main__":
    unittest.main()
