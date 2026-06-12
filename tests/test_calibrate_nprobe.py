"""Unit tests for the nprobe-calibration recall math."""

import importlib.util
import os
import unittest

_SPEC = importlib.util.spec_from_file_location(
    "calibrate_nprobe",
    os.path.join(
        os.path.dirname(__file__), "..", "scripts",
        "calibrate_nprobe.py"),
)
calibrate_nprobe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(calibrate_nprobe)

recall_at_k = calibrate_nprobe.recall_at_k
knee_nprobe = calibrate_nprobe.knee_nprobe
extract_question = calibrate_nprobe.extract_question


class ExtractQuestionTest(unittest.TestCase):
    def test_recovers_question_from_template(self):
        content = (
            "Answer the given question. ... For example, "
            "<answer> Beijing </answer>. Question: Who wrote Hamlet?\n"
        )
        self.assertEqual(
            extract_question(content), "Who wrote Hamlet?")

    def test_plain_question_passes_through(self):
        self.assertEqual(
            extract_question("What is Python?"), "What is Python?")


class RecallAtKTest(unittest.TestCase):
    def test_perfect_recall(self):
        pred = [[1, 2, 3], [4, 5, 6]]
        gold = [[1, 2, 3], [4, 5, 6]]
        self.assertEqual(recall_at_k(pred, gold), 1.0)

    def test_partial_recall_is_fraction_of_gold(self):
        pred = [[1, 9, 8]]
        gold = [[1, 2, 3]]
        self.assertAlmostEqual(recall_at_k(pred, gold), 1 / 3)

    def test_ignores_order_and_extra_preds(self):
        pred = [[3, 2, 1, 7]]
        gold = [[1, 2]]
        self.assertEqual(recall_at_k(pred, gold), 1.0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            recall_at_k([[1]], [[1], [2]])

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            recall_at_k([], [])


class KneeNprobeTest(unittest.TestCase):
    def test_picks_smallest_above_tolerance(self):
        recalls = {32: 0.90, 128: 0.985, 512: 0.995, 4096: 1.0}
        self.assertEqual(knee_nprobe(recalls, tol=0.99), 512)

    def test_falls_back_to_ceiling(self):
        recalls = {32: 0.50, 4096: 1.0}
        self.assertEqual(knee_nprobe(recalls, tol=0.99), 4096)


if __name__ == "__main__":
    unittest.main()
