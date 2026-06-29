"""Unit tests for gold-title extraction and recall-probe helpers."""

import unittest

from scripts.data_process.thesis_qa import extract_gold_titles
from scripts.gold_recall_probe import (
    normalize_title,
    recall_at_k,
    title_from_contents,
)


class ExtractGoldTitlesTest(unittest.TestCase):
    def test_hotpot_supporting_fact_pairs(self):
        example = {
            "supporting_facts": [["Title A", 0], ["Title B", 2], ["Title A", 1]]
        }
        self.assertEqual(
            extract_gold_titles(example, "hotpotqa"), ["Title A", "Title B"]
        )

    def test_columnar_dict_shape(self):
        example = {"supporting_facts": {"title": ["X", "Y"], "sent_id": [0, 1]}}
        self.assertEqual(
            extract_gold_titles(example, "2wikimultihopqa"), ["X", "Y"]
        )

    def test_musique_supporting_paragraph_titles(self):
        example = {
            "paragraphs": [
                {"title": "P1", "is_supporting": True},
                {"title": "P2", "is_supporting": False},
                {"title": "P3", "is_supporting": True},
            ]
        }
        self.assertEqual(extract_gold_titles(example, "musique"), ["P1", "P3"])

    def test_empty_when_no_signal(self):
        self.assertEqual(extract_gold_titles({}, "hotpotqa"), [])


class RecallHelpersTest(unittest.TestCase):
    def test_normalize_title_lowers_and_collapses_space(self):
        self.assertEqual(normalize_title("  The   Movie "), "the movie")

    def test_title_from_contents_takes_first_line(self):
        contents = "Barack Obama\nHe was the 44th president."
        self.assertEqual(title_from_contents(contents), "Barack Obama")

    def test_recall_at_k_counts_matched_gold(self):
        self.assertEqual(recall_at_k(["a", "C", "D"], ["A", "B"]), 0.5)

    def test_recall_at_k_zero_when_no_gold(self):
        self.assertEqual(recall_at_k(["x"], []), 0.0)


if __name__ == "__main__":
    unittest.main()
