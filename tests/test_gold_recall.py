"""Unit tests for gold-title extraction and recall-probe helpers."""

import unittest

from scripts.data_process.thesis_qa import (
    extract_gold_passages,
    extract_gold_titles,
)
from scripts.gold_recall_probe import (
    _oracle_recall_at_k,
    mean_recall_by_k,
    normalize_title,
    recall_at_k,
    title_from_contents,
)


class ExtractGoldTitlesTest(unittest.TestCase):
    def test_hotpot_supporting_fact_pairs(self):
        facts = [["Title A", 0], ["Title B", 2], ["Title A", 1]]
        example = {"supporting_facts": facts}
        self.assertEqual(
            extract_gold_titles(example, "hotpotqa"), ["Title A", "Title B"]
        )

    def test_columnar_dict_shape(self):
        facts = {"title": ["X", "Y"], "sent_id": [0, 1]}
        example = {"supporting_facts": facts}
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


class ExtractGoldPassagesTest(unittest.TestCase):
    def test_hotpot_context_list_form(self):
        example = {
            "supporting_facts": [["Title A", 0], ["Title B", 1]],
            "context": [
                ["Title A", ["A first sentence.", "A second."]],
                ["Title B", ["B sentence."]],
                ["Distractor", ["noise"]],
            ],
        }
        self.assertEqual(
            extract_gold_passages(example, "hotpotqa"),
            [
                ("Title A", "A first sentence. A second."),
                ("Title B", "B sentence."),
            ],
        )

    def test_2wiki_columnar_context_in_metadata(self):
        example = {
            "metadata": {
                "supporting_facts": {"title": ["X"], "sent_id": [0]},
                "context": {
                    "title": ["X", "Y"],
                    "content": [["X body."], ["Y body."]],
                },
            }
        }
        self.assertEqual(
            extract_gold_passages(example, "2wikimultihopqa"),
            [("X", "X body.")],
        )

    def test_musique_paragraph_text(self):
        example = {
            "paragraphs": [
                {
                    "title": "P1",
                    "is_supporting": True,
                    "paragraph_text": "P1 text.",
                },
                {
                    "title": "P2",
                    "is_supporting": False,
                    "paragraph_text": "P2 text.",
                },
            ]
        }
        self.assertEqual(
            extract_gold_passages(example, "musique"),
            [("P1", "P1 text.")],
        )

    def test_empty_when_no_gold_titles(self):
        self.assertEqual(extract_gold_passages({}, "hotpotqa"), [])

    def test_skips_gold_title_without_body(self):
        example = {
            "supporting_facts": [["Title A", 0], ["NoBody", 0]],
            "context": [["Title A", ["A body."]]],
        }
        self.assertEqual(
            extract_gold_passages(example, "hotpotqa"),
            [("Title A", "A body.")],
        )


class RecallSweepTest(unittest.TestCase):
    def test_mean_recall_by_k_rises_with_k(self):
        retrieved = [["a", "b", "c", "d"]]
        gold = [["c", "d"]]
        out = mean_recall_by_k(retrieved, gold, [2, 4])
        self.assertEqual(out[2], 0.0)
        self.assertEqual(out[4], 1.0)

    def test_mean_recall_by_k_empty_gold(self):
        self.assertEqual(mean_recall_by_k([], [], [3]), {3: 0.0})

    def test_oracle_recall_at_k_per_example(self):
        owner = [(0, "A"), (0, "B")]
        oracle_titles = [["A", "z"], ["x", "y"]]
        self.assertEqual(
            _oracle_recall_at_k(owner, oracle_titles, 1, 2), 0.5
        )


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
