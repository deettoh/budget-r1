"""Unit tests for prompt-prefix shape and ``derive_gold_budget``."""

import unittest

from scripts.data_process import thesis_qa
from scripts.data_process.thesis_qa import derive_gold_budget, make_search_prefix


class ThesisDataTest(unittest.TestCase):
    def test_data_generator_only_supports_rl_mode(self):
        self.assertEqual(thesis_qa.SUPPORTED_MODES, ("rl",))

    def test_make_search_prefix_can_require_budget_first(self):
        prefix = make_search_prefix(
            "Who wrote the book?", require_budget=True, max_budget=5
        )

        self.assertIn("Question: Who wrote the book?", prefix)
        self.assertIn("first output exactly one retrieval budget", prefix)
        self.assertIn("<budget>k</budget>", prefix)
        self.assertIn("[0, 5]", prefix)

    def test_make_search_prefix_keeps_search_r1_mode_without_budget(self):
        prefix = make_search_prefix(
            "Who wrote the book?", require_budget=False, max_budget=5
        )

        self.assertIn("<search> query </search>", prefix)
        self.assertNotIn("<budget>", prefix)

    def test_derive_gold_budget_from_supporting_fact_titles(self):
        example = {
            "supporting_facts": [["A", 0], ["A", 1], ["B", 0]],
        }

        self.assertEqual(derive_gold_budget(example, "hotpotqa"), 2)

    def test_derive_gold_budget_from_musique_paragraph_ids(self):
        example = {
            "paragraphs": [
                {"idx": 10, "is_supporting": True},
                {"idx": 11, "is_supporting": False},
                {"idx": 12, "is_supporting": True},
            ],
        }

        self.assertEqual(derive_gold_budget(example, "musique"), 2)

    def test_derive_gold_budget_from_flashrag_metadata_columnar_dict(self):
        # FlashRAG nests supporting_facts as parallel arrays in metadata
        example = {
            "question": "Which film came out first?",
            "metadata": {
                "supporting_facts": {
                    "title": [
                        "Move (1970 film)",
                        "Mediterranee (1963 film)",
                        "Move (1970 film)",
                    ],
                    "sent_id": [0, 0, 1],
                },
            },
        }

        self.assertEqual(derive_gold_budget(example, "2wikimultihopqa"), 2)

    def test_derive_gold_budget_raises_when_no_signal(self):
        example = {"question": "Who?", "metadata": {"type": "comparison"}}

        with self.assertRaises(ValueError):
            derive_gold_budget(example, "hotpotqa")


if __name__ == "__main__":
    unittest.main()
