"""Unit tests for prompt-prefix shape and ``derive_gold_budget``."""

import unittest

from scripts.data_process import thesis_qa
from scripts.data_process.thesis_qa import (
    cap_records_per_budget,
    derive_gold_budget,
    extract_gold_titles,
    make_rl_record,
    make_search_prefix,
)


class ThesisDataTest(unittest.TestCase):
    def test_data_generator_only_supports_rl_mode(self):
        self.assertEqual(thesis_qa.SUPPORTED_MODES, ("rl",))

    def test_make_search_prefix_requires_budget_before_search(self):
        # think-first, reasoning precedes the declaration
        # budget still binds before any search
        prefix = make_search_prefix(
            "Who wrote the book?", require_budget=True, max_budget=5
        )

        self.assertIn("Question: Who wrote the book?", prefix)
        self.assertIn("First think inside <think>", prefix)
        self.assertIn(
            "before any search, output exactly one retrieval budget",
            prefix,
        )
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


class ExtractGoldTitlesMusiqueTest(unittest.TestCase):
    """FlashRAG musique carries titles only inside the decomposition.

    Every musique row lacks supporting_facts / paragraphs, so the
    extractor must read metadata.question_decomposition[*]
    .support_paragraph.title (the hole that zeroed grounding on
    musique in jobs 6695-6722).
    """

    def _flashrag_musique_example(self):
        return {
            "question": "Who is the spouse of the Green performer?",
            "golden_answers": ["Miquette Giraudy"],
            "metadata": {
                "answerable": True,
                "question_decomposition": [
                    {
                        "id": 460946,
                        "question": "Green >> performer",
                        "answer": "Steve Hillage",
                        "paragraph_support_idx": 10,
                        "support_paragraph": {
                            "idx": 10,
                            "title": "Green (Steve Hillage album)",
                            "paragraph_text": "Green is the fourth...",
                            "is_supporting": True,
                        },
                    },
                    {
                        "id": 294723,
                        "question": "#1 >> spouse",
                        "answer": "Miquette Giraudy",
                        "paragraph_support_idx": 5,
                        "support_paragraph": {
                            "idx": 5,
                            "title": "Miquette Giraudy",
                            "paragraph_text": "Miquette Giraudy...",
                            "is_supporting": True,
                        },
                    },
                ],
            },
        }

    def test_titles_extracted_from_decomposition_support(self):
        titles = extract_gold_titles(
            self._flashrag_musique_example(), "musique"
        )
        self.assertEqual(
            titles,
            ["Green (Steve Hillage album)", "Miquette Giraudy"],
        )

    def test_decomposition_without_support_paragraph_is_safe(self):
        example = {
            "metadata": {
                "question_decomposition": [
                    {"id": 1, "question": "q", "answer": "a"},
                ],
            },
        }
        self.assertEqual(extract_gold_titles(example, "musique"), [])


class BudgetTemplateTest(unittest.TestCase):
    def test_minimal_keeps_native_wording_plus_budget(self):
        # probe template: native prompt + one budget sentence, so
        # the frozen-floor delta isolates the declaration ask
        prefix = make_search_prefix(
            "Who wrote the book?", require_budget=True,
            max_budget=5, budget_template="minimal",
        )
        self.assertIn("<budget>k</budget>", prefix)
        self.assertIn("integer in [0, 5]", prefix)
        self.assertIn(
            "as many times as your want, up to your budget k",
            prefix,
        )
        self.assertNotIn("First think inside", prefix)
        self.assertIn("Question: Who wrote the book?", prefix)

    def test_default_template_is_think_first(self):
        prefix = make_search_prefix(
            "Who wrote the book?", require_budget=True, max_budget=5,
        )
        self.assertIn("First think inside", prefix)

    def test_unknown_template_raises(self):
        with self.assertRaises(ValueError):
            make_search_prefix(
                "Q?", require_budget=True, max_budget=5,
                budget_template="fancy",
            )


class CapRecordsPerBudgetTest(unittest.TestCase):
    def _records(self, budgets):
        return [
            {"extra_info": {"gold_budget": g, "index": i}}
            for i, g in enumerate(budgets)
        ]

    def test_caps_each_budget_value_independently(self):
        records = self._records([2, 2, 2, 2, 3, 3, 4])
        capped = cap_records_per_budget(records, cap=2, seed=42)
        from collections import Counter
        dist = Counter(
            r["extra_info"]["gold_budget"] for r in capped
        )
        self.assertEqual(dist[2], 2)
        self.assertEqual(dist[3], 2)
        self.assertEqual(dist[4], 1)

    def test_deterministic_for_same_seed(self):
        records = self._records([2] * 50 + [3] * 5)
        a = cap_records_per_budget(records, cap=10, seed=7)
        b = cap_records_per_budget(records, cap=10, seed=7)
        self.assertEqual(
            [r["extra_info"]["index"] for r in a],
            [r["extra_info"]["index"] for r in b],
        )

    def test_no_op_when_cap_not_positive(self):
        records = self._records([2, 2, 3])
        self.assertEqual(
            cap_records_per_budget(records, cap=0, seed=42), records
        )

    def test_does_not_mutate_input(self):
        records = self._records([2, 2, 3])
        before = [r["extra_info"]["index"] for r in records]
        cap_records_per_budget(records, cap=1, seed=42)
        self.assertEqual(
            [r["extra_info"]["index"] for r in records], before
        )

    def test_raises_when_gold_budget_missing(self):
        records = [{"extra_info": {"index": 0}}]
        with self.assertRaises(ValueError):
            cap_records_per_budget(records, cap=1, seed=42)


class MakeRlRecordTest(unittest.TestCase):
    def _example(self):
        return {
            "question": "Who wrote the book?",
            "golden_answers": ["Ada"],
            "supporting_facts": [["Title A", 0], ["Title B", 1]],
        }

    def test_gold_titles_added_when_require_budget(self):
        rec = make_rl_record(
            self._example(), 0, "hotpotqa", "dev",
            require_budget=True, max_budget=5,
        )
        self.assertEqual(
            rec["extra_info"]["gold_titles"], ["Title A", "Title B"]
        )
        self.assertEqual(rec["extra_info"]["gold_budget"], 2)

    def test_gold_metadata_present_without_budget_prompt(self):
        # changed 2026-07-06 for the §16 control parquet: gold
        # metadata is always derived so the per-budget cap (and
        # cost-only grounding) work on control-prompt data too
        # require_budget now controls ONLY the prompt wording
        rec = make_rl_record(
            self._example(), 0, "hotpotqa", "dev",
            require_budget=False, max_budget=5,
        )
        self.assertNotIn("<budget>", rec["prompt"][0]["content"])
        self.assertEqual(
            rec["extra_info"]["gold_titles"], ["Title A", "Title B"]
        )
        self.assertEqual(rec["extra_info"]["gold_budget"], 2)


if __name__ == "__main__":
    unittest.main()
