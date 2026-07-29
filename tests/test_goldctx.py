"""Unit tests for the gold-context (oracle-RAG) eval parquet builder.

Covers gold-passage formatting and record assembly, including the
None return when a row exposes no gold passages.

Typical usage example:

  python3 -m unittest tests.test_goldctx
"""

import unittest

from scripts.build_goldctx_data import (
    format_gold_passages,
    make_goldctx_record,
)


class FormatGoldPassagesTest(unittest.TestCase):
    def test_numbers_docs_with_title_and_body(self):
        block = format_gold_passages(
            [("T1", "body one"), ("T2", "body two")]
        )
        self.assertEqual(
            block,
            "Doc 1(Title: T1) body one\nDoc 2(Title: T2) body two\n",
        )

    def test_empty_pairs(self):
        self.assertEqual(format_gold_passages([]), "")


class MakeGoldctxRecordTest(unittest.TestCase):
    def test_builds_record_with_gold_context_and_answer(self):
        example = {
            "question": "Who won?",
            "golden_answers": ["Alice"],
            "supporting_facts": [["T1", 0]],
            "context": [["T1", ["Alice won the prize."]]],
        }
        rec = make_goldctx_record(example, 7, "hotpotqa", "dev")
        self.assertEqual(rec["data_source"], "hotpotqa")
        self.assertEqual(
            rec["reward_model"]["ground_truth"]["target"], ["Alice"]
        )
        self.assertEqual(rec["extra_info"], {"split": "dev", "index": 7})
        content = rec["prompt"][0]["content"]
        self.assertIn("Alice won the prize.", content)
        self.assertIn("Who won?", content)

    def test_none_when_no_gold_passages(self):
        example = {"question": "Q", "golden_answers": ["A"]}
        self.assertIsNone(
            make_goldctx_record(example, 0, "hotpotqa", "dev")
        )


if __name__ == "__main__":
    unittest.main()
