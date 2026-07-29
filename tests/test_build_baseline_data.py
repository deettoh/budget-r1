"""Unit tests for the baseline prompt builders.

Covers question extraction from an agentic prompt and the
closed-book and naive-RAG prompt shapes.

Typical usage example:

  python3 -m unittest tests.test_build_baseline_data
"""

import importlib.util
import os
import unittest

_SPEC = importlib.util.spec_from_file_location(
    "build_baseline_data",
    os.path.join(
        os.path.dirname(__file__), "..", "scripts",
        "build_baseline_data.py"),
)
build_baseline_data = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(build_baseline_data)


class PromptBuilderTest(unittest.TestCase):
    def test_closedbook_has_no_search_instruction(self):
        p = build_baseline_data.make_closedbook_prefix("Who?")
        self.assertNotIn("<search>", p)
        self.assertNotIn("<information>", p)
        self.assertIn("<answer>", p)
        self.assertTrue(p.endswith("Question: Who?\n"))

    def test_naiverag_injects_passages_no_search(self):
        p = build_baseline_data.make_naiverag_prefix(
            "Who?", "Doc 1(Title: X) body\n")
        self.assertIn("<information>Doc 1(Title: X) body", p)
        self.assertNotIn("<search>", p)
        self.assertTrue(p.endswith("Question: Who?\n"))

    def test_format_passages_matches_harness(self):
        docs = [{"document": {"contents": "Title A\nline one"}}]
        self.assertEqual(
            build_baseline_data.format_passages(docs),
            "Doc 1(Title: Title A) line one\n")

    def test_extract_question(self):
        self.assertEqual(
            build_baseline_data.extract_question(
                "blah Question: What is X?\n"),
            "What is X?")


if __name__ == "__main__":
    unittest.main()
