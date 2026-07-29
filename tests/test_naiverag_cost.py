"""Tests for naive-RAG injected-passage extraction.

Covers the last-span rule that skips the instruction line, which
itself mentions both <information> markers literally.

Typical usage example:

  python3 -m unittest tests.test_naiverag_cost
"""

import os
import sys
import unittest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.naiverag_cost import last_information_span


class LastInformationSpanTest(unittest.TestCase):
    def test_skips_instruction_mention_and_returns_docs(self):
        content = (
            "Answer based on the given information. The top results "
            "are provided between <information> and </information>.\n"
            "<information>Doc 1(Title: Foo) real passage text"
            "</information>\n<think>"
        )
        self.assertEqual(
            last_information_span(content),
            "Doc 1(Title: Foo) real passage text",
        )

    def test_returns_empty_when_no_block(self):
        self.assertEqual(last_information_span("no tags here"), "")

    def test_single_block_returned_verbatim(self):
        content = "<information>only block</information>"
        self.assertEqual(last_information_span(content), "only block")


if __name__ == "__main__":
    unittest.main()
