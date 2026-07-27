"""Unit tests for the SFT retrieved-span loss mask."""

import unittest

from search_r1.sft_masking import (
    build_loss_mask,
    find_subsequence,
    information_span_flags,
)


# stand-in marker encodings; real runs pass tokenizer ids
OPEN = [90, 91]
CLOSE = [92, 93]


class FindSubsequenceTest(unittest.TestCase):
    def test_returns_first_match_index(self):
        self.assertEqual(find_subsequence([1, 90, 91, 2], OPEN), 1)

    def test_respects_start(self):
        seq = [90, 91, 0, 90, 91]
        self.assertEqual(find_subsequence(seq, OPEN, start=2), 3)

    def test_missing_returns_minus_one(self):
        self.assertEqual(find_subsequence([1, 2, 3], OPEN), -1)

    def test_empty_needle_returns_minus_one(self):
        self.assertEqual(find_subsequence([1, 2], []), -1)


class InformationSpanFlagsTest(unittest.TestCase):
    def test_no_markers_keeps_all(self):
        ids = [1, 2, 3, 4]
        flags = information_span_flags(ids, OPEN, CLOSE)
        self.assertEqual(flags, [1, 1, 1, 1])

    def test_single_span_masked_inclusive(self):
        ids = [1, 90, 91, 5, 92, 93, 8]
        # mask indices 1..5 (open..close inclusive), keep 0, 6
        self.assertEqual(
            information_span_flags(ids, OPEN, CLOSE),
            [1, 0, 0, 0, 0, 0, 1],
        )

    def test_two_spans_masked(self):
        ids = [90, 91, 92, 93, 7, 90, 91, 92, 93]
        self.assertEqual(
            information_span_flags(ids, OPEN, CLOSE),
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
        )

    def test_unclosed_open_masks_to_end(self):
        ids = [1, 90, 91, 5, 6]
        self.assertEqual(
            information_span_flags(ids, OPEN, CLOSE),
            [1, 0, 0, 0, 0],
        )

    def test_empty_markers_keep_all(self):
        ids = [1, 2, 3]
        self.assertEqual(information_span_flags(ids, [], CLOSE), [1, 1, 1])


class BuildLossMaskTest(unittest.TestCase):
    def test_response_region_shifted_back_one(self):
        # prompt 3, response 2, no padding: grade positions 2,3
        self.assertEqual(build_loss_mask(3, 2, 5), [0, 0, 1, 1, 0])

    def test_padding_excluded(self):
        # prompt 2, response 2, size 6 -> ones at 1,2; tail zero
        self.assertEqual(build_loss_mask(2, 2, 6), [0, 1, 1, 0, 0, 0])

    def test_info_keep_zeroes_retrieved_targets(self):
        # response tokens r=0,1,2,3; r2,r3 are <information> (keep 0)
        # base ones at prompt-1..prompt+resp-2 = idx 1,2,3,4
        keep = [1, 1, 0, 0]
        # info tokens r2,r3 -> zero idx prompt+2-1=3, prompt+3-1=4
        self.assertEqual(
            build_loss_mask(2, 4, 7, keep), [0, 1, 1, 0, 0, 0, 0]
        )


if __name__ == "__main__":
    unittest.main()
