"""Equivalence tests for the chunked log-prob/entropy path.

Asserts logprobs_and_entropy_from_logits_chunked matches the stock
log_softmax/entropy_from_logits path in both forward values and
gradients. Requires torch; skipped automatically where unavailable
(e.g. the CPU-only dev box), and exercised on the HPC env.
"""

import unittest

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
class ChunkedLogProbTest(unittest.TestCase):
    def setUp(self):
        from verl.utils.torch_functional import (
            entropy_from_logits,
            logprobs_from_logits_naive,
            logprobs_and_entropy_from_logits_chunked,
        )

        self.entropy_from_logits = entropy_from_logits
        self.logprobs_from_logits_naive = logprobs_from_logits_naive
        self.chunked = logprobs_and_entropy_from_logits_chunked
        torch.manual_seed(0)

    def _reference(self, logits, labels):
        log_prob = self.logprobs_from_logits_naive(logits, labels)
        entropy = self.entropy_from_logits(logits)
        return log_prob, entropy

    def test_forward_matches_stock_path(self):
        logits = torch.randn(37, 128, dtype=torch.float64)
        labels = torch.randint(0, 128, (37,))

        ref_lp, ref_ent = self._reference(logits, labels)
        lp, ent = self.chunked(logits, labels, chunk_size=8)

        torch.testing.assert_close(lp, ref_lp)
        torch.testing.assert_close(ent, ref_ent)

    def test_backward_matches_stock_path(self):
        labels = torch.randint(0, 128, (37,))
        base = torch.randn(37, 128, dtype=torch.float64)

        ref_logits = base.clone().requires_grad_(True)
        ref_lp, ref_ent = self._reference(ref_logits, labels)
        (ref_lp.sum() + ref_ent.sum()).backward()

        chunk_logits = base.clone().requires_grad_(True)
        lp, ent = self.chunked(chunk_logits, labels, chunk_size=8)
        (lp.sum() + ent.sum()).backward()

        torch.testing.assert_close(chunk_logits.grad, ref_logits.grad)

    def test_preserves_label_shape(self):
        logits = torch.randn(4, 9, 128, dtype=torch.float64)
        labels = torch.randint(0, 128, (4, 9))
        lp, ent = self.chunked(logits, labels, chunk_size=5)
        self.assertEqual(tuple(lp.shape), (4, 9))
        self.assertEqual(tuple(ent.shape), (4, 9))

    def test_rejects_nonpositive_chunk_size(self):
        logits = torch.randn(3, 16, dtype=torch.float64)
        labels = torch.randint(0, 16, (3,))
        with self.assertRaises(ValueError):
            self.chunked(logits, labels, chunk_size=0)


if __name__ == "__main__":
    unittest.main()
