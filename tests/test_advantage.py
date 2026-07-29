"""Tests for group-normalized cost in the GRPO advantage (option a).

Covers the group z-score and the subtraction of normalized cost from
the advantage. Requires torch, so it runs on the HPC base env rather
than the CPU-only dev box.

Typical usage example:

  srun python -m unittest tests.test_advantage
"""

import unittest

import numpy as np
import torch

from verl.trainer.ppo.core_algos import (
    _group_zscore,
    compute_grpo_outcome_advantage,
)


def _rewards_from_scores(scores):
    """Pack per-sample scalars into a last-token reward tensor."""
    bsz = len(scores)
    tensor = torch.zeros(bsz, 3)
    for i, s in enumerate(scores):
        tensor[i, -1] = s
    return tensor


class GroupZscoreTest(unittest.TestCase):
    def test_within_group_zscore_has_zero_mean(self):
        values = torch.tensor([1.0, 3.0, 10.0, 20.0])
        index = np.array(["a", "a", "b", "b"])
        z = _group_zscore(values, index, epsilon=1e-6)
        self.assertAlmostEqual(float(z[:2].mean()), 0.0, places=4)
        self.assertAlmostEqual(float(z[2:].mean()), 0.0, places=4)
        # cheaper sample in each group is below its group mean
        self.assertLess(float(z[0]), float(z[1]))
        self.assertLess(float(z[2]), float(z[3]))

    def test_singleton_group_gets_zero(self):
        values = torch.tensor([5.0])
        index = np.array(["solo"])
        z = _group_zscore(values, index, epsilon=1e-6)
        self.assertAlmostEqual(float(z[0]), 0.0, places=4)


class AdvantageCostTest(unittest.TestCase):
    def test_cost_off_is_identical_to_baseline(self):
        rewards = _rewards_from_scores([1.0, 0.0, 1.0, 0.0])
        mask = torch.ones(4, 3)
        index = np.array(["a", "a", "b", "b"])
        base, _ = compute_grpo_outcome_advantage(rewards, mask, index)
        # passing cost with coeff 0 must not change anything
        cost = torch.tensor([1.0, 4.0, 2.0, 3.0])
        off, _ = compute_grpo_outcome_advantage(
            rewards, mask, index, cost=cost, cost_coeff=0.0
        )
        self.assertTrue(torch.allclose(base, off))

    def test_higher_cost_lowers_advantage_within_tied_group(self):
        # tied reward zeroes the score z, so only cost separates them
        rewards = _rewards_from_scores([1.0, 1.0])
        mask = torch.ones(2, 3)
        index = np.array(["q", "q"])
        cost = torch.tensor([1.0, 5.0])
        adv, _ = compute_grpo_outcome_advantage(
            rewards, mask, index, cost=cost, cost_coeff=0.5
        )
        cheap = float(adv[0, -1])
        pricey = float(adv[1, -1])
        self.assertGreater(cheap, pricey)

    def test_all_fail_group_gives_zero_cost_gradient(self):
        # the v4 death spiral, all-zero groups let cheap garbage win
        rewards = _rewards_from_scores([0.0, 0.0, 0.0])
        mask = torch.ones(3, 3)
        index = np.array(["q", "q", "q"])
        cost = torch.tensor([0.0, 2.0, 5.0])
        adv, _ = compute_grpo_outcome_advantage(
            rewards, mask, index, cost=cost, cost_coeff=0.5
        )
        self.assertTrue(torch.allclose(adv, torch.zeros_like(adv)))

    def test_gate_keeps_cost_tiebreak_for_scoring_rollouts(self):
        # cheaper correct rollout wins, the failed one pays no cost
        rewards = _rewards_from_scores([1.0, 1.0, 0.0])
        mask = torch.ones(3, 3)
        index = np.array(["q", "q", "q"])
        cost = torch.tensor([1.0, 5.0, 0.0])
        with_cost, _ = compute_grpo_outcome_advantage(
            rewards, mask, index, cost=cost, cost_coeff=0.5
        )
        no_cost, _ = compute_grpo_outcome_advantage(
            rewards, mask, index, cost=cost, cost_coeff=0.0
        )
        self.assertGreater(
            float(with_cost[0, -1]), float(with_cost[1, -1])
        )
        self.assertAlmostEqual(
            float(with_cost[2, -1]), float(no_cost[2, -1]), places=5
        )


if __name__ == "__main__":
    unittest.main()
