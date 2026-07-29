"""Unit tests for the resume checkpoint finder.

Covers step-number parsing, the absent-directory and no-checkpoint
cases, and the optimizer state round-trip including a checkpoint
written without an optimizer file.

Typical usage example:

  python3 -m unittest tests.test_resume_utils
"""

import os
import tempfile
import unittest

from search_r1.resume_utils import (
    find_latest_checkpoint,
    load_optimizer_state,
    save_optimizer_state,
)

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class FindLatestCheckpointTest(unittest.TestCase):
    def test_returns_none_for_missing_dir(self):
        self.assertIsNone(
            find_latest_checkpoint("/no/such/actor/dir")
        )

    def test_returns_none_when_no_step_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "best"))
            self.assertIsNone(find_latest_checkpoint(tmp))

    def test_picks_highest_step_numerically(self):
        with tempfile.TemporaryDirectory() as tmp:
            for step in (5, 10, 2):
                os.makedirs(
                    os.path.join(tmp, f"global_step_{step}")
                )
            path, step = find_latest_checkpoint(tmp)
            self.assertEqual(step, 10)
            self.assertTrue(path.endswith("global_step_10"))

    def test_ignores_non_step_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "global_step_3"))
            os.makedirs(os.path.join(tmp, "best"))
            open(os.path.join(tmp, "global_step_9"), "w").close()
            path, step = find_latest_checkpoint(tmp)
            # the step_9 file is not a dir, so step_3 wins
            self.assertEqual(step, 3)


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class OptimizerStateRoundtripTest(unittest.TestCase):
    def _stepped_optimizer(self):
        param = torch.nn.Parameter(torch.ones(4))
        opt = torch.optim.AdamW([param], lr=1e-3)
        (param.pow(2).sum()).backward()
        opt.step()
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lr_lambda=lambda s: 1.0
        )
        sched.step()
        return param, opt, sched

    def test_restores_moments_and_scheduler_step(self):
        param, opt, sched = self._stepped_optimizer()
        with tempfile.TemporaryDirectory() as tmp:
            save_optimizer_state(opt, sched, tmp)
            param2 = torch.nn.Parameter(torch.ones(4))
            opt2 = torch.optim.AdamW([param2], lr=1e-3)
            sched2 = torch.optim.lr_scheduler.LambdaLR(
                opt2, lr_lambda=lambda s: 1.0
            )
            loaded = load_optimizer_state(opt2, sched2, tmp)
            self.assertTrue(loaded)
            self.assertTrue(
                torch.allclose(
                    opt2.state[param2]["exp_avg"],
                    opt.state[param]["exp_avg"],
                )
            )
            self.assertEqual(sched2.last_epoch, sched.last_epoch)

    def test_returns_false_when_no_optim_file(self):
        _, opt, sched = self._stepped_optimizer()
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(load_optimizer_state(opt, sched, tmp))

    def test_scheduler_optional_on_save(self):
        _, opt, _ = self._stepped_optimizer()
        with tempfile.TemporaryDirectory() as tmp:
            save_optimizer_state(opt, None, tmp)
            _, opt2, _ = self._stepped_optimizer()
            self.assertTrue(load_optimizer_state(opt2, None, tmp))


if __name__ == "__main__":
    unittest.main()
