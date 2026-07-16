"""Unit tests for the resume checkpoint finder."""

import os
import tempfile
import unittest

from search_r1.resume_utils import find_latest_checkpoint


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


if __name__ == "__main__":
    unittest.main()
