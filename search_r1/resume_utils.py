"""Locate the latest training checkpoint for warm-restart resume."""

import os
import re

_STEP_DIR = re.compile(r"^global_step_(\d+)$")


def find_latest_checkpoint(actor_dir: str):
    """Return (path, step) of the highest global_step_N, else None.

    None when actor_dir is absent or holds no step checkpoints.
    """
    if not os.path.isdir(actor_dir):
        return None
    best_step = -1
    best_path = None
    for name in os.listdir(actor_dir):
        match = _STEP_DIR.match(name)
        if match is None:
            continue
        path = os.path.join(actor_dir, name)
        step = int(match.group(1))
        if step > best_step and os.path.isdir(path):
            best_step = step
            best_path = path
    if best_path is None:
        return None
    return best_path, best_step
