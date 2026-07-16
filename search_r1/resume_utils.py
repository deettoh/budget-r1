"""Locate and restore training state for warm-restart resume."""

import os
import re

_STEP_DIR = re.compile(r"^global_step_(\d+)$")
_OPTIM_FILE = "optim_state.pt"


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


def save_optimizer_state(optimizer, lr_scheduler, directory: str) -> None:
    """Save optimizer and scheduler state into directory."""
    import torch

    state = {
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": (
            lr_scheduler.state_dict() if lr_scheduler is not None else None
        ),
    }
    torch.save(state, os.path.join(directory, _OPTIM_FILE))


def load_optimizer_state(optimizer, lr_scheduler, directory: str) -> bool:
    """Restore optimizer and scheduler state, return True if loaded.

    Returns False when the directory has no optim file so a
    pre-feature checkpoint resumes with a fresh optimizer instead of
    failing.
    """
    import torch

    path = os.path.join(directory, _OPTIM_FILE)
    if not os.path.exists(path):
        return False
    # weights_only rejects the non-tensor param_groups, ckpt is ours
    state = torch.load(path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    if lr_scheduler is not None and state.get("lr_scheduler") is not None:
        lr_scheduler.load_state_dict(state["lr_scheduler"])
    return True
