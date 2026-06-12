"""HPC compatibility patches auto-loaded by every Python process.

Deploy as sitecustomize.py in the conda env site-packages so site.py
loads it at startup for the driver and every Ray worker. Patches Ray
probes that os.listdir restricted /proc and /dev paths, and the
bash-exec that the Slurm sandbox blocks.
"""

from __future__ import annotations

import os


# move ~/.local to sys.path tail so its torch can't shadow conda's
import sys as _sys
import site as _site
_user_base = _site.getuserbase()
_env_paths = [p for p in _sys.path if not p.startswith(_user_base)]
_user_paths = [p for p in _sys.path if p.startswith(_user_base)]
_sys.path[:] = _env_paths + _user_paths


# an instance not a function or pathlib binds it as a method
_orig_listdir = os.listdir


class _HpcSafeListdir:
    """Drop-in os.listdir that tolerates locked-down /dev and /proc."""

    def __call__(self, path="."):
        try:
            return _orig_listdir(path)
        except PermissionError:
            if isinstance(path, bytes):
                p = path.decode(errors="replace")
            else:
                p = str(path)
            if p.startswith("/dev") or p.startswith("/proc"):
                return []
            raise


os.listdir = _HpcSafeListdir()


# stub psutil it bypasses the os.listdir patch above
try:
    import psutil

    psutil.pids = lambda: []
    psutil.Process.parents = lambda self: []
except ImportError:
    # psutil not installed here nothing to patch
    pass


# skip Ray's bash worker-exec wrapper HPC blocks bash in Slurm
try:
    import sys

    from ray._private.runtime_env import context as _ray_rt_context
    from ray._private.utils import update_envs as _ray_update_envs
    from ray.core.generated.common_pb2 import Language as _RayLanguage

    _orig_exec_worker = _ray_rt_context.RuntimeEnvContext.exec_worker

    def _hpc_exec_worker(self, passthrough_args, language):
        """Exec the worker without bash when no shell prefix is set."""
        if self.command_prefix or language != _RayLanguage.PYTHON:
            return _orig_exec_worker(self, passthrough_args, language)
        _ray_update_envs(self.env_vars)
        py = self.py_executable or sys.executable
        os.execvp(py, [py, *passthrough_args])

    _ray_rt_context.RuntimeEnvContext.exec_worker = _hpc_exec_worker
except ImportError:
    # Ray not importable here nothing to patch
    pass
