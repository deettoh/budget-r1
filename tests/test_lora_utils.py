"""Unit tests for the torch-free LoRA resume helpers."""

import os
import tempfile
import unittest

from search_r1.lora_utils import (
    ADAPTER_CONFIG_FILE,
    ADAPTER_WEIGHTS_FILE,
    resolve_adapter_path,
    strip_fsdp_prefix,
)


def _write_adapter_dir(path, *, weights=True, config=True):
    """Create an adapter dir with the requested marker files."""
    os.makedirs(path, exist_ok=True)
    if weights:
        open(os.path.join(path, ADAPTER_WEIGHTS_FILE), "wb").close()
    if config:
        open(os.path.join(path, ADAPTER_CONFIG_FILE), "w").close()


class StripFsdpPrefixTest(unittest.TestCase):
    def test_strips_wrapper_prefix(self):
        name = "_fsdp_wrapped_module.base_model.model.layers.0"
        self.assertEqual(
            strip_fsdp_prefix(name), "base_model.model.layers.0")

    def test_passes_clean_name_through(self):
        self.assertEqual(
            strip_fsdp_prefix("base_model.weight"),
            "base_model.weight")


class ResolveAdapterPathTest(unittest.TestCase):
    def test_returns_none_when_config_absent(self):
        self.assertIsNone(resolve_adapter_path(None))
        self.assertIsNone(resolve_adapter_path({}))

    def test_returns_none_when_path_unset(self):
        self.assertIsNone(resolve_adapter_path({"enabled": True}))
        self.assertIsNone(
            resolve_adapter_path({"adapter_path": None}))

    def test_roundtrip_returns_valid_dir(self):
        with tempfile.TemporaryDirectory() as root:
            adapter = os.path.join(root, "global_step_2")
            _write_adapter_dir(adapter)
            resolved = resolve_adapter_path(
                {"adapter_path": adapter})
            self.assertEqual(resolved, adapter)

    def test_returns_none_when_path_empty(self):
        self.assertIsNone(
            resolve_adapter_path({"adapter_path": ""}))

    def test_raises_when_weights_missing(self):
        with tempfile.TemporaryDirectory() as root:
            adapter = os.path.join(root, "broken")
            _write_adapter_dir(adapter, weights=False)
            with self.assertRaises(FileNotFoundError):
                resolve_adapter_path({"adapter_path": adapter})

    def test_raises_when_config_missing(self):
        with tempfile.TemporaryDirectory() as root:
            adapter = os.path.join(root, "broken")
            _write_adapter_dir(adapter, config=False)
            with self.assertRaises(FileNotFoundError):
                resolve_adapter_path({"adapter_path": adapter})


if __name__ == "__main__":
    unittest.main()
