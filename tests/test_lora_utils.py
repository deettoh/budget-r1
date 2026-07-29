"""Unit tests for the torch-free LoRA resume helpers.

Covers FSDP prefix stripping, adapter-path resolution including the
fail-fast on a configured but incomplete directory, and adapter-name
reinsertion into lora keys.

Typical usage example:

  python3 -m unittest tests.test_lora_utils
"""

import os
import tempfile
import unittest

from search_r1.lora_utils import (
    ADAPTER_CONFIG_FILE,
    ADAPTER_WEIGHTS_FILE,
    insert_adapter_name,
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


class InsertAdapterNameTest(unittest.TestCase):
    def test_inserts_default_name_into_lora_keys(self):
        state = {
            "base_model.model.layers.0.q_proj.lora_A.weight": "a",
            "base_model.model.layers.0.q_proj.lora_B.weight": "b",
        }
        renamed = insert_adapter_name(state)
        self.assertEqual(
            sorted(renamed),
            [
                "base_model.model.layers.0.q_proj"
                ".lora_A.default.weight",
                "base_model.model.layers.0.q_proj"
                ".lora_B.default.weight",
            ])
        self.assertEqual(
            renamed["base_model.model.layers.0.q_proj"
                    ".lora_A.default.weight"], "a")

    def test_custom_adapter_name(self):
        state = {"m.lora_A.weight": "a"}
        self.assertEqual(
            insert_adapter_name(state, adapter_name="other"),
            {"m.lora_A.other.weight": "a"})

    def test_non_lora_keys_pass_through(self):
        state = {"m.embed_tokens.weight": "w"}
        self.assertEqual(insert_adapter_name(state), state)

    def test_returns_new_dict_input_unchanged(self):
        state = {"m.lora_B.weight": "b"}
        renamed = insert_adapter_name(state)
        self.assertIsNot(renamed, state)
        self.assertEqual(state, {"m.lora_B.weight": "b"})


if __name__ == "__main__":
    unittest.main()
