"""Torch-free helpers for LoRA adapter save and resume.

Path resolution and name-mapping only, unit-testable without a GPU.
The heavy PeftModel/FSDP calls stay in fsdp_workers.py.
"""

import os

_FSDP_PREFIX = "_fsdp_wrapped_module."
ADAPTER_WEIGHTS_FILE = "adapter_model.safetensors"
ADAPTER_CONFIG_FILE = "adapter_config.json"


def strip_fsdp_prefix(name: str) -> str:
    """Return the param name with the FSDP wrapper prefix removed."""
    return name.replace(_FSDP_PREFIX, "")


def insert_adapter_name(
    state_dict: dict, adapter_name: str = "default"
) -> dict:
    """Return a new dict with adapter names restored in lora keys.

    PEFT strips the name when saving, plain load_state_dict needs it
    back. Doing it here avoids peft loaders that need
    transformers>=4.50.
    """
    renamed = {}
    for key, value in state_dict.items():
        for marker in ("lora_A", "lora_B"):
            suffix = f".{marker}.weight"
            if key.endswith(suffix):
                key = (key[:-len(suffix)]
                       + f".{marker}.{adapter_name}.weight")
                break
        renamed[key] = value
    return renamed


def resolve_adapter_path(lora_config) -> str | None:
    """Return a validated adapter dir to resume from, else None.

    Raises:
        FileNotFoundError: Path configured but adapter files missing,
            fail fast instead of starting from a fresh adapter.
    """
    if not lora_config:
        return None
    path = lora_config.get("adapter_path", None)
    if not path:
        return None
    path = os.path.expanduser(path)
    required = (ADAPTER_WEIGHTS_FILE, ADAPTER_CONFIG_FILE)
    missing = [
        name for name in required
        if not os.path.exists(os.path.join(path, name))
    ]
    if missing:
        raise FileNotFoundError(
            f"lora.adapter_path={path} missing: {', '.join(missing)}")
    return path
