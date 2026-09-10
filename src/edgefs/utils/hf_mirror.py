"""HuggingFace Hub mirror for mainland China (AutoDL / SeetaCloud)."""

from __future__ import annotations

import os

# https://hf-mirror.com — community mirror, widely used on AutoDL
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"


def use_hf_mirror(endpoint: str | None = None) -> str:
    """Set HF hub endpoint env vars before any huggingface_hub / transformers call."""
    ep = (endpoint or os.environ.get("HF_ENDPOINT") or DEFAULT_HF_ENDPOINT).rstrip("/")
    os.environ["HF_ENDPOINT"] = ep
    os.environ["HUGGINGFACE_HUB_ENDPOINT"] = ep
    # Disable hf_transfer; can be flaky on some hosts
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    return ep
