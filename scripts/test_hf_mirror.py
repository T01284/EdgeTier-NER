#!/usr/bin/env python3
"""Test HuggingFace mirror connectivity."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edgefs.utils.hf_mirror import use_hf_mirror

use_hf_mirror()
print("endpoint:", os.environ.get("HF_ENDPOINT"))

ok: list[str] = []
fail: list[str] = []

try:
    from huggingface_hub import hf_hub_download

    p = hf_hub_download("bert-base-cased", "config.json", local_dir="/tmp/hf_test_bert")
    print("bert config:", p)
    ok.append("bert-base-cased/config.json")
except Exception as e:
    fail.append(f"bert config: {e}")

try:
    from edgefs.data.prepare import _load_tner_ontonotes_raw

    for raw in (
        Path("/root/autodl-tmp/EdgeFS_NER/data/raw/ontonotes"),
        Path(__file__).resolve().parents[1] / "data" / "raw" / "ontonotes",
    ):
        if (raw / "dataset" / "label.json").exists():
            train, dev, test = _load_tner_ontonotes_raw(raw)
            print("ontonotes local:", len(train), len(dev), len(test))
            ok.append("ontonotes local json")
            break
    else:
        fail.append("ontonotes: raw dataset not found (upload via upload_hf_assets.py)")
except Exception as e:
    fail.append(f"ontonotes: {e}")

try:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("distilbert-base-cased")
    print("distilbert vocab:", tok.vocab_size)
    ok.append("distilbert-base-cased tokenizer")
except Exception as e:
    fail.append(f"distilbert tokenizer: {e}")

print("OK:", ok)
print("FAIL:", fail)
sys.exit(0 if not fail else 1)
