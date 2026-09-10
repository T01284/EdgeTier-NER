#!/usr/bin/env python3
"""Test HuggingFace mirror connectivity on remote."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_ssh import connect, run

TEST = r'''
cd /root/EdgeFS_NER && source .venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACE_HUB_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0
python - <<'PY'
from edgefs.utils.hf_mirror import use_hf_mirror
use_hf_mirror()
print("endpoint:", __import__("os").environ.get("HF_ENDPOINT"))

ok = []
fail = []

# 1) small HF file HEAD
try:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("bert-base-cased", "config.json", local_dir="/tmp/hf_test_bert")
    print("bert config:", p)
    ok.append("bert-base-cased/config.json")
except Exception as e:
    fail.append(f"bert config: {e}")

# 2) datasets load (small slice)
try:
    from datasets import load_dataset
    ds = load_dataset("tner", "ontonotes5", split="train[:5]")
    print("ontonotes5 rows:", len(ds), "cols:", ds.column_names)
    ok.append("tner/ontonotes5")
except Exception as e:
    fail.append(f"ontonotes5: {e}")

# 3) transformers tokenizer
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("distilbert-base-cased")
    print("distilbert vocab:", tok.vocab_size)
    ok.append("distilbert-base-cased tokenizer")
except Exception as e:
    fail.append(f"distilbert tokenizer: {e}")

print("OK:", ok)
print("FAIL:", fail)
PY
'''

c = connect()
_, out, err = run(c, f"bash -lc {TEST!r}", timeout=180)
print(out or err)
c.close()
