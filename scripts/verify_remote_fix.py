#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_ssh import connect, run

client = connect()
cmds = [
    "grep -c normalize_ner_tag /root/EdgeFS_NER/src/edgefs/data/schema.py",
    "grep -c align_emissions_for_distill /root/EdgeFS_NER/src/edgefs/training/losses.py",
    "cd /root/EdgeFS_NER && source .venv/bin/activate && python -c \"from edgefs.data.prepare import load_processed_corpus; c=load_processed_corpus('/root/autodl-tmp/EdgeFS_NER/data/processed/conll2003'); print(len(c.tagset.tags), c.tagset.tags)\"",
    "ps aux | grep train_full | grep -v grep || true",
    "tail -15 /root/autodl-tmp/EdgeFS_NER/logs/gpu_experiments.log",
]
for cmd in cmds:
    _, out, err = run(client, f"bash -lc {cmd!r}", timeout=60)
    print(f"=== {cmd[:60]} ===")
    print(out or err)
client.close()
