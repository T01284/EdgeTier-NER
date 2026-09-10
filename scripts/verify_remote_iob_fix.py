#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_ssh import connect, run

client = connect()
cmds = [
    "grep -A5 'def normalize_ner_tag' /root/EdgeFS_NER/src/edgefs/data/schema.py",
    (
        "grep -E 'epoch=|dev_f1=' /root/autodl-tmp/EdgeFS_NER/logs/gpu_retrain_iob_fix.log | tail -15"
    ),
]
for cmd in cmds:
    _, out, err = run(client, f"bash -lc {cmd!r}", timeout=120)
    print(f"=== {cmd[:70]} ===")
    print(out or err)
client.close()
