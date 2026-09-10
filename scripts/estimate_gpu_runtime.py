#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_ssh import connect, run

LOG = "/root/autodl-tmp/EdgeFS_NER/logs/gpu_retrain_iob_fix.log"

client = connect()
_, head, _ = run(client, f"bash -lc 'head -3 {LOG}'", timeout=20)
_, epoch_lines, _ = run(
    client,
    f"bash -lc \"grep '^epoch=' {LOG} | tail -20\"",
    timeout=30,
)
_, tail, _ = run(client, f"bash -lc 'tail -1 {LOG}'", timeout=20)
client.close()

print("=== start ===")
print(head)
print("=== recent epoch summaries ===")
print(epoch_lines or "(none yet)")
print("=== current ===")
print(tail)

m = re.search(r"start (\S+)", head)
start = datetime.fromisoformat(m.group(1).replace("Z", "+00:00")) if m else None
now = datetime.now(timezone.utc)
if start:
    elapsed_min = (now - start).total_seconds() / 60
    print(f"\nElapsed since start: {elapsed_min:.1f} min")
