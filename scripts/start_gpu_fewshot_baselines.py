#!/usr/bin/env python3
"""Start few-shot baseline GPU pipeline on remote (non-blocking)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_ssh import connect, run

BASE = "/root/autodl-tmp/EdgeFS_NER"


def main() -> int:
    client = connect()
    steps = [
        "chmod +x /root/EdgeFS_NER/scripts/run_gpu_fewshot_baselines.sh",
        f"mkdir -p {BASE}/logs",
        f"nohup bash /root/EdgeFS_NER/scripts/run_gpu_fewshot_baselines.sh "
        f"> {BASE}/logs/nohup_fewshot_baselines.out 2>&1 & echo STARTED",
    ]
    for cmd in steps:
        code, out, err = run(client, f"bash -lc {cmd!r}", timeout=30)
        print(out or err, end="")
        if code != 0 and "nohup" not in cmd:
            print(err)
            return code

    time.sleep(8)
    _, out, _ = run(
        client,
        f"bash -lc 'tail -20 {BASE}/logs/nohup_fewshot_baselines.out 2>/dev/null'",
        timeout=30,
    )
    print("--- log tail ---")
    print(out or "(empty)")
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
