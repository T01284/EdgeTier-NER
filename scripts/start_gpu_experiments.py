#!/usr/bin/env python3
"""Start full GPU experiment pipeline on remote server (non-blocking)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_ssh import connect, run


def main() -> int:
    client = connect()
    steps = [
        "cd /root/EdgeFS_NER && python scripts/gen_autodl_train_configs.py",
        "chmod +x /root/EdgeFS_NER/scripts/run_gpu_experiments.sh",
        "mkdir -p /root/autodl-tmp/EdgeFS_NER/logs",
        "nohup bash /root/EdgeFS_NER/scripts/run_gpu_experiments.sh "
        "> /root/autodl-tmp/EdgeFS_NER/logs/nohup_experiments.out 2>&1 & echo STARTED",
    ]
    for cmd in steps:
        code, out, err = run(client, f"bash -lc {cmd!r}", timeout=120)
        print(out or err, end="")
        if code != 0:
            return code

    time.sleep(8)
    _, out, _ = run(
        client,
        "bash -lc 'ps aux | grep run_gpu_experiments | grep -v grep; "
        "tail -20 /root/autodl-tmp/EdgeFS_NER/logs/gpu_experiments.log 2>/dev/null | cut -c1-200'",
        timeout=30,
    )
    print("--- status ---")
    print(out)
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
