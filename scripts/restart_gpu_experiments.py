#!/usr/bin/env python3
"""Reinstall package on remote and restart GPU experiment pipeline."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_ssh import connect, run


def main() -> int:
    client = connect()
    steps = [
        "cd /root/EdgeFS_NER && source .venv/bin/activate && pip install -e . -q",
        "chmod +x /root/EdgeFS_NER/scripts/run_gpu_experiments.sh",
        "mkdir -p /root/autodl-tmp/EdgeFS_NER/logs",
        "nohup bash /root/EdgeFS_NER/scripts/run_gpu_experiments.sh "
        "> /root/autodl-tmp/EdgeFS_NER/logs/nohup_experiments.out 2>&1 & echo STARTED",
    ]
    for cmd in steps:
        code, out, err = run(client, f"bash -lc {cmd!r}", timeout=120)
        print(out or err, end="")
        if code != 0:
            print(err)
            client.close()
            return code

    time.sleep(10)
    _, out, _ = run(
        client,
        "bash -lc 'ps aux | grep train_full | grep -v grep; "
        "tail -8 /root/autodl-tmp/EdgeFS_NER/logs/gpu_experiments.log'",
        timeout=30,
    )
    print("--- status ---")
    print(out)
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
