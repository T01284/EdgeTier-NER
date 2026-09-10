#!/usr/bin/env python3
"""Start GPU experiment pipeline on remote server (non-blocking)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_ssh import connect, run


def main() -> int:
    client = connect()
    steps = [
        "chmod +x /root/EdgeFS_NER/scripts/run_gpu_pipeline.sh",
        "mkdir -p /root/autodl-tmp/EdgeFS_NER/logs",
        "nohup bash /root/EdgeFS_NER/scripts/run_gpu_pipeline.sh "
        "> /root/autodl-tmp/EdgeFS_NER/logs/nohup.out 2>&1 & echo STARTED",
    ]
    for cmd in steps:
        code, out, err = run(client, f"bash -lc {cmd!r}", timeout=30)
        print(out or err, end="")
        if code != 0 and "nohup" not in cmd:
            print(err)
            return code

    time.sleep(8)
    code, out, err = run(
        client,
        "bash -lc 'tail -30 /root/autodl-tmp/EdgeFS_NER/logs/nohup.out 2>/dev/null'",
        timeout=30,
    )
    print("--- log tail ---")
    print(out or err or "(empty)")
    code2, out2, _ = run(
        client,
        "bash -lc 'ps aux | grep run_gpu_pipeline | grep -v grep'",
        timeout=15,
    )
    print("--- process ---")
    print(out2 or "(not running)")
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
