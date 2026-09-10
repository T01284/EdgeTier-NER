#!/usr/bin/env python3
"""Tail remote GPU pipeline log."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_ssh import connect, run


def main() -> int:
    client = connect()
    for cmd in [
        "ps aux | grep run_gpu_pipeline | grep -v grep | wc -l",
        "tail -15 /root/autodl-tmp/EdgeFS_NER/logs/nohup.out",
        "ls -la /root/autodl-tmp/EdgeFS_NER/outputs/runs/*/metrics.json 2>/dev/null || true",
    ]:
        _, out, err = run(client, f"bash -lc {cmd!r}", timeout=30)
        print(f"=== {cmd} ===")
        print(out or err)
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
