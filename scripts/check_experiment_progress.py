#!/usr/bin/env python3
"""Detailed remote GPU experiment progress."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_ssh import connect, run

REMOTE_LOG = "/root/autodl-tmp/EdgeFS_NER/logs/gpu_experiments.log"
REMOTE_RUNS = "/root/autodl-tmp/EdgeFS_NER/outputs/runs"


def main() -> int:
    c = connect()

    _, out, _ = run(c, "ps aux | grep train_ | grep -v grep", timeout=30)
    print("=== running processes ===")
    print(out.strip() or "(none)")

    _, out, _ = run(
        c,
        f"grep -E '^==|^=== GPU|done' {REMOTE_LOG} | tail -30",
        timeout=30,
    )
    print("\n=== pipeline stages (recent) ===")
    print(out.strip() or "(empty)")

    _, out, _ = run(
        c,
        f"find {REMOTE_RUNS} -name metrics.json -o -name fewshot_summary.json | sort",
        timeout=30,
    )
    print("\n=== completed artifacts ===")
    print(out.strip() or "(none)")

    _, out, _ = run(c, f"tail -3 {REMOTE_LOG}", timeout=30)
    print("\n=== log tail ===")
    print(out.strip())

    _, out, _ = run(
        c,
        f"grep -c 'experiments done\\|pipeline done' {REMOTE_LOG} || true",
        timeout=15,
    )
    done = out.strip()
    print(f"\n=== completion markers: {done} ===")

    c.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
