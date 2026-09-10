#!/usr/bin/env python3
"""Upload changed files and restart GPU pipeline resume."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_ssh import connect, load_remote_config, run

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/root/EdgeFS_NER"
FILES = [
    "configs/train/fewshot_conll2003_autodl.yaml",
    "configs/train/fewshot_conll2003_5w5s_autodl.yaml",
    "configs/train/fewshot_conll2003.yaml",
    "src/edgefs/data/fewshot.py",
    "scripts/run_gpu_pipeline.sh",
]


def main() -> int:
    cfg = load_remote_config()
    client = connect(cfg)
    sftp = client.open_sftp()
    for rel in FILES:
        local = ROOT / rel
        remote = f"{REMOTE}/{rel.replace(chr(92), '/')}"
        remote_dir = remote.rsplit("/", 1)[0]
        run(client, f"mkdir -p {remote_dir}", timeout=15)
        print(f"put {local} -> {remote}")
        sftp.put(str(local), remote)
    sftp.close()

    code, out, err = run(
        client,
        f"bash -lc 'cd {REMOTE} && source .venv/bin/activate && pip install -e . -q'",
        timeout=120,
    )
    if code != 0:
        print(err or out)
        return code

    steps = [
        f"chmod +x {REMOTE}/scripts/run_gpu_pipeline.sh",
        "mkdir -p /root/autodl-tmp/EdgeFS_NER/logs",
        f"nohup bash {REMOTE}/scripts/run_gpu_pipeline.sh "
        "> /root/autodl-tmp/EdgeFS_NER/logs/nohup.out 2>&1 & echo STARTED",
    ]
    for cmd in steps:
        code, out, err = run(client, f"bash -lc {cmd!r}", timeout=30)
        print(out or err, end="")
        if code != 0:
            return code

    time.sleep(6)
    _, out, _ = run(
        client,
        "bash -lc 'ps aux | grep run_gpu_pipeline | grep -v grep; "
        "tail -20 /root/autodl-tmp/EdgeFS_NER/logs/gpu_pipeline.log | cut -c1-200'",
        timeout=30,
    )
    print("--- status ---")
    print(out)
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
