#!/usr/bin/env python3
"""Upload mirror fixes and restart GPU experiments."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_ssh import connect, run

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/root/EdgeFS_NER"
FILES = [
    "src/edgefs/utils/hf_mirror.py",
    "src/edgefs/data/prepare.py",
    "src/edgefs/data/bert_dataset.py",
    "src/edgefs/models/bert_crf.py",
    "scripts/prepare_datasets.py",
    "scripts/train_full.py",
    "scripts/train_fewshot.py",
    "scripts/run_gpu_experiments.sh",
]


def main() -> int:
    client = connect()
    sftp = client.open_sftp()
    for rel in FILES:
        local = ROOT / rel
        remote = f"{REMOTE}/{rel.replace(chr(92), '/')}"
        remote_dir = remote.rsplit("/", 1)[0]
        run(client, f"mkdir -p {remote_dir}", timeout=15)
        print(f"put {local.name} -> {remote}")
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

    start_cmd = (
        "export HF_ENDPOINT=https://hf-mirror.com "
        "HUGGINGFACE_HUB_ENDPOINT=https://hf-mirror.com "
        "HF_HUB_ENABLE_HF_TRANSFER=0 && "
        f"nohup bash {REMOTE}/scripts/run_gpu_experiments.sh "
        "> /root/autodl-tmp/EdgeFS_NER/logs/nohup_experiments.out 2>&1 & echo STARTED"
    )
    steps = [
        f"chmod +x {REMOTE}/scripts/run_gpu_experiments.sh",
        "mkdir -p /root/autodl-tmp/EdgeFS_NER/logs",
        start_cmd,
    ]
    for cmd in steps:
        code, out, err = run(client, f"bash -lc {cmd!r}", timeout=30)
        print(out or err, end="")
        if code != 0:
            return code

    time.sleep(10)
    _, out, _ = run(
        client,
        "bash -lc 'ps aux | grep run_gpu_experiments | grep -v grep; "
        "tail -15 /root/autodl-tmp/EdgeFS_NER/logs/gpu_experiments.log | cut -c1-220'",
        timeout=30,
    )
    print("--- status ---")
    print(out)
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
