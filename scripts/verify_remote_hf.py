#!/usr/bin/env python3
"""Upload prepare fix and verify OntoNotes + HF on remote."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_ssh import connect, run

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/root/EdgeFS_NER"

FILES = [
    "src/edgefs/data/prepare.py",
    "scripts/test_hf_mirror.py",
    "scripts/prepare_datasets.py",
]


def main() -> int:
    client = connect()
    sftp = client.open_sftp()
    for rel in FILES:
        local = ROOT / rel
        remote = f"{REMOTE}/{rel.replace(chr(92), '/')}"
        run(client, f"mkdir -p {remote.rsplit('/', 1)[0]}", timeout=15)
        print(f"put {rel}")
        sftp.put(str(local), remote)
    sftp.close()

    run(client, f"bash -lc 'cd {REMOTE} && source .venv/bin/activate && pip install -e . -q'", timeout=120)

    _, out, err = run(
        client,
        f"bash -lc 'cd {REMOTE} && source .venv/bin/activate && "
        f"export HF_ENDPOINT=https://hf-mirror.com HUGGINGFACE_HUB_ENDPOINT=https://hf-mirror.com && "
        f"python scripts/test_hf_mirror.py'",
        timeout=120,
    )
    print("--- HF test ---")
    print(out or err)

    _, out, err = run(
        client,
        f"bash -lc 'cd {REMOTE} && source .venv/bin/activate && "
        f"python scripts/prepare_datasets.py --config configs/dataset/ontonotes_autodl.yaml'",
        timeout=600,
    )
    print("--- prepare ontonotes ---")
    print(out or err)
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
