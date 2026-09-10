#!/usr/bin/env python3
"""Pull remote processed CoNLL/OntoNotes/CLUENER splits for local eval alignment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_ssh import connect, load_remote_config

REMOTE_FILES = [
    ("conll2003/test.conll", "data/processed_remote/conll2003/test.conll"),
    ("conll2003/dev.conll", "data/processed_remote/conll2003/dev.conll"),
    ("conll2003/train.conll", "data/processed_remote/conll2003/train.conll"),
    ("ontonotes/test.conll", "data/processed_remote/ontonotes/test.conll"),
    ("cluener/test.conll", "data/processed_remote/cluener/test.conll"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Pull remote processed datasets")
    parser.add_argument(
        "--remote-base",
        default="/root/autodl-tmp/EdgeFS_NER/data/processed",
    )
    parser.add_argument("--files", nargs="*", default=[f[0] for f in REMOTE_FILES])
    args = parser.parse_args()

    cfg = load_remote_config()
    client = connect(cfg)
    sftp = client.open_sftp()
    pulled = 0
    for rel in args.files:
        remote = f"{args.remote_base}/{rel}"
        local = Path("data/processed_remote") / rel
        local.parent.mkdir(parents=True, exist_ok=True)
        try:
            sftp.stat(remote)
        except FileNotFoundError:
            print(f"skip missing {remote}")
            continue
        sftp.get(remote, str(local))
        print(f"pull {remote} -> {local} ({local.stat().st_size} B)")
        pulled += 1
    sftp.close()
    client.close()
    print(f"Pulled {pulled} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
