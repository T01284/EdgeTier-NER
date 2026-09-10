#!/usr/bin/env python3
"""Download training outputs from remote GPU server."""

from __future__ import annotations

import argparse
from pathlib import Path

from remote_ssh import connect, load_remote_config, run


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull remote run artifacts to local")
    parser.add_argument(
        "--remote-dir",
        default=None,
        help="Remote run directory (default: outputs/runs/conll2003_full under project_dir)",
    )
    parser.add_argument(
        "--local-dir",
        default="outputs/runs/conll2003_full",
        help="Local destination directory",
    )
    args = parser.parse_args()

    cfg = load_remote_config()
    remote = cfg["project_dir"]
    remote_dir = args.remote_dir or f"/root/autodl-tmp/EdgeFS_NER/outputs/runs/conll2003_full"
    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    client = connect(cfg)
    sftp = client.open_sftp()

    code, out, err = run(client, f"bash -lc {repr(f'ls -la {remote_dir}')}", timeout=30)
    print(out or err)

    files_to_pull = [
        "best.pt",
        "char_vocab.json",
        "word_vocab.json",
        "metrics.json",
        "train.log",
        "train.pid",
    ]

    pulled = []
    for name in files_to_pull:
        remote_path = f"{remote_dir}/{name}"
        try:
            sftp.stat(remote_path)
        except FileNotFoundError:
            print(f"skip (missing): {name}")
            continue
        local_path = local_dir / name
        print(f"pull {remote_path} -> {local_path}")
        sftp.get(remote_path, str(local_path))
        pulled.append(name)

    sftp.close()
    client.close()

    print(f"\nPulled {len(pulled)} files to {local_dir.resolve()}")
    metrics = local_dir / "metrics.json"
    if metrics.exists():
        print(metrics.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
