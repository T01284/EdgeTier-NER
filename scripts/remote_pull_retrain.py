#!/usr/bin/env python3
"""Pull all experiment run artifacts from remote GPU server."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_ssh import connect, load_remote_config, run


RUN_DIRS = [
    "ontonotes_full",
    "ontonotes_bilstm",
    "ontonotes_bert",
    "ontonotes_distilbert",
    "ontonotes_tinybert",
    "cluener_full",
    "cluener_bilstm",
    "cluener_bert",
    "cluener_distilbert",
    "cluener_tinybert",
    "ontonotes_fewshot_5w1s",
    "ontonotes_fewshot_5w5s",
]


def pull_tree(sftp, remote_dir: str, local_dir: Path) -> int:
    local_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    try:
        entries = sftp.listdir_attr(remote_dir)
    except FileNotFoundError:
        print(f"missing: {remote_dir}")
        return 0

    for ent in entries:
        name = ent.filename
        if name in (".", ".."):
            continue
        remote_path = f"{remote_dir}/{name}"
        local_path = local_dir / name
        if ent.st_mode & 0o40000:
            count += pull_tree(sftp, remote_path, local_path)
        else:
            if name.endswith((".json", ".pt", ".log", ".txt")):
                sftp.get(remote_path, str(local_path))
                count += 1
                print(f"pull {remote_path}")
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--remote-base",
        default="/root/autodl-tmp/EdgeFS_NER/outputs/runs",
    )
    parser.add_argument("--local-base", default="outputs/runs")
    parser.add_argument("--runs", nargs="*", default=RUN_DIRS)
    args = parser.parse_args()

    cfg = load_remote_config()
    client = connect(cfg)
    sftp = client.open_sftp()

    _, out, _ = run(
        client,
        f"bash -lc 'tail -5 /root/autodl-tmp/EdgeFS_NER/logs/gpu_retrain_iob_fix.log 2>/dev/null'",
        timeout=30,
    )
    print("=== remote log tail ===")
    print(out)

    total = 0
    for run_name in args.runs:
        remote_dir = f"{args.remote_base}/{run_name}"
        local_dir = Path(args.local_base) / run_name
        total += pull_tree(sftp, remote_dir, local_dir)

    sftp.close()
    client.close()
    print(f"\nPulled {total} files under {args.local_base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
