#!/usr/bin/env python3
"""Start long-running training job on remote GPU (nohup + log file)."""

from __future__ import annotations

import argparse
import shlex
from pathlib import PurePosixPath

from edgefs.config import load_config
from remote_ssh import connect, load_remote_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Start remote training in background")
    parser.add_argument(
        "--config",
        default="configs/train/full_conll2003.yaml",
        help="Training config YAML",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Skip dataset preparation step",
    )
    args = parser.parse_args()

    train_cfg = load_config(args.config)
    dataset_config = train_cfg.get("dataset_config", "configs/dataset/conll2003.yaml")
    output_dir = train_cfg.get("train", {}).get("output_dir", "outputs/runs/default")
    log_dir_name = PurePosixPath(output_dir).name

    cfg = load_remote_config()
    client = connect(cfg)
    remote = cfg["project_dir"]
    log_dir = f"{remote}/{output_dir}"
    log_file = f"{log_dir}/train.log"
    pid_file = f"{log_dir}/train.pid"

    prepare_cmd = ""
    if not args.skip_prepare:
        prepare_cmd = (
            f"python scripts/prepare_datasets.py --config {dataset_config} && "
        )

    inner = (
        f"mkdir -p {log_dir} && cd {remote} && source .venv/bin/activate && "
        f"(nohup bash -c {shlex.quote(prepare_cmd + f'python scripts/train_full.py --config {args.config} --device cuda')} "
        f"> {log_file} 2>&1 < /dev/null &) && sleep 1 && pgrep -af train_full.py | tail -1 | awk '{{print $1}}' > {pid_file} && cat {pid_file}"
    )

    channel = client.get_transport().open_session()
    channel.exec_command(f"bash -lc {shlex.quote(inner)}")
    import time
    time.sleep(2)
    out = ""
    if channel.recv_ready():
        out = channel.recv(4096).decode("utf-8", errors="replace")
    err = ""
    if channel.recv_stderr_ready():
        err = channel.recv_stderr(4096).decode("utf-8", errors="replace")
    client.close()

    if err and not out.strip():
        raise SystemExit(f"Failed to start training: {err}")

    pid = out.strip().splitlines()[-1]
    print("Training started on remote GPU.")
    print(f"  Config: {args.config}")
    print(f"  PID: {pid}")
    print(f"  Log: {log_file}")
    print(f"  Output dir: {log_dir}/")
    print()
    print("Monitor (from this PC):")
    print("  .\\.venv\\Scripts\\python.exe scripts\\remote_tail.py --run " + log_dir_name)
    print()
    print("Or SSH manually:")
    print(f"  ssh -p {cfg['port']} {cfg['user']}@{cfg['host']}")
    print(f"  tail -f {log_file}")


if __name__ == "__main__":
    main()
