#!/usr/bin/env python3
"""Tail remote training log."""

from __future__ import annotations

import argparse

from remote_ssh import connect, load_remote_config, run


def main() -> None:
    parser = argparse.ArgumentParser(description="Tail remote training log")
    parser.add_argument(
        "--run",
        default="conll2003_full",
        help="Run folder name under outputs/runs/",
    )
    parser.add_argument("--lines", type=int, default=40, help="Number of lines to show")
    parser.add_argument(
        "--log",
        default=None,
        help="Remote log path override",
    )
    args = parser.parse_args()

    cfg = load_remote_config()
    remote = cfg["project_dir"]
    run_dir = f"{remote}/outputs/runs/{args.run}"
    log_file = args.log or f"{run_dir}/train.log"
    pid_file = f"{run_dir}/train.pid"

    client = connect(cfg)
    for cmd in [
        f"test -f {pid_file} && echo PID=$(cat {pid_file}) || echo PID=unknown",
        f"ps -p $(cat {pid_file} 2>/dev/null) -o pid,cmd 2>/dev/null || echo process_not_running",
        f"tail -n {args.lines} {log_file} 2>/dev/null || echo 'log not created yet'",
    ]:
        code, out, err = run(client, f"bash -lc {repr(cmd)}", timeout=30)
        print(out or err)
    client.close()


if __name__ == "__main__":
    main()
