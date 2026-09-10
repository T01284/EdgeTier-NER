#!/usr/bin/env python3
"""Run a command on remote GPU server."""

from __future__ import annotations

import argparse
import sys

from remote_ssh import connect, load_remote_config, run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run command on remote GPU via SSH")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Shell command to run remotely")
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    if not args.command:
        print("Usage: remote_run.py -- python scripts/train_full.py ...")
        sys.exit(1)

    cmd = " ".join(args.command)
    if cmd.startswith("--"):
        cmd = cmd[2:].strip()

    cfg = load_remote_config()
    client = connect(cfg)
    try:
        remote_dir = cfg["project_dir"]
        inner = f"cd {remote_dir} && source .venv/bin/activate && {cmd}"
        wrapped = f"bash -lc {inner!r}"
        code, out, err = run(client, wrapped, timeout=args.timeout)
        if out:
            print(out, end="" if out.endswith("\n") else "\n")
        if err:
            print(err, file=sys.stderr, end="" if err.endswith("\n") else "\n")
        sys.exit(code)
    finally:
        client.close()


if __name__ == "__main__":
    main()
