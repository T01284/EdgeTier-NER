#!/usr/bin/env python3
"""Sync repo to remote GPU server and install dependencies."""

from __future__ import annotations

import argparse

from remote_ssh import connect, load_remote_config, setup_remote_env, sync_project


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync EdgeFS_NER to remote GPU")
    parser.add_argument("--setup", action="store_true", help="Create venv and pip install gpu deps")
    parser.add_argument("--sync-only", action="store_true", help="Only sync files")
    args = parser.parse_args()

    cfg = load_remote_config()
    client = connect(cfg)
    try:
        print(f"Syncing -> {cfg['project_dir']}")
        sync_project(client, cfg)
        if args.setup or not args.sync_only:
            print("Setting up remote Python environment ...")
            setup_remote_env(client, cfg)
    finally:
        client.close()
    print("Done.")


if __name__ == "__main__":
    main()
