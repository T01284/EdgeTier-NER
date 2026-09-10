#!/usr/bin/env python3
"""Probe remote GPU host."""

from remote_ssh import connect, load_remote_config, run


def main() -> None:
    cfg = load_remote_config()
    print(f"Connecting {cfg['user']}@{cfg['host']}:{cfg['port']} ...")
    client = connect(cfg)
    for cmd in ["uname -a", "nvidia-smi -L", "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"]:
        code, out, err = run(client, cmd)
        print(f"$ {cmd}")
        print(out or err)
        if code != 0:
            print(f"(exit {code})")
    client.close()
    print("OK")


if __name__ == "__main__":
    main()
