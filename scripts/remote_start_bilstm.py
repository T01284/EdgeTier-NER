#!/usr/bin/env python3
"""Download GloVe and start BiLSTM training on remote (single background job)."""

from __future__ import annotations

import argparse
import shlex
from pathlib import PurePosixPath

from edgefs.config import load_config
from remote_ssh import connect, load_remote_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Download GloVe then start BiLSTM training")
    parser.add_argument(
        "--config",
        default="configs/train/full_conll2003_bilstm.yaml",
    )
    parser.add_argument("--skip-prepare", action="store_true")
    args = parser.parse_args()

    train_cfg = load_config(args.config)
    dataset_config = train_cfg.get("dataset_config", "configs/dataset/conll2003.yaml")
    output_dir = train_cfg.get("train", {}).get("output_dir", "outputs/runs/conll2003_bilstm")
    embed_dim = int(train_cfg.get("embeddings", {}).get("dim", 100))

    cfg = load_remote_config()
    remote = cfg["project_dir"]
    log_dir = f"{remote}/{output_dir}"
    log_file = f"{log_dir}/train.log"
    pid_file = f"{log_dir}/train.pid"

    prepare = ""
    if not args.skip_prepare:
        prepare = f"python scripts/prepare_datasets.py --config {dataset_config} && "

    inner = (
        f"mkdir -p {log_dir} && cd {remote} && source .venv/bin/activate && "
        f"(nohup bash -c {shlex.quote(prepare + 'python scripts/download_glove.py --dim ' + str(embed_dim) + ' && python scripts/train_full.py --config ' + args.config + ' --device cuda')} "
        f"> {log_file} 2>&1 < /dev/null &) && sleep 1 && pgrep -af train_full.py | tail -1 | awk '{{print $1}}' > {pid_file} && cat {pid_file}"
    )

    client = connect(cfg)
    channel = client.get_transport().open_session()
    channel.exec_command(f"bash -lc {shlex.quote(inner)}")
    import time
    time.sleep(2)
    out = channel.recv(4096).decode("utf-8", errors="replace") if channel.recv_ready() else ""
    client.close()

    run_name = PurePosixPath(output_dir).name
    pid = out.strip().splitlines()[-1] if out.strip() else "unknown"
    print("Remote job started (GloVe download + BiLSTM train).")
    print(f"  Config: {args.config}")
    print(f"  PID: {pid}")
    print(f"  Log: {log_file}")
    print(f"  Monitor: .\\.venv\\Scripts\\python.exe scripts\\remote_tail.py --run {run_name}")


if __name__ == "__main__":
    main()
