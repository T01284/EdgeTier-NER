#!/usr/bin/env python3
"""Pull all available artifacts from remote AutoDL data disk."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_ssh import connect, load_remote_config, run


def pull_dir(sftp, remote_dir: str, local_dir: Path, patterns: tuple[str, ...]) -> list[str]:
    local_dir.mkdir(parents=True, exist_ok=True)
    pulled: list[str] = []
    try:
        names = sftp.listdir(remote_dir)
    except FileNotFoundError:
        print(f"missing remote dir: {remote_dir}")
        return pulled
    for name in names:
        if patterns and not any(name.endswith(p.replace("*", "")) or name == p for p in patterns):
            if not name.endswith((".pt", ".json", ".log", ".pid")):
                continue
        remote_path = f"{remote_dir}/{name}"
        local_path = local_dir / name
        try:
            sftp.get(remote_path, str(local_path))
            pulled.append(name)
            print(f"pull {remote_path} -> {local_path}")
        except Exception as exc:
            print(f"skip {name}: {exc}")
    return pulled


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--remote-base",
        default="/root/autodl-tmp/EdgeFS_NER/outputs/runs",
    )
    parser.add_argument("--local-base", default="outputs/runs")
    args = parser.parse_args()

    cfg = load_remote_config()
    client = connect(cfg)
    sftp = client.open_sftp()

    _, out, _ = run(client, f"bash -lc 'ls -la {args.remote_base} 2>/dev/null || echo NO_RUNS'", timeout=30)
    print("=== remote runs ===")
    print(out)

    RUN_NAMES = [
        "conll2003_full", "conll2003_bilstm", "conll2003_bert",
        "conll2003_distilbert", "conll2003_tinybert", "conll2003_distill",
        "conll2003_ablation_wordlevel",
        "ontonotes_full", "ontonotes_bilstm", "ontonotes_bert",
        "ontonotes_distilbert", "ontonotes_tinybert",
        "cluener_full", "cluener_bilstm", "cluener_bert",
        "cluener_distilbert", "cluener_tinybert",
        "conll2003_fewshot_5w1s", "conll2003_fewshot_5w5s",
        "ontonotes_fewshot_4w1s", "ontonotes_fewshot_4w5s",
        "conll2003_fewshot_protobert_4w1s", "conll2003_fewshot_protobert_4w5s",
        "conll2003_fewshot_nnshot_4w1s", "conll2003_fewshot_nnshot_4w5s",
        "conll2003_fewshot_structshot_4w1s", "conll2003_fewshot_structshot_4w5s",
    ]
    total = 0
    for run_name in RUN_NAMES:
        remote_dir = f"{args.remote_base}/{run_name}"
        local_dir = Path(args.local_base) / run_name
        pulled = pull_dir(sftp, remote_dir, local_dir, ())
        total += len(pulled)

    sftp.close()
    client.close()
    print(f"\nPulled {total} files total under {args.local_base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
