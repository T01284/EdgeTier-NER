#!/usr/bin/env python3
"""Upload local HF assets (OntoNotes + optional model cache) to remote GPU server."""

from __future__ import annotations

import sys
import tarfile
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_ssh import connect, load_remote_config, run

ROOT = Path(__file__).resolve().parents[1]
ONTONOTES_LOCAL = ROOT / "data" / "raw" / "ontonotes" / "dataset"
MODELS = [
    "models--bert-base-cased",
    "models--distilbert-base-cased",
    "models--huawei-noah--TinyBERT_General_4L_312D",
]


def upload_ontonotes(client, remote_data_root: str) -> None:
    if not (ONTONOTES_LOCAL / "label.json").exists():
        raise FileNotFoundError(
            f"Missing {ONTONOTES_LOCAL}; run scripts/download_hf_assets.py first"
        )
    remote_dir = f"{remote_data_root}/data/raw/ontonotes/dataset"
    run(client, f"mkdir -p {remote_dir}", timeout=30)
    sftp = client.open_sftp()
    for path in sorted(ONTONOTES_LOCAL.glob("*.json")):
        remote_path = f"{remote_dir}/{path.name}"
        print(f"put {path.name} -> {remote_path}")
        sftp.put(str(path), remote_path)
    sftp.close()
    print("OntoNotes upload done")


def upload_model_cache(client, remote_data_root: str) -> None:
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    present = [cache_root / name for name in MODELS if (cache_root / name).exists()]
    if not present:
        print("No local model cache found; skip model upload (remote mirror works for BERT)")
        return

    remote_hub = f"{remote_data_root}/.hf_cache/hub"
    run(client, f"mkdir -p {remote_hub}", timeout=30)

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tar_path = Path(tmp.name)
    try:
        with tarfile.open(tar_path, "w:gz") as tar:
            for model_dir in present:
                tar.add(model_dir, arcname=model_dir.name)
        sftp = client.open_sftp()
        remote_tar = f"{remote_hub}/hf_models.tar.gz"
        print(f"upload model cache ({tar_path.stat().st_size / 1e6:.1f} MB)")
        sftp.put(str(tar_path), remote_tar)
        sftp.close()
        code, out, err = run(
            client,
            f"cd {remote_hub} && tar -xzf hf_models.tar.gz && rm -f hf_models.tar.gz",
            timeout=600,
        )
        if code != 0:
            raise RuntimeError(err or out)
        print(f"Model cache extracted to {remote_hub}")
    finally:
        tar_path.unlink(missing_ok=True)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--models", action="store_true", help="Also upload local HF model cache")
    args = parser.parse_args()

    cfg = load_remote_config()
    remote_data_root = cfg.get("data_root", "/root/autodl-tmp/EdgeFS_NER")
    client = connect(cfg)
    upload_ontonotes(client, remote_data_root)
    if args.models:
        upload_model_cache(client, remote_data_root)
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
