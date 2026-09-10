#!/usr/bin/env python3
"""Download HuggingFace assets locally (OntoNotes JSON + optional BERT weights)."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ONTONOTES_FILES = [
    "label.json",
    "train00.json",
    "train01.json",
    "train02.json",
    "train03.json",
    "valid.json",
    "test.json",
]

BERT_MODELS = [
    "bert-base-cased",
    "distilbert-base-cased",
    "huawei-noah/TinyBERT_General_4L_312D",
]


def _hf_cache_snapshot(repo_id: str, repo_type: str = "dataset") -> Path | None:
    cache_name = f"{repo_type}s--{repo_id.replace('/', '--')}"
    base = Path.home() / ".cache" / "huggingface" / "hub" / cache_name / "snapshots"
    if not base.exists():
        return None
    snaps = sorted(base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    return snaps[0] if snaps else None


def download_ontonotes(dest: Path) -> None:
    from huggingface_hub import hf_hub_download

    dest.mkdir(parents=True, exist_ok=True)
    for name in ONTONOTES_FILES:
        out = dest / name
        if out.exists() and out.stat().st_size > 0:
            print(f"skip {out.name}")
            continue
        print(f"download dataset/{name}")
        hf_hub_download(
            repo_id="tner/ontonotes5",
            filename=f"dataset/{name}",
            repo_type="dataset",
            local_dir=str(dest.parent),
        )
        cached = dest.parent / "dataset" / name
        if cached.exists() and cached != out:
            shutil.copy2(cached, out)
    label = dest / "label.json"
    if label.exists():
        tag2id = json.loads(label.read_text(encoding="utf-8"))
        print(f"OntoNotes labels: {len(tag2id)}")


def download_models(models: list[str]) -> None:
    from transformers import AutoModel, AutoTokenizer

    for name in models:
        print(f"download model {name}")
        AutoTokenizer.from_pretrained(name)
        AutoModel.from_pretrained(name)
        print(f"  cached: {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dest",
        type=Path,
        default=ROOT / "data" / "raw" / "ontonotes" / "dataset",
        help="Local OntoNotes dataset directory",
    )
    parser.add_argument("--models", action="store_true", help="Also cache BERT/DistilBERT/TinyBERT")
    parser.add_argument("--from-cache", action="store_true", help="Copy OntoNotes from HF cache only")
    args = parser.parse_args()

    if args.from_cache:
        snap = _hf_cache_snapshot("tner/ontonotes5", "dataset")
        if snap is None:
            print("HF cache for tner/ontonotes5 not found; run without --from-cache first")
            return 1
        args.dest.mkdir(parents=True, exist_ok=True)
        for name in ONTONOTES_FILES:
            src = snap / "dataset" / name
            if not src.exists():
                print(f"missing in cache: {name}")
                return 1
            shutil.copy2(src, args.dest / name)
            print(f"copied {name}")
    else:
        download_ontonotes(args.dest)

    if args.models:
        download_models(BERT_MODELS)

    print(f"OntoNotes ready at {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
