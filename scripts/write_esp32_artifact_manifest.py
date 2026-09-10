#!/usr/bin/env python3
"""Write ESP32 deploy artifact manifest + firmware build-info header for reproducibility."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEPLOY = ROOT / "deploy" / "esp32-s3" / "models"
DEFAULT_OUT_HEADER = ROOT / "deploy" / "esp32-s3" / "include" / "edge_build_info.h"
MANIFEST_NAME = "artifact_manifest.json"

TRACKED_FILES = (
    "edgefs_model.espdl",
    "edgefs_model.onnx",
    "deploy_manifest.json",
    "char_vocab.json",
    "crf_transitions.npy",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_git(root: Path, *args: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def git_info(root: Path) -> dict:
    commit = _run_git(root, "rev-parse", "HEAD")
    short = _run_git(root, "rev-parse", "--short", "HEAD")
    branch = _run_git(root, "rev-parse", "--abbrev-ref", "HEAD")
    dirty = _run_git(root, "status", "--porcelain") not in (None, "")
    describe = _run_git(root, "describe", "--tags", "--always", "--dirty")
    return {
        "commit": commit,
        "commit_short": short,
        "branch": branch,
        "dirty": dirty,
        "describe": describe,
    }


def file_record(deploy_dir: Path, name: str) -> dict | None:
    path = deploy_dir / name
    if not path.exists():
        return None
    stat = path.stat()
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "size_bytes": stat.st_size,
        "sha256": sha256_file(path),
    }


def default_tag(espdl_sha256: str) -> str:
    return f"edgefs-esp32-{espdl_sha256[:8]}"


def build_manifest(
    deploy_dir: Path,
    *,
    tag: str | None = None,
    paper_reference: str | None = None,
    platformio_env: str = "esp32-s3-n16r8-espdl-uart",
    checkpoint: str | None = None,
    quantize: dict | None = None,
) -> dict:
    deploy_dir = Path(deploy_dir)
    espdl = deploy_dir / "edgefs_model.espdl"
    if not espdl.exists():
        raise FileNotFoundError(f"missing {espdl}")

    espdl_sha = sha256_file(espdl)
    artifact_tag = tag or default_tag(espdl_sha)

    deploy_manifest_path = deploy_dir / "deploy_manifest.json"
    deploy_manifest = {}
    if deploy_manifest_path.exists():
        deploy_manifest = json.loads(deploy_manifest_path.read_text(encoding="utf-8"))
    if checkpoint is None:
        checkpoint = deploy_manifest.get("checkpoint")

    checkpoint_record = None
    if checkpoint:
        ckpt_path = ROOT / checkpoint if not Path(checkpoint).is_absolute() else Path(checkpoint)
        if ckpt_path.exists():
            checkpoint_record = {
                "path": str(ckpt_path.relative_to(ROOT)).replace("\\", "/"),
                "size_bytes": ckpt_path.stat().st_size,
                "sha256": sha256_file(ckpt_path),
            }

    files: dict[str, dict] = {}
    for name in TRACKED_FILES:
        rec = file_record(deploy_dir, name)
        if rec is not None:
            files[name] = rec

    include_dir = deploy_dir.parent / "include"
    for header in sorted(include_dir.glob("edge_*.h")):
        if header.name == "edge_build_info.h":
            continue
        files[f"include/{header.name}"] = {
            "path": str(header.relative_to(ROOT)).replace("\\", "/"),
            "size_bytes": header.stat().st_size,
            "sha256": sha256_file(header),
        }

    manifest = {
        "artifact_tag": artifact_tag,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "paper_reference": paper_reference or "Neurocomputing submission - Table 3 ESP32-S3 INT8",
        "git": git_info(ROOT),
        "firmware": {
            "platformio_env": platformio_env,
            "board": "esp32-s3-wroom-n8r8",
            "edge_max_seq_len": 128,
            "edge_max_word_len": 20,
            "edge_num_tags": int(deploy_manifest.get("num_tags", 8)),
        },
        "quantize": quantize
        or {
            "backend": "torch",
            "bits": 8,
            "calib_algorithm": "minmax",
            "target": "esp32s3",
        },
        "checkpoint": checkpoint_record,
        "files": files,
        "primary_espdl_sha256": espdl_sha,
    }
    manifest["git"]["artifact_tag"] = artifact_tag
    return manifest


def write_build_info_header(manifest: dict, out_path: Path) -> None:
    tag = manifest["artifact_tag"]
    espdl_sha = manifest["primary_espdl_sha256"]
    commit = manifest["git"].get("commit_short") or "unknown"
    lines = [
        "#pragma once",
        "",
        f'#define EDGE_ARTIFACT_TAG "{tag}"',
        f'#define EDGE_ESPDL_SHA256 "{espdl_sha}"',
        f'#define EDGE_ESPDL_SHA256_PREFIX "{espdl_sha[:8]}"',
        f'#define EDGE_GIT_COMMIT "{commit}"',
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(
    deploy_dir: Path | str = DEFAULT_DEPLOY,
    *,
    tag: str | None = None,
    paper_reference: str | None = None,
    platformio_env: str = "esp32-s3-n16r8-espdl-uart",
    checkpoint: str | None = None,
    quantize: dict | None = None,
    header_path: Path | str = DEFAULT_OUT_HEADER,
) -> Path:
    deploy_dir = Path(deploy_dir)
    manifest = build_manifest(
        deploy_dir,
        tag=tag,
        paper_reference=paper_reference,
        platformio_env=platformio_env,
        checkpoint=checkpoint,
        quantize=quantize,
    )
    manifest_path = deploy_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_build_info_header(manifest, Path(header_path))
    return manifest_path


def create_git_tag(tag: str, manifest_path: Path, message: str | None = None) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    espdl_sha = manifest["primary_espdl_sha256"]
    body = message or (
        f"ESP32 deploy artifact {tag}\n\n"
        f"espdl sha256: {espdl_sha}\n"
        f"manifest: {manifest_path.relative_to(ROOT)}"
    )
    existing = _run_git(ROOT, "tag", "-l", tag)
    if existing:
        raise RuntimeError(f"git tag already exists: {tag}")
    subprocess.check_call(["git", "-C", str(ROOT), "tag", "-a", tag, "-m", body])


def load_manifest(deploy_dir: Path | str = DEFAULT_DEPLOY) -> dict | None:
    path = Path(deploy_dir) / MANIFEST_NAME
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Write ESP32 artifact manifest and build header")
    parser.add_argument("--deploy-dir", default=str(DEFAULT_DEPLOY))
    parser.add_argument("--tag", default=None, help="Artifact tag (default: edgefs-esp32-<espdl_sha8>)")
    parser.add_argument("--paper-reference", default=None)
    parser.add_argument("--platformio-env", default="esp32-s3-n16r8-espdl-uart")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--git-tag",
        action="store_true",
        help="Create annotated git tag matching artifact_tag after writing manifest",
    )
    args = parser.parse_args()

    manifest_path = write_manifest(
        args.deploy_dir,
        tag=args.tag,
        paper_reference=args.paper_reference,
        platformio_env=args.platformio_env,
        checkpoint=args.checkpoint,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Wrote {manifest_path}")
    print(f"Wrote {DEFAULT_OUT_HEADER}")

    if args.git_tag:
        create_git_tag(manifest["artifact_tag"], manifest_path)
        print(f"Created git tag: {manifest['artifact_tag']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
