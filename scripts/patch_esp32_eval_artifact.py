#!/usr/bin/env python3
"""Backfill artifact_manifest fields into existing ESP32 eval JSON files."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from write_esp32_artifact_manifest import load_manifest

RESULT_FILES = [
    "deploy/results/esp32_full_eval.json",
    "deploy/results/esp32_smoke20_v3.json",
    "deploy/results/esp32_uart_smoke.json",
]


def patch(path: Path, manifest: dict) -> bool:
    if not path.exists():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    data["artifact_tag"] = manifest.get("artifact_tag")
    data["espdl_sha256"] = manifest.get("primary_espdl_sha256")
    data["artifact_manifest"] = manifest
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return True


def main() -> int:
    manifest = load_manifest(ROOT / "deploy/esp32-s3/models")
    if not manifest:
        raise SystemExit("missing artifact_manifest.json")
    n = sum(1 for rel in RESULT_FILES if patch(ROOT / rel, manifest))
    print(f"Patched {n} result files with tag={manifest.get('artifact_tag')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
