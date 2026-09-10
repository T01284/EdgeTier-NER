#!/usr/bin/env python3
"""Single-sentence ONNX+CRF inference on Raspberry Pi 3B."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edgefs_pi.onnx_runner import DeployAssets, OnnxNERRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Pi 3B NER inference")
    parser.add_argument("--deploy-dir", type=str, default="assets/deploy")
    parser.add_argument("--text", type=str, required=True)
    args = parser.parse_args()

    assets = DeployAssets.load(args.deploy_dir)
    runner = OnnxNERRunner(assets)
    tagged = runner.predict(args.text)
    for token, tag in tagged:
        print(f"{token}\t{tag}")


if __name__ == "__main__":
    main()
