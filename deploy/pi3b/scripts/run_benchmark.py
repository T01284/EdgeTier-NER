#!/usr/bin/env python3
"""Benchmark latency and peak RSS on Raspberry Pi 3B."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edgefs_pi.benchmark import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Pi 3B benchmark")
    parser.add_argument("--deploy-dir", type=str, default="assets/deploy")
    parser.add_argument(
        "--samples",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "shared" / "sample_inputs.json"),
    )
    parser.add_argument("--output", type=str, default="results/pi3b_benchmark.json")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    args = parser.parse_args()

    summary = run_benchmark(
        deploy_dir=args.deploy_dir,
        sample_file=args.samples,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
