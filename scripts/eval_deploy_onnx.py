#!/usr/bin/env python3
"""Evaluate deploy ONNX bundle on CoNLL test set: F1 + optional latency benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy" / "pi3b"))
sys.path.insert(0, str(ROOT / "src"))

from edgefs_pi.eval_conll import eval_deploy


def main() -> int:
    parser = argparse.ArgumentParser(description="Deploy ONNX CoNLL test evaluation")
    parser.add_argument(
        "--deploy-dir",
        default=str(ROOT / "outputs/export/conll2003_full/deploy"),
    )
    parser.add_argument(
        "--conll-test",
        default=str(ROOT / "data/processed_remote/conll2003/test.conll"),
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--no-benchmark", action="store_true", help="F1 only, skip latency timing")
    parser.add_argument("--max-sentences", type=int, default=None)
    parser.add_argument(
        "--output",
        default=str(ROOT / "deploy/results/host_deploy_eval.json"),
    )
    args = parser.parse_args()

    summary = eval_deploy(
        deploy_dir=Path(args.deploy_dir),
        conll_path=Path(args.conll_test),
        warmup=args.warmup,
        repeat=args.repeat,
        benchmark=not args.no_benchmark,
        max_sentences=args.max_sentences,
        platform="host_onnx_deploy",
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
