#!/usr/bin/env python3
"""Full CoNLL test evaluation + benchmark on Raspberry Pi 3B."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edgefs_pi.eval_conll import eval_deploy


def main() -> None:
    parser = argparse.ArgumentParser(description="Pi 3B full CoNLL deploy eval")
    parser.add_argument("--deploy-dir", type=str, default="assets/deploy")
    parser.add_argument(
        "--conll-test",
        type=str,
        default=str(ROOT / "data" / "processed_remote" / "conll2003" / "test.conll"),
    )
    parser.add_argument("--output", type=str, default="results/pi3b_full_eval.json")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--max-sentences", type=int, default=None)
    args = parser.parse_args()

    summary = eval_deploy(
        deploy_dir=Path(args.deploy_dir),
        conll_path=Path(args.conll_test),
        warmup=args.warmup,
        repeat=args.repeat,
        benchmark=True,
        max_sentences=args.max_sentences,
        platform="raspberry_pi_3b_plus",
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
