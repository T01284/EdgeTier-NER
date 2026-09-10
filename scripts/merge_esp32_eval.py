#!/usr/bin/env python3
"""Merge ESP32 eval checkpoints/partials into final full-test JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "deploy" / "pi3b"))

from edgefs.evaluation.metrics import compute_ner_metrics

HOST_F1 = 0.8237179487179487


def load_partial(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "y_true" in data and "y_pred" in data:
        return {
            "y_true": data["y_true"],
            "y_pred": data["y_pred"],
            "latencies_ms": data.get("latencies_ms") or data.get("latency_ms_per_sentence", []),
            "truncated": int(data.get("truncated_over_max_tokens", 0)),
        }
    raise ValueError(f"unsupported partial format: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parts",
        nargs="+",
        default=[
            "deploy/results/esp32_full_eval.checkpoint.json",
        ],
    )
    parser.add_argument("--output", default="deploy/results/esp32_full_eval.json")
    args = parser.parse_args()

    y_true: list[list[str]] = []
    y_pred: list[list[str]] = []
    latencies: list[float] = []
    truncated = 0
    for part in args.parts:
        chunk = load_partial(Path(part))
        y_true.extend(chunk["y_true"])
        y_pred.extend(chunk["y_pred"])
        latencies.extend(chunk["latencies_ms"])
        truncated += chunk["truncated"]

    metrics = compute_ner_metrics(y_true, y_pred)
    ordered = sorted(latencies)
    merged = {
        "platform": "esp32_s3_int8_uart",
        "num_sentences": len(latencies),
        "truncated_over_max_tokens": truncated,
        "esp32_max_tokens": 64,
        "metrics": metrics,
        "latency_ms_mean": sum(latencies) / len(latencies),
        "latency_ms_p50": ordered[len(ordered) // 2],
        "latency_ms_p95": ordered[int(len(ordered) * 0.95)],
        "latency_ms_per_sentence": latencies,
        "host_reference_f1": HOST_F1,
        "f1_drop_vs_host": round(HOST_F1 - metrics["f1"], 4),
        "merged_parts": args.parts,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(merged, indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
