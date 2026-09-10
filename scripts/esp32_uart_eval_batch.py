#!/usr/bin/env python3
"""Run ESP32 UART eval in batches and merge into full test results."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
EVAL = ROOT / "scripts" / "esp32_uart_eval.py"
CONLL = ROOT / "data" / "processed_remote" / "conll2003" / "test.conll"
OUT = ROOT / "deploy" / "results" / "esp32_full_eval.json"
BATCH = 100


def sentence_count() -> int:
    sys.path.insert(0, str(ROOT / "deploy" / "pi3b"))
    from edgefs_pi.conll_io import read_conll

    return len(read_conll(CONLL))


def run_batch(start: int, count: int, port: str) -> dict:
    cmd = [
        str(PY),
        str(EVAL),
        "--port",
        port,
        "--skip-upload",
        "--start",
        str(start),
        "--max-sentences",
        str(count),
        "--output",
        str(OUT.with_name(f"esp32_batch_{start}.json")),
    ]
    subprocess.run(cmd, check=True)
    return json.loads(OUT.with_name(f"esp32_batch_{start}.json").read_text(encoding="utf-8"))


def merge(parts: list[dict]) -> dict:
    from edgefs.evaluation.metrics import compute_ner_metrics

    sys.path.insert(0, str(ROOT / "deploy" / "pi3b"))
    from edgefs_pi.conll_io import read_conll

    sentences = read_conll(CONLL)
    tags_path = ROOT / "outputs" / "export" / "conll2003_full" / "deploy" / "tags.json"
    tag_names = json.loads(tags_path.read_text(encoding="utf-8"))["tags"]

    # Reconstruct from per-batch predictions stored in checkpoint files
    y_true, y_pred, latencies = [], [], []
    for part in parts:
        start = part.get("start_index", 0)
        ckpt = OUT.with_name(f"esp32_batch_{start}.json.checkpoint.json")
        if ckpt.exists():
            data = json.loads(ckpt.read_text(encoding="utf-8"))
            y_true.extend(data["y_true"])
            y_pred.extend(data["y_pred"])
            latencies.extend(data["latencies_ms"])
        else:
            # fallback: single-batch metrics only
            n = part.get("num_sentences", 0)
            latencies.extend(part.get("latency_ms_per_sentence", []))

    if y_true and y_pred:
        metrics = compute_ner_metrics(y_true, y_pred)
    else:
        metrics = parts[-1].get("metrics", {})

    ordered = sorted(latencies) if latencies else [0.0]
    return {
        "platform": "esp32_s3_int8_uart",
        "num_sentences": len(latencies),
        "metrics": metrics,
        "latency_ms_mean": sum(latencies) / len(latencies) if latencies else 0.0,
        "latency_ms_p50": ordered[len(ordered) // 2],
        "latency_ms_p95": ordered[int(len(ordered) * 0.95)],
        "latency_ms_per_sentence": latencies,
        "batches": len(parts),
        "host_reference_f1": 0.8237179487179487,
        "f1_drop_vs_host": round(0.8237179487179487 - metrics.get("f1", 0.0), 4),
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM32")
    parser.add_argument("--batch-size", type=int, default=BATCH)
    args = parser.parse_args()

    total = sentence_count()
    parts: list[dict] = []
    for start in range(0, total, args.batch_size):
        count = min(args.batch_size, total - start)
        print(f"== batch start={start} count={count} ==")
        parts.append(run_batch(start, count, args.port))

    merged = merge(parts)
    OUT.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(merged, indent=2))
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
