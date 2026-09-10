#!/usr/bin/env python3
"""Aggregate host/Pi/ESP32 deploy test results into one summary."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "deploy" / "results"
HOST_F1_REF = 0.8237179487179487


def load_json(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def main() -> None:
    host = load_json(RESULTS / "host_deploy_eval.json")
    pi_full_report = load_json(RESULTS / "pi3b_full_test.json")
    pi_bench_report = load_json(RESULTS / "pi3b_test.json")
    esp32 = load_json(RESULTS / "esp32_full_eval.json")
    esp32_smoke = load_json(RESULTS / "esp32_uart_smoke.json")
    esp32_smoke20 = load_json(RESULTS / "esp32_smoke20_v3.json") or load_json(
        RESULTS / "esp32_smoke20_v2.json"
    )
    esp32_int8_host = load_json(RESULTS / "esp32_int8_host_eval.json")

    pi_summary = (pi_full_report or {}).get("summary") or {}
    pi_bench = (pi_bench_report or {}).get("summary") or {}

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host_reference_f1": round(HOST_F1_REF, 4),
        "host_onnx_deploy": {
            "f1": round((host or {}).get("metrics", {}).get("f1", 0), 4),
            "latency_ms_mean": (host or {}).get("latency_ms_mean"),
            "num_sentences": (host or {}).get("num_sentences"),
            "model_size_kb": (host or {}).get("model_size_kb"),
            "file": "deploy/results/host_deploy_eval.json",
        },
        "pi3b_benchmark": {
            "latency_ms_mean": pi_bench.get("latency_ms_mean"),
            "peak_rss_mb": pi_bench.get("peak_rss_mb"),
            "model_size_kb": pi_bench.get("model_size_kb"),
            "num_sample_sentences": len(pi_bench.get("results") or []),
            "file": "deploy/results/pi3b_test.json",
        },
        "pi3b_full": {
            "f1": round(pi_summary.get("metrics", {}).get("f1", 0), 4),
            "latency_ms_mean": pi_summary.get("latency_ms_mean"),
            "latency_ms_p95": pi_summary.get("latency_ms_p95"),
            "peak_rss_mb": pi_summary.get("peak_rss_mb"),
            "model_size_kb": pi_summary.get("model_size_kb"),
            "num_sentences": pi_summary.get("num_sentences"),
            "elapsed_s": pi_summary.get("elapsed_s"),
            "file": "deploy/results/pi3b_full_test.json",
        },
        "esp32_s3_int8": {
            "f1": round((esp32 or {}).get("metrics", {}).get("f1", 0), 4),
            "latency_ms_mean": (esp32 or {}).get("latency_ms_mean"),
            "latency_ms_p95": (esp32 or {}).get("latency_ms_p95"),
            "num_sentences": (esp32 or {}).get("num_sentences"),
            "truncated_over_max_tokens": (esp32 or {}).get("truncated_over_max_tokens"),
            "esp32_max_tokens": (esp32 or {}).get("esp32_max_tokens"),
            "artifact_tag": (esp32 or {}).get("artifact_tag"),
            "model_size_kb": 343,
            "host_int8_ppq_f1": round(
                (esp32_int8_host or {}).get("metrics", {}).get("f1", 0), 4
            ),
            "file": "deploy/results/esp32_full_eval.json",
        },
        "esp32_smoke20": {
            "f1": round((esp32_smoke20 or {}).get("metrics", {}).get("f1", 0), 4),
            "latency_ms_mean": (esp32_smoke20 or {}).get("latency_ms_mean"),
            "num_sentences": (esp32_smoke20 or {}).get("num_sentences"),
            "file": "deploy/results/esp32_smoke20_v3.json",
        },
        "cross_platform": {
            "host_vs_pi_f1_match": abs(
                (host or {}).get("metrics", {}).get("f1", 0)
                - pi_summary.get("metrics", {}).get("f1", 0)
            )
            < 1e-4,
            "host_vs_pi_f1": round(pi_summary.get("metrics", {}).get("f1", 0), 4),
            "training_checkpoint_f1": round(HOST_F1_REF, 4),
            "deploy_onnx_vs_training_gap": round(
                HOST_F1_REF - pi_summary.get("metrics", {}).get("f1", 0), 4
            ),
        },
    }
    if esp32 and esp32.get("metrics", {}).get("f1") is not None:
        summary["esp32_s3_int8"]["f1_drop_vs_host"] = round(
            HOST_F1_REF - esp32["metrics"]["f1"], 4
        )

    out = RESULTS / "deploy_all_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
