"""Latency and memory benchmark helpers for Raspberry Pi 3 Model B Plus."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from edgefs_pi.onnx_runner import DeployAssets, OnnxNERRunner


@dataclass
class SentenceBenchmark:
    sentence_id: str
    text: str
    latency_ms: float
    peak_rss_mb: float
    num_tokens: int
    tags: list[str]


def peak_rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        try:
            import resource

            usage = resource.getrusage(resource.RUSAGE_SELF)
            rss_kb = usage.ru_maxrss
            if rss_kb > 10_000_000:
                return rss_kb / (1024 * 1024)
            return rss_kb / 1024
        except Exception:
            return 0.0


def run_benchmark(
    deploy_dir: str | Path,
    sample_file: str | Path,
    warmup: int = 3,
    repeat: int = 10,
) -> dict:
    assets = DeployAssets.load(deploy_dir)
    runner = OnnxNERRunner(assets)
    samples = json.loads(Path(sample_file).read_text(encoding="utf-8"))

    for _ in range(warmup):
        runner.predict(samples[0]["text"])

    results: list[SentenceBenchmark] = []
    latencies: list[float] = []
    for sample in samples:
        text = sample["text"]
        t0 = time.perf_counter()
        for _ in range(repeat):
            tagged = runner.predict(text)
        elapsed_ms = (time.perf_counter() - t0) * 1000 / repeat
        latencies.append(elapsed_ms)
        results.append(
            SentenceBenchmark(
                sentence_id=sample["id"],
                text=text,
                latency_ms=elapsed_ms,
                peak_rss_mb=peak_rss_mb(),
                num_tokens=len(text.split()),
                tags=[tag for _, tag in tagged],
            )
        )

    onnx_path = assets.onnx_path
    model_size_kb = round(onnx_path.stat().st_size / 1024, 1) if onnx_path.exists() else 0.0

    summary = {
        "platform": "raspberry_pi_3b_plus",
        "hardware": {
            "board": "Raspberry Pi 3 Model B Plus",
            "soc": "BCM2837B0",
            "cpu": "Quad-core Cortex-A53 1.4 GHz",
            "ram_mb": 1024,
        },
        "deploy_dir": str(deploy_dir),
        "warmup": warmup,
        "repeat": repeat,
        "latency_ms_mean": sum(latencies) / len(latencies),
        "latency_ms_per_sentence": latencies,
        "peak_rss_mb": max(r.peak_rss_mb for r in results),
        "model_size_kb": model_size_kb,
        "results": [asdict(r) for r in results],
    }
    return summary
