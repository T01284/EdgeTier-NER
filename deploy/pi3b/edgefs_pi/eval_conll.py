"""CoNLL test-set evaluation for deploy ONNX pipeline."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from edgefs_pi.conll_io import read_conll
from edgefs_pi.metrics import compute_ner_metrics
from edgefs_pi.benchmark import peak_rss_mb
from edgefs_pi.onnx_runner import DeployAssets, OnnxNERRunner


@dataclass
class SentenceResult:
    sentence_id: int
    num_tokens: int
    latency_ms: float | None
    tags: list[str]


def eval_deploy(
    deploy_dir: Path,
    conll_path: Path,
    warmup: int = 3,
    repeat: int = 1,
    benchmark: bool = True,
    max_sentences: int | None = None,
    platform: str = "deploy_onnx",
) -> dict:
    assets = DeployAssets.load(deploy_dir)
    runner = OnnxNERRunner(assets)
    sentences = read_conll(conll_path)
    if max_sentences is not None:
        sentences = sentences[:max_sentences]

    if sentences:
        for _ in range(warmup):
            runner.predict(" ".join(sentences[0].tokens))

    y_true: list[list[str]] = []
    y_pred: list[list[str]] = []
    latencies: list[float] = []
    results: list[SentenceResult] = []

    t_total = time.perf_counter()
    for idx, sent in enumerate(sentences):
        text = " ".join(sent.tokens)
        latency_ms: float | None = None
        if benchmark:
            t0 = time.perf_counter()
            for _ in range(repeat):
                tagged = runner.predict(text)
            latency_ms = (time.perf_counter() - t0) * 1000.0 / repeat
            latencies.append(latency_ms)
        else:
            tagged = runner.predict(text)

        pred = [tag for _, tag in tagged]
        gold = sent.tags[: len(pred)]
        y_true.append(gold)
        y_pred.append(pred)
        results.append(
            SentenceResult(
                sentence_id=idx,
                num_tokens=len(sent.tokens),
                latency_ms=latency_ms,
                tags=pred,
            )
        )

    metrics = compute_ner_metrics(y_true, y_pred)
    onnx_path = assets.onnx_path
    summary: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform,
        "deploy_dir": str(deploy_dir),
        "conll_path": str(conll_path),
        "num_sentences": len(sentences),
        "warmup": warmup,
        "repeat": repeat,
        "benchmark": benchmark,
        "elapsed_s": round(time.perf_counter() - t_total, 2),
        "metrics": metrics,
        "peak_rss_mb": round(peak_rss_mb(), 2),
        "model_size_kb": round(onnx_path.stat().st_size / 1024, 1) if onnx_path.exists() else 0.0,
    }
    if latencies:
        ordered = sorted(latencies)
        summary["latency_ms_mean"] = sum(latencies) / len(latencies)
        summary["latency_ms_p50"] = ordered[len(ordered) // 2]
        summary["latency_ms_p95"] = ordered[int(len(ordered) * 0.95)]
        summary["latency_ms_per_sentence"] = latencies
    summary["results"] = [asdict(r) for r in results[:5]]
    summary["results_note"] = "first 5 sentence predictions stored; full preds used for metrics"
    return summary
