#!/usr/bin/env python3
"""Collect all experiment metrics into deploy/results/all_gpu_metrics.json."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "outputs" / "runs"


def load_metrics(run: str) -> dict | None:
    p = RUNS / run / "metrics.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def load_fewshot(run: str) -> dict | None:
    p = RUNS / run / "fewshot_summary.json"
    if not p.is_file():
        return None
    rows = json.loads(p.read_text(encoding="utf-8"))
    f1s = [float(r["f1"]) for r in rows if "f1" in r]
    if not f1s:
        return None
    mean = sum(f1s) / len(f1s)
    var = sum((x - mean) ** 2 for x in f1s) / len(f1s)
    std = var**0.5
    return {"mean_f1": mean, "std_f1": std, "n": len(f1s), "seeds": rows}


def main() -> int:
    full_runs = {
        "conll2003": [
            "conll2003_bilstm",
            "conll2003_bert",
            "conll2003_distilbert",
            "conll2003_tinybert",
            "conll2003_full",
        ],
        "ontonotes": [
            "ontonotes_bilstm",
            "ontonotes_bert",
            "ontonotes_distilbert",
            "ontonotes_tinybert",
            "ontonotes_full",
        ],
        "cluener": [
            "cluener_bilstm",
            "cluener_bert",
            "cluener_distilbert",
            "cluener_tinybert",
            "cluener_full",
        ],
    }
    fewshot_runs = {
        "conll2003_5w1s": "conll2003_fewshot_5w1s",
        "conll2003_5w5s": "conll2003_fewshot_5w5s",
        "conll2003_protobert_4w1s": "conll2003_fewshot_protobert_4w1s",
        "conll2003_protobert_4w5s": "conll2003_fewshot_protobert_4w5s",
        "conll2003_nnshot_4w1s": "conll2003_fewshot_nnshot_4w1s",
        "conll2003_nnshot_4w5s": "conll2003_fewshot_nnshot_4w5s",
        "conll2003_structshot_4w1s": "conll2003_fewshot_structshot_4w1s",
        "conll2003_structshot_4w5s": "conll2003_fewshot_structshot_4w5s",
        "ontonotes_5w1s": "ontonotes_fewshot_4w1s",
        "ontonotes_5w5s": "ontonotes_fewshot_4w5s",
        "ontonotes_4w1s": "ontonotes_fewshot_4w1s",
        "ontonotes_4w5s": "ontonotes_fewshot_4w5s",
    }
    ablations = {
        "wordlevel": "conll2003_ablation_wordlevel",
        "distill": "conll2003_distill",
    }

    out: dict = {"full": {}, "fewshot": {}, "ablation": {}}
    for ds, names in full_runs.items():
        out["full"][ds] = {}
        for name in names:
            m = load_metrics(name)
            if m:
                out["full"][ds][name] = m.get("test", {})

    for key, name in fewshot_runs.items():
        fs = load_fewshot(name)
        if fs:
            out["fewshot"][key] = fs

    for key, name in ablations.items():
        m = load_metrics(name)
        if m:
            out["ablation"][key] = m.get("test", {})

    dest = ROOT / "deploy" / "results" / "all_gpu_metrics.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
