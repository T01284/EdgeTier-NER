#!/usr/bin/env python3
"""Verify GPU experiment pipeline checkpoints (local or remote paths)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Phase 1: everything before OntoNotes full training (CoNLL + prep + few-shot ablations).
PHASE1_CHECKS: list[tuple[str, str, str]] = [
    ("prepare CoNLL-2003", "data/processed_remote/conll2003/train.conll", "file"),
    ("prepare OntoNotes", "data/processed_remote/ontonotes/train.conll", "file"),
    ("prepare CLUENER", "data/processed_remote/cluener/train.conll", "file"),
    ("CoNLL Ours", "outputs/runs/conll2003_full/metrics.json", "metrics"),
    ("CoNLL BiLSTM", "outputs/runs/conll2003_bilstm/metrics.json", "metrics"),
    ("CoNLL BiLSTM-v3", "outputs/runs/conll2003_bilstm_v3/metrics.json", "metrics"),
    ("CoNLL word-level ablation", "outputs/runs/conll2003_ablation_wordlevel/metrics.json", "metrics"),
    ("CoNLL distill", "outputs/runs/conll2003_distill/metrics.json", "metrics"),
    ("CoNLL BERT", "outputs/runs/conll2003_bert/metrics.json", "metrics"),
    ("CoNLL DistilBERT", "outputs/runs/conll2003_distilbert/metrics.json", "metrics"),
    ("CoNLL TinyBERT", "outputs/runs/conll2003_tinybert/metrics.json", "metrics"),
    ("few-shot 5w1s", "outputs/runs/conll2003_fewshot_5w1s/fewshot_summary.json", "fewshot"),
    ("few-shot 5w5s", "outputs/runs/conll2003_fewshot_5w5s/fewshot_summary.json", "fewshot"),
    ("few-shot proto-only", "outputs/runs/conll2003_fewshot_proto_only/fewshot_summary.json", "fewshot"),
    ("few-shot contrastive-only", "outputs/runs/conll2003_fewshot_contrastive_only/fewshot_summary.json", "fewshot"),
    ("few-shot ProtoBERT 4w1s", "outputs/runs/conll2003_fewshot_protobert_4w1s/fewshot_summary.json", "fewshot"),
    ("few-shot ProtoBERT 4w5s", "outputs/runs/conll2003_fewshot_protobert_4w5s/fewshot_summary.json", "fewshot"),
    ("few-shot NNShot 4w1s", "outputs/runs/conll2003_fewshot_nnshot_4w1s/fewshot_summary.json", "fewshot"),
    ("few-shot NNShot 4w5s", "outputs/runs/conll2003_fewshot_nnshot_4w5s/fewshot_summary.json", "fewshot"),
    ("few-shot StructShot 4w1s", "outputs/runs/conll2003_fewshot_structshot_4w1s/fewshot_summary.json", "fewshot"),
    ("few-shot StructShot 4w5s", "outputs/runs/conll2003_fewshot_structshot_4w5s/fewshot_summary.json", "fewshot"),
]

PHASE2_CHECKS: list[tuple[str, str, str]] = [
    ("OntoNotes Ours", "outputs/runs/ontonotes_full/metrics.json", "metrics"),
    ("OntoNotes BiLSTM", "outputs/runs/ontonotes_bilstm/metrics.json", "metrics"),
    ("OntoNotes BERT", "outputs/runs/ontonotes_bert/metrics.json", "metrics"),
    ("OntoNotes DistilBERT", "outputs/runs/ontonotes_distilbert/metrics.json", "metrics"),
    ("OntoNotes TinyBERT", "outputs/runs/ontonotes_tinybert/metrics.json", "metrics"),
    ("CLUENER Ours", "outputs/runs/cluener_full/metrics.json", "metrics"),
    ("CLUENER BiLSTM", "outputs/runs/cluener_bilstm/metrics.json", "metrics"),
    ("CLUENER BERT", "outputs/runs/cluener_bert/metrics.json", "metrics"),
    ("CLUENER DistilBERT", "outputs/runs/cluener_distilbert/metrics.json", "metrics"),
    ("CLUENER TinyBERT", "outputs/runs/cluener_tinybert/metrics.json", "metrics"),
    ("OntoNotes few-shot 5w1s", "outputs/runs/ontonotes_fewshot_5w1s/fewshot_summary.json", "fewshot"),
    ("OntoNotes few-shot 5w5s", "outputs/runs/ontonotes_fewshot_5w5s/fewshot_summary.json", "fewshot"),
    ("export ONNX", "outputs/export/conll2003_full/deploy/model_emissions.onnx", "file"),
]


def _check_one(base: Path, rel: str, kind: str) -> tuple[bool, str]:
    path = base / rel
    if kind == "file":
        if not path.is_file() or path.stat().st_size == 0:
            return False, "missing or empty"
        return True, f"ok ({path.stat().st_size} bytes)"

    if not path.is_file():
        return False, "missing"

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, f"invalid json: {exc}"

    if kind == "metrics":
        f1 = (data.get("test") or {}).get("f1")
        if f1 is None:
            return False, "no test.f1"
        f1v = float(f1)
        if f1v <= 0.0 and "cluener" in rel.lower():
            return False, f"test_f1={f1v:.4f} (broken labels?)"
        return True, f"test_f1={f1v:.4f}"

    if kind == "fewshot":
        if not isinstance(data, list) or not data:
            return False, "empty summary"
        f1s = [float(row["f1"]) for row in data if "f1" in row]
        if not f1s:
            return False, "no f1 values"
        mean = sum(f1s) / len(f1s)
        return True, f"mean_f1={mean:.4f} n={len(f1s)}"

    return True, "ok"


def run_checks(base: Path, checks: list[tuple[str, str, str]]) -> tuple[int, int]:
    ok = 0
    for name, rel, kind in checks:
        passed, detail = _check_one(base, rel, kind)
        mark = "PASS" if passed else "FAIL"
        print(f"[{mark}] {name}: {detail}")
        if passed:
            ok += 1
    return ok, len(checks)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify experiment pipeline checkpoints")
    parser.add_argument(
        "--base",
        type=Path,
        default=ROOT,
        help="Project root containing outputs/ and data/",
    )
    parser.add_argument(
        "--phase",
        choices=("1", "2", "all"),
        default="all",
        help="Which phase to verify (1=pre-OntoNotes, 2=remaining)",
    )
    args = parser.parse_args()
    base = args.base.resolve()

    total_ok = 0
    total = 0
    if args.phase in ("1", "all"):
        print("=== Phase 1 (CoNLL + prep + few-shot ablations) ===")
        ok, n = run_checks(base, PHASE1_CHECKS)
        total_ok += ok
        total += n
        print()

    if args.phase in ("2", "all"):
        print("=== Phase 2 (OntoNotes + CLUENER + export) ===")
        ok, n = run_checks(base, PHASE2_CHECKS)
        total_ok += ok
        total += n
        print()

    print(f"Summary: {total_ok}/{total} passed")
    if args.phase == "1":
        return 0 if total_ok == total else 1
    return 0 if total_ok == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
