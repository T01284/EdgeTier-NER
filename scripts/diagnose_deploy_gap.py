#!/usr/bin/env python3
"""Diagnose training checkpoint F1 vs deploy ONNX F1 on CoNLL-2003."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "deploy" / "pi3b"))

from edgefs.data.conll import read_conll
from edgefs.data.dataset import NERDataset
from edgefs.evaluation.metrics import compute_ner_metrics
from edgefs.models.char_cnn_crf import CharCNNCRF, CharCNNCRFConfig
from edgefs_pi.crf_decode import viterbi_decode
from edgefs_pi.onnx_runner import DeployAssets, OnnxNERRunner


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def eval_checkpoint_on_conll(
    checkpoint: Path,
    conll_test: Path,
) -> dict:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    tagset = ckpt["tagset"]
    tag2id = {t: i for i, t in enumerate(tagset)}
    vocab_path = checkpoint.parent / "char_vocab.json"
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))

    model = CharCNNCRF(CharCNNCRFConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    sentences = [
        s for s in read_conll(conll_test) if all(t in tag2id for t in s.tags)
    ]
    ds = NERDataset(
        sentences,
        vocab["char2id"],
        tag2id,
        vocab["pad_id"],
        vocab["unk_id"],
    )
    y_true: list[list[str]] = []
    y_pred: list[list[str]] = []
    with torch.no_grad():
        for i in range(len(ds)):
            batch = ds[i]
            pred = model.decode(
                batch["input_ids"].unsqueeze(0),
                batch["mask"].unsqueeze(0),
            )[0]
            y_pred.append([tagset[t] for t in pred])
            y_true.append(
                [tagset[t] for t in batch["tags"][batch["mask"]].tolist()]
            )
    metrics = compute_ner_metrics(y_true, y_pred)
    return {
        "conll_path": str(conll_test),
        "conll_sha256": sha256_file(conll_test),
        "num_sentences": len(sentences),
        "metrics": metrics,
    }


def eval_deploy_roundtrip(
    deploy_dir: Path,
    conll_test: Path,
) -> dict:
    assets = DeployAssets.load(deploy_dir)
    runner = OnnxNERRunner(assets)
    sentences = read_conll(conll_test)
    y_true: list[list[str]] = []
    y_pred_onnx: list[list[str]] = []
    y_pred_pt: list[list[str]] = []

    ckpt_default = ROOT / "outputs/runs/conll2003_full/best.pt"
    ckpt = torch.load(ckpt_default, map_location="cpu", weights_only=False)
    model = CharCNNCRF(CharCNNCRFConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    tagset = ckpt["tagset"]
    transitions = assets.transitions

    max_emission_diff = 0.0
    with torch.no_grad():
        for sent in sentences:
            text = " ".join(sent.tokens)
            inp, mask = runner._encode_text(text)
            seq = int(mask.sum())
            ids = torch.from_numpy(inp)
            m = torch.from_numpy(mask)
            em_pt = model.classifier(model.encode(ids, m))[0, :seq].numpy()
            em_onnx = runner.session.run(
                None, runner._session_inputs(inp, mask)
            )[0][0][:seq]
            max_emission_diff = max(
                max_emission_diff, float(np.max(np.abs(em_pt - em_onnx)))
            )
            path = viterbi_decode(em_pt, transitions)
            y_pred_pt.append([tagset[i] for i in path])
            y_pred_onnx.append([t for _, t in runner.predict(text)])
            y_true.append(sent.tags[:seq])

    return {
        "deploy_dir": str(deploy_dir),
        "conll_path": str(conll_test),
        "pytorch_same_encoding": compute_ner_metrics(y_true, y_pred_pt),
        "onnx_deploy": compute_ner_metrics(y_true, y_pred_onnx),
        "predictions_identical": y_pred_pt == y_pred_onnx,
        "max_emission_abs_diff": max_emission_diff,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose train vs deploy F1 gap")
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "outputs/runs/conll2003_full/best.pt"),
    )
    parser.add_argument(
        "--conll-test",
        default=str(ROOT / "data/processed_remote/conll2003/test.conll"),
    )
    parser.add_argument(
        "--deploy-dir",
        default=str(ROOT / "outputs/export/conll2003_full/deploy"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "deploy/results/deploy_gap_diagnosis.json"),
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    conll_test = Path(args.conll_test)
    deploy_dir = Path(args.deploy_dir)
    metrics_path = checkpoint.parent / "metrics.json"
    saved = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}

    ckpt_eval = eval_checkpoint_on_conll(checkpoint, conll_test)
    deploy_eval = eval_deploy_roundtrip(deploy_dir, conll_test)

    saved_f1 = float((saved.get("test") or {}).get("f1", 0))
    local_f1 = float(ckpt_eval["metrics"]["f1"])
    deploy_f1 = float(deploy_eval["onnx_deploy"]["f1"])

    if abs(local_f1 - deploy_f1) < 0.005:
        onnx_verdict = "ONNX export matches PyTorch on deploy encoding path"
    else:
        onnx_verdict = "ONNX export diverges from PyTorch — investigate export"

    if abs(saved_f1 - local_f1) > 0.02:
        test_verdict = (
            "metrics.json test F1 differs from re-eval on local test.conll — "
            "training likely used a different processed test split (remote data/processed/)"
        )
    else:
        test_verdict = "metrics.json consistent with local test.conll"

    report = {
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "checkpoint": str(checkpoint),
        "metrics_json_test_f1": saved_f1,
        "local_checkpoint_reeval": ckpt_eval,
        "deploy_roundtrip": deploy_eval,
        "gaps": {
            "metrics_json_vs_local_checkpoint": round(saved_f1 - local_f1, 4),
            "local_checkpoint_vs_deploy_onnx": round(local_f1 - deploy_f1, 4),
            "metrics_json_vs_deploy_onnx": round(saved_f1 - deploy_f1, 4),
        },
        "verdicts": {
            "onnx_export": onnx_verdict,
            "test_set_alignment": test_verdict,
            "paper_action": (
                "Report deploy F1 (~70%) in Table 3; align Table 1 training F1 with the "
                "same test.conll used in deployment, or document that Table 1 uses the "
                "canonical training split while Table 3 uses the published deploy bundle test file."
            ),
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
