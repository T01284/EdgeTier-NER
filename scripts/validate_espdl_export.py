#!/usr/bin/env python3
"""Validate ESP-DL export: espdl file + ONNX parity vs torch export wrapper."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from esp32_embed import firmware_calib_embed


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ESP-DL export artifacts")
    parser.add_argument(
        "--deploy-dir",
        default="deploy/esp32-s3/models",
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/runs/conll2003_full/best.pt",
    )
    parser.add_argument(
        "--samples",
        default="deploy/shared/sample_inputs.json",
    )
    args = parser.parse_args()

    deploy = Path(args.deploy_dir)
    espdl = deploy / "edgefs_model.espdl"
    onnx_path = deploy / "edgefs_model.onnx"
    report: dict = {"ok": False}

    if not espdl.exists():
        raise SystemExit(f"missing {espdl}")
    report["espdl_kb"] = round(espdl.stat().st_size / 1024, 1)

    calib = build_calibration_tensors(
        deploy,
        Path(args.samples),
        num_random=0,
        max_len=128,
        max_word_len=20,
        as_float=False,
    )
    sample = calib[0:1]

    full_model = load_char_cnn(Path(args.checkpoint))
    post_embed = PostEmbedEmissionWrapper(full_model)
    with torch.no_grad():
        torch_out = EmissionWrapper(full_model)(sample).numpy()
        post_out = post_embed(firmware_calib_embed(full_model, sample)).numpy()
    report["post_embed_max_abs_diff"] = float(np.max(np.abs(torch_out - post_out)))
    report["post_embed_ok"] = report["post_embed_max_abs_diff"] < 1e-4

    if onnx_path.exists():
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        embed = firmware_calib_embed(full_model, sample).detach().numpy().astype(np.float32)
        onnx_out = sess.run(None, {input_name: embed})[0]
        max_diff = float(np.max(np.abs(torch_out - onnx_out)))
        report["onnx_max_abs_diff"] = max_diff
        report["onnx_ok"] = max_diff < 0.05
    else:
        report["onnx_ok"] = False
        report["onnx_note"] = f"missing {onnx_path}"

    report["output_shape"] = list(torch_out.shape)
    report["sample_emission_mean"] = float(torch_out.mean())
    report["ok"] = report.get("onnx_ok", False) and report.get("post_embed_ok", False)

    out = deploy / "espdl_validation.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
