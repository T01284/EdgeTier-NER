#!/usr/bin/env python3
"""Export trained checkpoint to ONNX (+ ESP-PPQ stub)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from edgefs.export.onnx_export import export_espdl_stub, export_onnx
from edgefs.export.deploy_bundle import export_deploy_bundle
from edgefs.models.char_cnn_crf import CharCNNCRF, CharCNNCRFConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Export model for edge deployment")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/export")
    parser.add_argument("--max-len", type=int, default=128)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_cfg = CharCNNCRFConfig(**ckpt["model_config"])
    model = CharCNNCRF(model_cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    out_dir = Path(args.output_dir)
    deploy_dir = out_dir / "deploy"
    onnx_path = export_onnx(
        model,
        deploy_dir / "model_emissions.onnx",
        model_cfg.vocab_size,
        args.max_len,
        model_cfg.max_word_len,
    )
    vocab_path = Path(args.checkpoint).parent / "char_vocab.json"
    export_deploy_bundle(args.checkpoint, deploy_dir, char_vocab_path=vocab_path if vocab_path.exists() else None)
    export_espdl_stub(args.checkpoint, str(out_dir / "espdl"))
    print(f"Deploy bundle: {deploy_dir}")
    print(f"ONNX: {onnx_path}")
    print(f"ESP-PPQ stub: {out_dir / 'espdl'}")


if __name__ == "__main__":
    main()
