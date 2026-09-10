from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from edgefs.models.char_cnn_crf import CharCNNCRF, CharCNNCRFConfig


def export_deploy_bundle(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    char_vocab_path: str | Path | None = None,
) -> Path:
    """Export ONNX-sidecar assets shared by Pi 3B and ESP32-S3 test harnesses."""
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = ckpt["model_config"]
    model = CharCNNCRF(CharCNNCRFConfig(**model_cfg))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tags = ckpt.get("tagset") or []
    transitions = model.transition_matrix().cpu().numpy()
    np.save(output_dir / "crf_transitions.npy", transitions)

    (output_dir / "tags.json").write_text(
        json.dumps({"tags": tags}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "model_config.json").write_text(
        json.dumps(model_cfg, indent=2),
        encoding="utf-8",
    )

    if char_vocab_path and Path(char_vocab_path).exists():
        dest = output_dir / "char_vocab.json"
        dest.write_text(Path(char_vocab_path).read_text(encoding="utf-8"), encoding="utf-8")
    elif (checkpoint_path.parent / "char_vocab.json").exists():
        src = checkpoint_path.parent / "char_vocab.json"
        (output_dir / "char_vocab.json").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    manifest = {
        "checkpoint": str(checkpoint_path),
        "num_tags": len(tags),
        "tags": tags,
        "transitions_shape": list(transitions.shape),
        "model_config": model_cfg,
        "files": {
            "onnx": "model_emissions.onnx",
            "char_vocab": "char_vocab.json",
            "tags": "tags.json",
            "transitions": "crf_transitions.npy",
            "model_config": "model_config.json",
        },
    }
    (output_dir / "deploy_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_dir
