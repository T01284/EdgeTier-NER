"""Regression: deploy ONNX runner must match training dataset char encoding."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "deploy" / "pi3b"))

from edgefs.data.conll import read_conll
from edgefs.data.dataset import NERDataset
from edgefs_pi.onnx_runner import DeployAssets, OnnxNERRunner


def test_encode_matches_dataset() -> None:
    ckpt = torch.load(
        ROOT / "outputs/runs/conll2003_full/best.pt",
        map_location="cpu",
        weights_only=False,
    )
    vocab = json.loads(
        (ROOT / "outputs/runs/conll2003_full/char_vocab.json").read_text(encoding="utf-8")
    )
    tag2id = {t: i for i, t in enumerate(ckpt["tagset"])}
    sent = read_conll(ROOT / "data/processed_remote/conll2003/test.conll")[0]
    ds = NERDataset([sent], vocab["char2id"], tag2id, vocab["pad_id"], vocab["unk_id"])
    batch = ds[0]

    runner = OnnxNERRunner(
        DeployAssets.load(ROOT / "outputs/export/conll2003_full/deploy")
    )
    inp, mask = runner._encode_text(" ".join(sent.tokens))
    seq = int(mask.sum())
    for wi in range(seq):
        assert (
            batch["input_ids"][wi].tolist() == inp[0, wi].tolist()
        ), f"word index {wi} encoding mismatch"


if __name__ == "__main__":
    test_encode_matches_dataset()
    print("ok")
