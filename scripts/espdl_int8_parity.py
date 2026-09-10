#!/usr/bin/env python3
"""Quantize in-process and compare INT8 PPQ executor vs float post-embed."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from esp_ppq import QuantizationSettingFactory
from esp_ppq.api.espdl_interface import espdl_quantize_onnx
from esp_ppq.executor import TorchExecutor
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "deploy" / "pi3b"))
sys.path.insert(0, str(ROOT / "scripts"))

from edgefs.data.conll import read_conll
from edgefs_pi.crf_decode import viterbi_decode
from esp32_int8_parity import PostEmbedEmissionWrapper, encode_sentence, load_model
from quantize_to_espdl import build_embedded_calibration_tensors


def build_int8_embed(model, char_ids: np.ndarray, exp: int = -5) -> np.ndarray:
    inv = float(1 << -exp)
    weight = model.word_encoder.embedding.weight.detach().numpy()
    quant = np.clip(np.round(weight * inv), -128, 127).astype(np.int8)
    out = np.zeros((1, 128, 64, 20), dtype=np.int8)
    for t in range(128):
        for j in range(20):
            out[0, t, :, j] = quant[int(char_ids[t, j])]
    return out


def main() -> None:
    deploy = ROOT / "deploy" / "esp32-s3" / "models"
    vocab = json.loads((deploy / "char_vocab.json").read_text(encoding="utf-8"))
    tags = json.loads((deploy / "tags.json").read_text(encoding="utf-8"))["tags"]
    trans = np.load(deploy / "crf_transitions.npy")
    ckpt = ROOT / "outputs/runs/conll2003_full/best.pt"
    model = load_model(ckpt)

    calib = build_embedded_calibration_tensors(
        deploy,
        ROOT / "deploy/shared/sample_inputs.json",
        model,
        num_random=61,
        max_len=128,
        max_word_len=20,
    )
    dataloader = DataLoader(TensorDataset(calib), batch_size=1, shuffle=False)
    setting = QuantizationSettingFactory.espdl_setting()
    setting.quantize_activation_setting.calib_algorithm = "minmax"
    input_shape = [1, 128, 64, 20]

    graph = espdl_quantize_onnx(
        onnx_import_file=str(deploy / "edgefs_model.onnx"),
        espdl_export_file=str(deploy / "edgefs_model.espdl"),
        calib_dataloader=dataloader,
        calib_steps=32,
        input_shape=input_shape,
        inputs=[torch.zeros(input_shape, dtype=torch.float32)],
        target="esp32s3",
        num_of_bits=8,
        collate_fn=lambda batch: batch[0],
        setting=setting,
        device="cpu",
        error_report=False,
        skip_export=True,
        export_test_values=False,
        verbose=0,
    )

    sent = read_conll("data/processed_remote/conll2003/test.conll")[0]
    char_ids = encode_sentence(" ".join(sent.tokens), vocab)
    seq_len = len(sent.tokens)
    emb_int8 = build_int8_embed(model, char_ids)

    executor = TorchExecutor(graph=graph, device="cpu")
    int8_out = executor.forward({"embedded": torch.from_numpy(emb_int8.astype(np.float32))})[0]
    em_int8 = int8_out.detach().float().numpy()[0][:seq_len]

    with torch.no_grad():
        emb_f = torch.from_numpy(emb_int8.astype(np.float32) * (2**-5))
        em_float = PostEmbedEmissionWrapper(model)(emb_f).numpy()[0][:seq_len]

    print("gold:", sent.tags[:12])
    print("float:", [tags[i] for i in viterbi_decode(em_float, trans)[:12]])
    print("int8:", [tags[i] for i in viterbi_decode(em_int8, trans)[:12]])
    print("max_abs_diff:", float(np.max(np.abs(em_int8 - em_float))))


if __name__ == "__main__":
    main()
