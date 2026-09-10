#!/usr/bin/env python3
"""Full CoNLL eval through INT8 PPQ executor (expected ESP32 upper bound)."""

from __future__ import annotations

import argparse
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
from edgefs_pi.metrics import compute_ner_metrics
from esp32_embed import build_int8_embed_array, dequantize_embed, quantize_embedding_table
from esp32_int8_parity import encode_sentence, load_model
from quantize_to_espdl import build_embedded_calibration_tensors


def build_int8_embed(model, char_ids: np.ndarray, exp: int = -5) -> np.ndarray:
    quant = quantize_embedding_table(model, exp)
    return build_int8_embed_array(char_ids, quant)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conll-test", default="data/processed_remote/conll2003/test.conll")
    parser.add_argument("--output", default="deploy/results/esp32_int8_host_eval.json")
    args = parser.parse_args()

    deploy = ROOT / "deploy" / "esp32-s3" / "models"
    vocab = json.loads((deploy / "char_vocab.json").read_text(encoding="utf-8"))
    tags = json.loads((deploy / "tags.json").read_text(encoding="utf-8"))["tags"]
    trans = np.load(deploy / "crf_transitions.npy")
    model = load_model(ROOT / "outputs/runs/conll2003_full/best.pt")

    calib = build_embedded_calibration_tensors(
        deploy,
        ROOT / "deploy/shared/sample_inputs.json",
        model,
        num_random=61,
        max_len=128,
        max_word_len=20,
    )
    setting = QuantizationSettingFactory.espdl_setting()
    setting.quantize_activation_setting.calib_algorithm = "minmax"
    graph = espdl_quantize_onnx(
        onnx_import_file=str(deploy / "edgefs_model.onnx"),
        espdl_export_file=str(deploy / "edgefs_model.espdl"),
        calib_dataloader=DataLoader(TensorDataset(calib), batch_size=1, shuffle=False),
        calib_steps=32,
        input_shape=[1, 128, 64, 20],
        inputs=[torch.zeros([1, 128, 64, 20], dtype=torch.float32)],
        target="esp32s3",
        num_of_bits=8,
        collate_fn=lambda batch: batch[0],
        setting=setting,
        device="cpu",
        error_report=False,
        skip_export=True,
        verbose=0,
    )
    executor = TorchExecutor(graph=graph, device="cpu")

    y_true: list[list[str]] = []
    y_pred: list[list[str]] = []
    for sent in read_conll(args.conll_test):
        char_ids = encode_sentence(" ".join(sent.tokens), vocab)
        n = len(sent.tokens)
        emb = build_int8_embed(model, char_ids)
        emb_f = dequantize_embed(emb)
        em = executor.forward({"embedded": torch.from_numpy(emb_f)})[0]
        em_np = em.detach().float().numpy()[0][:n]
        path = viterbi_decode(em_np, trans)
        y_true.append(sent.tags[:n])
        y_pred.append([tags[i] for i in path])

    metrics = compute_ner_metrics(y_true, y_pred)
    out = {
        "platform": "esp32_int8_ppq_host",
        "num_sentences": len(y_true),
        "metrics": metrics,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
