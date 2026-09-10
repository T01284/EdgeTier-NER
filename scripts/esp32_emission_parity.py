#!/usr/bin/env python3
"""Compare ESP32 UART emissions vs Host PPQ for the same sentence."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
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
from esp32_embed import embed_exponent
from esp32_int8_parity import encode_sentence, load_model
from edge_word_encode import word_features_from_char_ids
from quantize_to_espdl import build_word_feature_calibration_tensors

DBG_RESPONSE = re.compile(
    r"OKDBG latency_us=(?P<lat>\d+) len=(?P<len>\d+) "
    r"em0=(?P<em0>[-\d.,]+)"
    r"(?: em1=(?P<em1>[-\d.,]+))?"
    r"(?: em2=(?P<em2>[-\d.,]+))?"
    r"(?: em3=(?P<em3>[-\d.,]+))?"
    r" tags=(?P<tags>[\d,]+)"
)


def build_ppq_executor(deploy: Path, model) -> TorchExecutor:
    from esp_ppq.api.interface import load_onnx_graph

    onnx_path = deploy / "edgefs_model.onnx"
    if onnx_path.exists():
        try:
            graph = load_onnx_graph(str(onnx_path))
            return TorchExecutor(graph=graph, device="cpu")
        except Exception:
            pass

    calib = build_word_feature_calibration_tensors(
        deploy,
        ROOT / "deploy/shared/sample_inputs.json",
        model,
        num_random=61,
        max_len=128,
        max_word_len=20,
    )
    num_filters = int(
        json.loads((deploy / "deploy_manifest.json").read_text(encoding="utf-8"))["model_config"][
            "num_filters"
        ]
    )
    setting = QuantizationSettingFactory.espdl_setting()
    setting.quantize_activation_setting.calib_algorithm = "minmax"
    graph = espdl_quantize_onnx(
        onnx_import_file=str(deploy / "edgefs_model.onnx"),
        espdl_export_file=str(deploy / "edgefs_model.espdl"),
        calib_dataloader=DataLoader(TensorDataset(calib), batch_size=1, shuffle=False),
        calib_steps=32,
        input_shape=[1, num_filters, 128],
        inputs=[calib[0:1]],
        target="esp32s3",
        num_of_bits=8,
        collate_fn=lambda batch: batch[0],
        setting=setting,
        device="cpu",
        error_report=False,
        skip_export=True,
        verbose=0,
    )
    return TorchExecutor(graph=graph, device="cpu")


def host_emissions(model, executor, vocab, text: str) -> tuple[np.ndarray, list[str]]:
    deploy = ROOT / "deploy" / "esp32-s3" / "models"
    tags = json.loads((deploy / "tags.json").read_text(encoding="utf-8"))["tags"]
    trans = np.load(deploy / "crf_transitions.npy")
    char_ids = encode_sentence(text, vocab)
    seq_len = len(text.split())
    embed_exp = embed_exponent(deploy)
    char_tensor = torch.from_numpy(char_ids).long().unsqueeze(0)
    feats = word_features_from_char_ids(model, char_tensor, embed_exp)
    em = executor.forward({"word_features": feats})[0]
    em_np = em.detach().float().numpy()[0][:seq_len]
    path = viterbi_decode(em_np, trans)
    pred = [tags[i] for i in path]
    return em_np, pred


def uart_evaldbg(port: str, text: str, baud: int = 115200) -> dict:
    import serial

    captured: dict = {}
    with serial.Serial(port, baudrate=baud, timeout=2.0) as ser:
        ser.setDTR(False)
        ser.setRTS(True)
        time.sleep(0.1)
        ser.setRTS(False)
        time.sleep(8.0)
        ser.reset_input_buffer()
        ser.write(b"PING\r\n")
        ser.flush()

        deadline = time.time() + 90.0
        while time.time() < deadline:
            line = ser.readline().decode("utf-8", errors="replace").strip()
            if line.startswith("READY board="):
                captured["ready_line"] = line
                break
            if line == "PONG":
                ser.write(b"PING\r\n")
                ser.flush()
        else:
            raise TimeoutError("device not READY")

        ser.write(f"EVALDBG\t{text}\r\n".encode("utf-8"))
        ser.flush()
        deadline = time.time() + 600.0
        while time.time() < deadline:
            line = ser.readline().decode("utf-8", errors="replace").strip()
            if not line:
                continue
            if line.startswith("ERR"):
                raise RuntimeError(line)
            m = DBG_RESPONSE.search(line)
            if m:
                em0 = [float(x) for x in m.group("em0").split(",") if x]
                em2 = []
                if m.group("em2"):
                    em2 = [float(x) for x in m.group("em2").split(",") if x]
                tag_ids = [int(x) for x in m.group("tags").split(",") if x]
                return {
                    "ready_line": captured.get("ready_line"),
                    "latency_us": int(m.group("lat")),
                    "len": int(m.group("len")),
                    "em0": em0,
                    "em2": em2,
                    "tag_ids": tag_ids,
                }
        raise TimeoutError("no OKDBG response")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM32")
    parser.add_argument("--sentence-index", type=int, default=0)
    parser.add_argument("--text", default="")
    parser.add_argument("--output", default="deploy/results/esp32_emission_parity.json")
    parser.add_argument("--skip-uart", action="store_true")
    args = parser.parse_args()

    deploy = ROOT / "deploy" / "esp32-s3" / "models"
    vocab = json.loads((deploy / "char_vocab.json").read_text(encoding="utf-8"))
    tags = json.loads((deploy / "tags.json").read_text(encoding="utf-8"))["tags"]
    model = load_model(ROOT / "outputs/runs/conll2003_full/best.pt")
    executor = build_ppq_executor(deploy, model)

    if args.text:
        text = args.text
    else:
        sent = read_conll("data/processed_remote/conll2003/test.conll")[args.sentence_index]
        text = " ".join(sent.tokens)

    host_em, host_tags = host_emissions(model, executor, vocab, text)
    host_em0 = host_em[0].tolist()
    host_tag_ids = [tags.index(t) for t in host_tags]

    result = {
        "text": text,
        "host_em0": host_em0,
        "host_tag_ids": host_tag_ids,
        "host_tags": host_tags[:12],
    }

    if not args.skip_uart:
        uart = uart_evaldbg(args.port, text)
        dev_em0 = uart["em0"]
        dev_em2 = uart.get("em2", [])
        dev_tags = [tags[i] for i in uart["tag_ids"]]
        diffs0 = [abs(a - b) for a, b in zip(host_em0, dev_em0)]
        host_em2 = host_em[2].tolist() if len(host_em) > 2 else []
        diffs2 = [abs(a - b) for a, b in zip(host_em2, dev_em2)]
        result.update(
            {
                "ready_line": uart.get("ready_line"),
                "device_em0": dev_em0,
                "device_em2": dev_em2,
                "host_em2": host_em2,
                "device_tag_ids": uart["tag_ids"],
                "device_tags": dev_tags[:12],
                "max_abs_diff_t0": max(diffs0) if diffs0 else None,
                "mean_abs_diff_t0": float(np.mean(diffs0)) if diffs0 else None,
                "max_abs_diff_t2": max(diffs2) if diffs2 else None,
                "mean_abs_diff_t2": float(np.mean(diffs2)) if diffs2 else None,
                "latency_us": uart["latency_us"],
            }
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
