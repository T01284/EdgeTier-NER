#!/usr/bin/env python3
"""Layer-wise PPQ host vs ESP32 UART parity for calib sentence."""

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

from esp32_embed import embed_exponent
from edge_word_encode import word_features_from_char_ids
from esp32_int8_parity import encode_sentence, load_model
from quantize_to_espdl import build_word_feature_calibration_tensors

LAYER_NAMES = [
    "word_features",
    "/blocks.0/relu/Relu_output_0",
    "/blocks.0/Add_output_0",
    "/blocks.1/relu/Relu_output_0",
    "/blocks.1/Add_output_0",
    "/blocks.2/relu/Relu_output_0",
    "/blocks.2/Add_output_0",
    "/classifier/MatMul_output_0",
    "emissions",
]

LYR_LINE = re.compile(
    r"LYR name=(?P<name>\S+) found=(?P<found>\d+) size=(?P<size>\d+) "
    r"exp=(?P<exp>-?\d+) max=(?P<max>[-\d.]+) mean=(?P<mean>[-\d.]+) "
    r"samp=(?P<samp>[-\d.,]+)"
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


def tensor_stats(tensor: torch.Tensor) -> dict:
    arr = tensor.detach().float().reshape(-1).cpu().numpy()
    sample_count = min(16, arr.size)
    return {
        "size": int(arr.size),
        "max_abs": float(np.max(np.abs(arr))) if arr.size else 0.0,
        "mean": float(np.mean(arr)) if arr.size else 0.0,
        "samples": arr[:sample_count].tolist(),
        "num_samples": sample_count,
    }


def host_layer_summaries(
    model,
    executor: TorchExecutor,
    vocab: dict,
    text: str,
    deploy: Path,
) -> list[dict]:
    char_ids = encode_sentence(text, vocab)
    embed_exp = embed_exponent(deploy)
    char_tensor = torch.from_numpy(char_ids).long().unsqueeze(0)
    word_features = word_features_from_char_ids(model, char_tensor, embed_exp)

    intermediate_names = [name for name in LAYER_NAMES if name not in {"word_features", "emissions"}]
    forward_out = executor.forward(
        {"word_features": word_features},
        output_names=intermediate_names + ["emissions"],
    )
    if isinstance(forward_out, dict):
        tensors = [forward_out[name] for name in intermediate_names + ["emissions"]]
    else:
        tensors = list(forward_out)

    summaries: list[dict] = []
    for name in LAYER_NAMES:
        if name == "word_features":
            stats = tensor_stats(word_features)
        elif name == "emissions":
            stats = tensor_stats(tensors[-1])
        else:
            idx = intermediate_names.index(name)
            stats = tensor_stats(tensors[idx])
        summaries.append({"name": name, "found": 1, **stats})
    return summaries


def uart_layerdump(port: str, text: str, baud: int = 115200) -> tuple[str | None, list[dict]]:
    import serial

    ready_line: str | None = None
    layers: list[dict] = []
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
                ready_line = line
                break
            if line == "PONG":
                ser.write(b"PING\r\n")
                ser.flush()
        else:
            raise TimeoutError("device not READY")

        cmd = f"LAYERDUMP\t{text}\r\n".encode("utf-8")
        ser.write(cmd)
        ser.flush()

        deadline = time.time() + 300.0
        while time.time() < deadline:
            line = ser.readline().decode("utf-8", errors="replace").strip()
            if not line:
                continue
            if line.startswith("ERR"):
                raise RuntimeError(line)
            if line == "OKLAYERS":
                return ready_line, layers
            m = LYR_LINE.search(line)
            if m:
                samples = [float(x) for x in m.group("samp").split(",") if x]
                layers.append(
                    {
                        "name": m.group("name"),
                        "found": int(m.group("found")),
                        "size": int(m.group("size")),
                        "exponent": int(m.group("exp")),
                        "max_abs": float(m.group("max")),
                        "mean": float(m.group("mean")),
                        "samples": samples,
                        "num_samples": len(samples),
                    }
                )
        raise TimeoutError("no OKLAYERS response")


def compare_layers(host: list[dict], device: list[dict], sample_tol: float, max_abs_tol: float) -> tuple[list[dict], str | None]:
    device_by_name = {layer["name"]: layer for layer in device}
    rows: list[dict] = []
    first_diverged: str | None = None

    for host_layer in host:
        name = host_layer["name"]
        dev_layer = device_by_name.get(name)
        row = {
            "name": name,
            "host_found": host_layer.get("found", 1),
            "device_found": None if dev_layer is None else dev_layer.get("found"),
            "host_max_abs": host_layer["max_abs"],
            "device_max_abs": None if dev_layer is None else dev_layer.get("max_abs"),
            "host_mean": host_layer["mean"],
            "device_mean": None if dev_layer is None else dev_layer.get("mean"),
            "host_samples": host_layer["samples"],
            "device_samples": [] if dev_layer is None else dev_layer.get("samples", []),
        }

        if dev_layer is None or not dev_layer.get("found"):
            row["status"] = "missing_on_device"
            row["max_sample_abs_diff"] = None
            row["max_abs_diff"] = None
        else:
            sample_pairs = list(zip(host_layer["samples"], dev_layer["samples"]))
            sample_diffs = [abs(a - b) for a, b in sample_pairs]
            row["max_sample_abs_diff"] = max(sample_diffs) if sample_diffs else 0.0
            row["max_abs_diff"] = abs(host_layer["max_abs"] - dev_layer["max_abs"])
            mean_diff = abs(host_layer["mean"] - dev_layer["mean"])
            row["mean_abs_diff"] = mean_diff
            # Linear index-0..15 samples differ when layout differs; trust max_abs first.
            diverged = row["max_abs_diff"] > max_abs_tol or (
                row["max_sample_abs_diff"] > sample_tol
                and (row["max_abs_diff"] > 0.05 or mean_diff > 0.05)
            )
            row["status"] = "diverged" if diverged else "ok"
            if diverged and first_diverged is None:
                first_diverged = name

        rows.append(row)

    return rows, first_diverged


def classify_root_cause(first_diverged: str | None, rows: list[dict]) -> str:
    if first_diverged is None:
        ok_rows = [r for r in rows if r["status"] == "ok"]
        if len(ok_rows) == len(rows):
            return "all_layers_match"
        return "no_clear_first_divergence"

    if first_diverged == "word_features":
        return "embedding_or_input_quantization"
    if first_diverged in {"/Relu_output_0", "/ReduceMax_output_0", "/Reshape_1_output_0"}:
        return "early_plain_conv_or_pooling"
    if first_diverged in {"/blocks.0/Transpose_output_0", "/blocks.0/relu/Relu_output_0", "/blocks.0/Add_output_0"}:
        return "blocks0_before_dilation"
    if first_diverged in {"/blocks.1/relu/Relu_output_0", "/blocks.1/Add_output_0"}:
        return "dilation_blocks1_likely_root_cause"
    if first_diverged in {"/blocks.2/relu/Relu_output_0", "/blocks.2/Add_output_0"}:
        return "dilation_blocks2_likely_root_cause"
    if first_diverged in {"emissions", "/classifier/MatMul_output_0"}:
        diverged_count = sum(1 for r in rows if r["status"] == "diverged")
        if diverged_count <= 2:
            return "late_classifier_only"
        return "accumulated_quantization_error"
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM32")
    parser.add_argument("--text", default="John works in Paris")
    parser.add_argument("--output", default="deploy/results/layerwise_parity.json")
    parser.add_argument("--sample-tol", type=float, default=0.5)
    parser.add_argument("--max-abs-tol", type=float, default=1.0)
    parser.add_argument("--skip-uart", action="store_true")
    args = parser.parse_args()

    deploy = ROOT / "deploy" / "esp32-s3" / "models"
    vocab = json.loads((deploy / "char_vocab.json").read_text(encoding="utf-8"))
    model = load_model(ROOT / "outputs/runs/conll2003_full/best.pt")
    executor = build_ppq_executor(deploy, model)

    host_layers = host_layer_summaries(model, executor, vocab, args.text, deploy)
    result: dict = {
        "text": args.text,
        "layer_names": LAYER_NAMES,
        "host_layers": host_layers,
        "thresholds": {
            "sample_tol": args.sample_tol,
            "max_abs_tol": args.max_abs_tol,
        },
    }

    if not args.skip_uart:
        ready_line, device_layers = uart_layerdump(args.port, args.text)
        rows, first_diverged = compare_layers(
            host_layers,
            device_layers,
            sample_tol=args.sample_tol,
            max_abs_tol=args.max_abs_tol,
        )
        result.update(
            {
                "ready_line": ready_line,
                "device_layers": device_layers,
                "comparison": rows,
                "first_diverged_layer": first_diverged,
                "root_cause_hypothesis": classify_root_cause(first_diverged, rows),
            }
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
