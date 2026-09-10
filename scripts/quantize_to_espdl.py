#!/usr/bin/env python3
"""Quantize Char-CNN emissions model to ESP-DL .espdl via ESP-PPQ."""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from onnx import helper as onnx_helper
from esp_ppq import QuantizationSettingFactory
from esp_ppq.api import espdl_quantize_onnx, espdl_quantize_torch
from esp_ppq.api.interface import load_onnx_graph
from esp_ppq.core import TargetPlatform
from torch.utils.data import DataLoader, TensorDataset

from edgefs.models.char_cnn_crf import CharCNNCRF, CharCNNCRFConfig, ConvBlock

scripts_dir = Path(__file__).resolve().parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))
from esp32_embed import firmware_calib_embed, embed_exponent
from edge_word_encode import word_features_from_char_ids

# ESP-DL on ESP32-S3 does not register Cast; keep the export graph Cast-free.
_FP32_OP_TYPES = frozenset()


def _patch_onnx_dtype_helper() -> None:
    """esp-ppq 1.3.x expects dict-like tensor_dtype_to_np_dtype; onnx>=1.16 uses a function."""
    if callable(onnx_helper.tensor_dtype_to_np_dtype) and not hasattr(
        onnx_helper.tensor_dtype_to_np_dtype, "__getitem__"
    ):
        fn = onnx_helper.tensor_dtype_to_np_dtype

        class _DtypeMap:
            def __getitem__(self, key: int):
                return fn(key)

            def __call__(self, key: int):
                return fn(key)

        onnx_helper.tensor_dtype_to_np_dtype = _DtypeMap()


_patch_onnx_dtype_helper()


def encode_sentence(
    text: str,
    char2id: dict[str, int],
    unk_id: int,
    pad_id: int,
    max_len: int = 128,
    max_word_len: int = 20,
) -> np.ndarray:
    words = text.split()[:max_len]
    arr = np.full((max_len, max_word_len), pad_id, dtype=np.int64)
    for i, word in enumerate(words):
        word_l = word.lower()
        for j, ch in enumerate(word_l[:max_word_len]):
            arr[i, j] = char2id.get(ch, unk_id)
    return arr


def build_calibration_tensors(
    deploy_dir: Path,
    samples_path: Path,
    num_random: int,
    max_len: int,
    max_word_len: int,
    as_float: bool,
) -> torch.Tensor:
    vocab = json.loads((deploy_dir / "char_vocab.json").read_text(encoding="utf-8"))
    char2id = vocab["char2id"]
    unk_id = vocab["unk_id"]
    pad_id = vocab["pad_id"]

    texts: list[str] = []
    if samples_path.exists():
        for item in json.loads(samples_path.read_text(encoding="utf-8")):
            texts.append(item["text"])

    words_pool = [
        "John", "Mary", "Paris", "London", "Google", "Berlin", "works", "visited",
        "opened", "office", "company", "China", "Beijing", "Microsoft", "Apple",
    ]
    rng = random.Random(42)
    for _ in range(num_random):
        n = rng.randint(3, 12)
        texts.append(" ".join(rng.choice(words_pool) for _ in range(n)))

    arrays = np.stack(
        [
            encode_sentence(t, char2id, unk_id, pad_id, max_len, max_word_len)
            for t in texts
        ]
    )
    if as_float:
        return torch.from_numpy(arrays).float()
    return torch.from_numpy(arrays).long()


class EmissionWrapper(nn.Module):
    """Full emissions path for host parity checks."""

    def __init__(self, model: CharCNNCRF) -> None:
        super().__init__()
        self.word_encoder = model.word_encoder
        self.blocks = nn.ModuleList(ConvBlockNTC(block) for block in model.blocks)
        self.dropout = model.dropout
        self.classifier = model.classifier

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.word_encoder(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.classifier(self.dropout(x))


class ConvBlockNTC(nn.Module):
    """ConvBlock on [B, T, C] — keeps transpose inside the block, not on the pooling edge."""

    def __init__(self, block: ConvBlock) -> None:
        super().__init__()
        self.conv = block.conv
        self.bn = block.bn
        self.relu = block.relu
        self.use_residual = block.use_residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.transpose(1, 2)
        y = self.relu(self.bn(self.conv(y)))
        y = y.transpose(1, 2)
        if self.use_residual:
            return y + x
        return y


def densify_dilated_conv1d(conv: nn.Conv1d) -> nn.Conv1d:
    """Expand dilated Conv1d into an equivalent dilation=1 sparse kernel.

    ESP-DL on-device Conv with dilations>1 diverges from PPQ golden for this
    TemporalCNN; densified kernels keep host math identical while avoiding
    dilated paths on chip.
    """
    dilation = conv.dilation[0] if isinstance(conv.dilation, tuple) else int(conv.dilation)
    kernel = conv.kernel_size[0]
    if dilation == 1:
        return conv
    new_k = (kernel - 1) * dilation + 1
    out_ch, in_ch, _ = conv.weight.shape
    new_w = torch.zeros(out_ch, in_ch, new_k, dtype=conv.weight.dtype)
    for i in range(kernel):
        new_w[:, :, i * dilation] = conv.weight.data[:, :, i]
    new_pad = (new_k - 1) // 2
    new_conv = nn.Conv1d(
        in_ch,
        out_ch,
        kernel_size=new_k,
        padding=new_pad,
        dilation=1,
        bias=conv.bias is not None,
    )
    new_conv.weight.data.copy_(new_w)
    if conv.bias is not None:
        new_conv.bias.data.copy_(conv.bias.data)
    return new_conv


def densify_model_blocks(model: CharCNNCRF) -> CharCNNCRF:
    for block in model.blocks:
        block.conv = densify_dilated_conv1d(block.conv)
    return model


class PostPoolEmissionWrapper(nn.Module):
    """ESP-DL graph starts after word conv+max; pooling runs in firmware."""

    def __init__(self, model: CharCNNCRF) -> None:
        super().__init__()
        self.blocks = model.blocks
        self.dropout = model.dropout
        self.classifier = model.classifier

    def forward(self, word_features: torch.Tensor) -> torch.Tensor:
        x = word_features
        for block in self.blocks:
            x = block(x)
        return self.classifier(self.dropout(x.transpose(1, 2)))


class PostEmbedEmissionWrapper(nn.Module):
    """ESP-DL graph starts at word_encoder.conv; embedding runs in firmware."""

    def __init__(self, model: CharCNNCRF) -> None:
        super().__init__()
        enc = model.word_encoder
        self.conv = enc.conv
        self.enc_dropout = enc.dropout
        self.blocks = nn.ModuleList(ConvBlockNTC(block) for block in model.blocks)
        self.dropout = model.dropout
        self.classifier = model.classifier

    def forward(self, embedded: torch.Tensor) -> torch.Tensor:
        b, t, c, w = embedded.shape
        x = embedded.reshape(b * t, c, w)
        x = torch.relu(self.conv(x))
        x = self.enc_dropout(x.max(dim=2).values)
        x = x.reshape(b, t, -1)
        for block in self.blocks:
            x = block(x)
        return self.classifier(self.dropout(x))


def load_char_cnn(checkpoint: Path) -> CharCNNCRF:
    ckpt = torch.load(checkpoint, map_location="cpu")
    model_cfg = CharCNNCRFConfig(**ckpt["model_config"])
    model = CharCNNCRF(model_cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def embedded_from_char_ids(model: CharCNNCRF, input_ids: torch.Tensor) -> torch.Tensor:
    b, t, w = input_ids.shape
    flat = input_ids.reshape(b * t, w)
    x = model.word_encoder.embedding(flat).transpose(1, 2)
    c = x.shape[1]
    return x.reshape(b, t, c, w)


def load_emission_model(checkpoint: Path) -> EmissionWrapper:
    return EmissionWrapper(load_char_cnn(checkpoint))


def build_fp32_dispatching(model: nn.Module, input_shape: list[int]) -> dict[str, TargetPlatform]:
    """Pre-export ONNX and mark fragile ops as FP32 for PPQ."""
    model = model.eval()
    dummy = torch.zeros(input_shape, dtype=torch.float32)
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        onnx_path = tmp.name
    try:
        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            input_names=["input_ids"],
            output_names=["emissions"],
            opset_version=13,
            dynamic_axes=None,
            dynamo=False,
        )
        graph = load_onnx_graph(onnx_import_file=onnx_path)
    finally:
        Path(onnx_path).unlink(missing_ok=True)

    override: dict[str, TargetPlatform] = {}
    for op in graph.operations.values():
        if op.type in _FP32_OP_TYPES:
            override[op.name] = TargetPlatform.FP32
    return override


def build_word_feature_calibration_tensors(
    deploy_dir: Path,
    samples_path: Path,
    model: CharCNNCRF,
    num_random: int,
    max_len: int,
    max_word_len: int,
) -> torch.Tensor:
    char_ids = build_calibration_tensors(
        deploy_dir,
        samples_path,
        num_random=num_random,
        max_len=max_len,
        max_word_len=max_word_len,
        as_float=False,
    )
    embed_exp = embed_exponent(deploy_dir)
    features: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(char_ids.shape[0]):
            feats = word_features_from_char_ids(model, char_ids[i : i + 1], embed_exp)
            features.append(feats)
    return torch.cat(features, dim=0)


def build_embedded_calibration_tensors(
    deploy_dir: Path,
    samples_path: Path,
    model: CharCNNCRF,
    num_random: int,
    max_len: int,
    max_word_len: int,
) -> torch.Tensor:
    char_ids = build_calibration_tensors(
        deploy_dir,
        samples_path,
        num_random=num_random,
        max_len=max_len,
        max_word_len=max_word_len,
        as_float=False,
    )
    embedded: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(char_ids.shape[0]):
            emb = firmware_calib_embed(model, char_ids[i : i + 1])
            embedded.append(emb)
    return torch.cat(embedded, dim=0)


def export_float_onnx(
    model: nn.Module,
    input_shape: list[int],
    onnx_path: Path,
    input_name: str = "embedded",
) -> None:
    model = model.eval()
    dummy = torch.zeros(input_shape, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=[input_name],
        output_names=["emissions"],
        opset_version=13,
        dynamic_axes=None,
        dynamo=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize to ESP-DL espdl")
    parser.add_argument(
        "--checkpoint",
        default="outputs/runs/conll2003_full/best.pt",
    )
    parser.add_argument(
        "--onnx",
        default="outputs/export/conll2003_full/deploy/model_emissions_fixed.onnx",
    )
    parser.add_argument(
        "--deploy-dir",
        default="deploy/esp32-s3/models",
    )
    parser.add_argument(
        "--output",
        default="deploy/esp32-s3/models/edgefs_model.espdl",
    )
    parser.add_argument("--bits", type=int, default=8, choices=[8, 16])
    parser.add_argument("--calib-samples", type=int, default=64)
    parser.add_argument("--calib-steps", type=int, default=32)
    parser.add_argument(
        "--calib-algorithm",
        default="minmax",
        choices=("minmax", "percentile", "kl"),
    )
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--max-word-len", type=int, default=20)
    parser.add_argument(
        "--input-mode",
        choices=("post-pool", "post-embed"),
        default="post-pool",
        help="post-pool: firmware word conv+max, espdl starts at blocks (default)",
    )
    parser.add_argument(
        "--densify-dilation",
        action="store_true",
        help="expand dilated TemporalCNN kernels to dilation=1 before export",
    )
    parser.add_argument(
        "--backend",
        choices=("torch", "onnx"),
        default="torch",
    )
    args = parser.parse_args()

    deploy_dir = Path(args.deploy_dir)
    samples_path = Path("deploy/shared/sample_inputs.json")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((deploy_dir / "deploy_manifest.json").read_text(encoding="utf-8"))
    embed_dim = int(manifest["model_config"]["embed_dim"])
    num_filters = int(manifest["model_config"]["num_filters"])

    model = load_char_cnn(Path(args.checkpoint))
    if args.densify_dilation:
        densify_model_blocks(model)
        print("Densified dilated conv kernels to dilation=1")
    if args.input_mode == "post-pool":
        calib = build_word_feature_calibration_tensors(
            deploy_dir,
            samples_path,
            model,
            num_random=max(0, args.calib_samples - 3),
            max_len=args.max_len,
            max_word_len=args.max_word_len,
        )
        input_shape = [1, num_filters, args.max_len]
        input_name = "word_features"
    else:
        calib = build_embedded_calibration_tensors(
            deploy_dir,
            samples_path,
            model,
            num_random=max(0, args.calib_samples - 3),
            max_len=args.max_len,
            max_word_len=args.max_word_len,
        )
        input_shape = [1, args.max_len, embed_dim, args.max_word_len]
        input_name = "embedded"
    dataset = TensorDataset(calib)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    test_input = calib[0:1].contiguous()

    def collate_fn(batch: list[tuple[torch.Tensor]]) -> torch.Tensor:
        return batch[0]

    setting = QuantizationSettingFactory.espdl_setting()
    setting.quantize_activation_setting.calib_algorithm = args.calib_algorithm
    print(
        f"Quantizing -> {output_path} ({args.bits}-bit, {args.input_mode}, esp32s3)"
    )

    if args.backend == "torch":
        full_model = model
        if args.input_mode == "post-pool":
            export_model = PostPoolEmissionWrapper(full_model)
        else:
            export_model = PostEmbedEmissionWrapper(full_model)
        dispatching = build_fp32_dispatching(export_model, input_shape)
        print(f"FP32 dispatch override for {len(dispatching)} ops")
        onnx_path = output_path.with_suffix(".onnx")
        export_float_onnx(export_model, input_shape, onnx_path, input_name=input_name)
        graph = espdl_quantize_onnx(
            onnx_import_file=str(onnx_path),
            espdl_export_file=str(output_path),
            calib_dataloader=dataloader,
            calib_steps=args.calib_steps,
            input_shape=input_shape,
            inputs=[test_input],
            target="esp32s3",
            num_of_bits=args.bits,
            collate_fn=collate_fn,
            dispatching_override=dispatching,
            setting=setting,
            device="cpu",
            error_report=True,
            skip_export=False,
            export_test_values=True,
            verbose=1,
        )
    else:
        graph = espdl_quantize_onnx(
            onnx_import_file=args.onnx,
            espdl_export_file=str(output_path),
            calib_dataloader=dataloader,
            calib_steps=args.calib_steps,
            input_shape=input_shape,
            inputs=[test_input],
            target="esp32s3",
            num_of_bits=args.bits,
            collate_fn=collate_fn,
            setting=setting,
            device="cpu",
            error_report=True,
            skip_export=False,
            export_test_values=True,
            verbose=1,
        )

    cast_ops = [name for name, op in graph.operations.items() if op.type == "Cast"]
    gather_ops = [name for name, op in graph.operations.items() if op.type == "Gather"]
    if cast_ops:
        raise RuntimeError(f"espdl graph contains unsupported Cast ops: {cast_ops[:8]}")
    if gather_ops:
        raise RuntimeError(f"espdl graph contains unsupported Gather ops: {gather_ops[:8]}")

    size_kb = output_path.stat().st_size / 1024
    print(f"Done: {output_path} ({size_kb:.1f} KB), graph nodes={len(graph.operations)}")

    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from write_esp32_artifact_manifest import write_manifest

    manifest_path = write_manifest(
        deploy_dir,
        checkpoint=args.checkpoint,
        quantize={
            "backend": args.backend,
            "input_mode": args.input_mode,
            "bits": args.bits,
            "calib_algorithm": args.calib_algorithm,
            "calib_samples": args.calib_samples,
            "calib_steps": args.calib_steps,
            "max_len": args.max_len,
            "target": "esp32s3",
        },
    )
    print(f"Wrote artifact manifest {manifest_path}")


if __name__ == "__main__":
    main()
