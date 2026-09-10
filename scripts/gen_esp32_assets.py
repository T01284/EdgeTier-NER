#!/usr/bin/env python3
"""Generate C headers for ESP32 deploy assets (char table, CRF transitions)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEPLOY = ROOT / "deploy" / "esp32-s3" / "models"
DEFAULT_OUT = ROOT / "deploy" / "esp32-s3" / "include"

sys.path.insert(0, str(ROOT / "scripts"))
from write_esp32_artifact_manifest import write_manifest


def write_char_table(vocab_path: Path, out_path: Path) -> None:
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    char2id = vocab["char2id"]
    pad_id = int(vocab["pad_id"])
    unk_id = int(vocab["unk_id"])

    table = [unk_id] * 128
    for ch, idx in char2id.items():
        if len(ch) == 1:
            table[ord(ch)] = int(idx)
    for c in range(ord("A"), ord("Z") + 1):
        lower = c - ord("A") + ord("a")
        if table[lower] != unk_id:
            table[c] = table[lower]

    lines = [
        "#pragma once",
        "",
        f"#define EDGE_CHAR_PAD_ID {pad_id}",
        f"#define EDGE_CHAR_UNK_ID {unk_id}",
        f"#define EDGE_CHAR_TABLE_SIZE {len(table)}",
        "",
        "static const unsigned char kEdgeCharTable[EDGE_CHAR_TABLE_SIZE] = {",
        ", ".join(str(v) for v in table),
        "};",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_transitions(npy_path: Path, out_path: Path, num_tags: int) -> None:
    arr = np.load(npy_path).astype(np.float32)
    flat = arr.reshape(-1)
    if flat.size != num_tags * num_tags:
        raise ValueError(f"unexpected transitions shape {arr.shape}, expected ({num_tags}, {num_tags})")

    lines = [
        "#pragma once",
        "",
        f"#define EDGE_NUM_TAGS {num_tags}",
        "",
        "static const float kEdgeCrfTransitions[EDGE_NUM_TAGS * EDGE_NUM_TAGS] = {",
        ", ".join(f"{v:.8f}f" for v in flat.tolist()),
        "};",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_model_array(espdl_path: Path, out_path: Path) -> None:
    data = espdl_path.read_bytes()
    pad = (-len(data)) % 16
    if pad:
        data = data + b"\x00" * pad
    chunks = ", ".join(f"0x{b:02x}" for b in data)
    lines = [
        "#pragma once",
        "",
        f"#define EDGE_ESPDL_MODEL_SIZE {len(data)}",
        "",
        "static const unsigned char kEdgeEspdlModel[EDGE_ESPDL_MODEL_SIZE]",
        "    __attribute__((aligned(16))) = {",
        chunks,
        "};",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _read_embed_input_exponent(deploy_dir: Path) -> int:
    return _read_tensor_exponent(deploy_dir / "edgefs_model.info", "embedded[INT8", -5)


def _read_tensor_exponent(info_path: Path, tensor_marker: str, default: int) -> int:
    if not info_path.exists():
        return default
    for line in info_path.read_text(encoding="utf-8").splitlines():
        if tensor_marker in line and "exponents:" in line:
            part = line.split("exponents:")[1].strip()
            start = part.index("[") + 1
            end = part.index("]", start)
            return int(part[start:end].split(",")[0].strip())
    return default


def _fold_conv_bn(
    weight: np.ndarray,
    bias: np.ndarray | None,
    bn_weight: np.ndarray,
    bn_bias: np.ndarray,
    bn_mean: np.ndarray,
    bn_var: np.ndarray,
    bn_eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse Conv1d + BatchNorm1d into a single biased convolution."""
    if bias is None:
        bias = np.zeros((weight.shape[0],), dtype=np.float32)
    scale = bn_weight / np.sqrt(bn_var + bn_eps)
    w_f = weight * scale.reshape(-1, 1, 1)
    b_f = (bias - bn_mean) * scale + bn_bias
    return w_f.astype(np.float32), b_f.astype(np.float32)


def write_blocks_weights(checkpoint: Path, out_path: Path, num_filters: int, num_tags: int) -> None:
    """Export BN-folded TemporalCNN blocks + classifier for firmware FP32 path."""
    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt["model_state"]
    model_cfg = ckpt["model_config"]
    dilations = tuple(int(d) for d in model_cfg.get("dilations", [1, 2, 4]))
    kernel = int(model_cfg.get("kernel_size", 3))
    num_blocks = len(dilations)

    weight_chunks: list[str] = []
    bias_chunks: list[str] = []
    for i, dil in enumerate(dilations):
        w = state[f"blocks.{i}.conv.weight"].detach().cpu().numpy().astype(np.float32)
        b = state[f"blocks.{i}.conv.bias"].detach().cpu().numpy().astype(np.float32)
        w_f, b_f = _fold_conv_bn(
            w,
            b,
            state[f"blocks.{i}.bn.weight"].detach().cpu().numpy().astype(np.float32),
            state[f"blocks.{i}.bn.bias"].detach().cpu().numpy().astype(np.float32),
            state[f"blocks.{i}.bn.running_mean"].detach().cpu().numpy().astype(np.float32),
            state[f"blocks.{i}.bn.running_var"].detach().cpu().numpy().astype(np.float32),
            1e-5,
        )
        if w_f.shape != (num_filters, num_filters, kernel):
            raise ValueError(f"block {i} weight shape {w_f.shape}")
        weight_chunks.append(", ".join(f"{v:.8f}f" for v in w_f.reshape(-1).tolist()))
        bias_chunks.append(", ".join(f"{v:.8f}f" for v in b_f.tolist()))

    cls_w = state["classifier.weight"].detach().cpu().numpy().astype(np.float32)
    cls_b = state["classifier.bias"].detach().cpu().numpy().astype(np.float32)
    if cls_w.shape != (num_tags, num_filters):
        raise ValueError(f"classifier weight shape {cls_w.shape}")

    dil_list = ", ".join(str(d) for d in dilations)
    lines = [
        "#pragma once",
        "",
        "#include <stdint.h>",
        "",
        f"#define EDGE_NUM_BLOCKS {num_blocks}",
        f"#define EDGE_BLOCK_KERNEL {kernel}",
        f"#define EDGE_BLOCKS_NUM_FILTERS {num_filters}",
        f"#define EDGE_BLOCKS_NUM_TAGS {num_tags}",
        f"static const int kEdgeBlockDilations[EDGE_NUM_BLOCKS] = {{{dil_list}}};",
        "",
        "static const float kEdgeBlockWeight"
        "[EDGE_NUM_BLOCKS * EDGE_BLOCKS_NUM_FILTERS * EDGE_BLOCKS_NUM_FILTERS * EDGE_BLOCK_KERNEL] = {",
        ",\n".join(weight_chunks),
        "};",
        "",
        "static const float kEdgeBlockBias[EDGE_NUM_BLOCKS * EDGE_BLOCKS_NUM_FILTERS] = {",
        ",\n".join(bias_chunks),
        "};",
        "",
        "static const float kEdgeClassifierWeight[EDGE_BLOCKS_NUM_TAGS * EDGE_BLOCKS_NUM_FILTERS] = {",
        ", ".join(f"{v:.8f}f" for v in cls_w.reshape(-1).tolist()),
        "};",
        "",
        "static const float kEdgeClassifierBias[EDGE_BLOCKS_NUM_TAGS] = {",
        ", ".join(f"{v:.8f}f" for v in cls_b.tolist()),
        "};",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_word_conv(
    checkpoint: Path,
    out_path: Path,
    num_filters: int,
    embed_dim: int,
    feature_exp: int,
) -> None:
    ckpt = torch.load(checkpoint, map_location="cpu")
    weight = (
        ckpt["model_state"]["word_encoder.conv.weight"].detach().cpu().numpy().astype(np.float32)
    )
    bias = ckpt["model_state"]["word_encoder.conv.bias"].detach().cpu().numpy().astype(np.float32)
    if weight.shape != (num_filters, embed_dim, 3):
        raise ValueError(f"unexpected conv weight shape {weight.shape}")

    w_flat = weight.reshape(-1)
    lines = [
        "#pragma once",
        "",
        "#include \"edge_char_embed.h\"",
        "",
        f"#define EDGE_NUM_FILTERS {num_filters}",
        f"#define EDGE_WORD_CONV_KERNEL 3",
        f"#define EDGE_WORD_FEATURE_EXP {feature_exp}",
        "",
        "static const float kEdgeWordConvWeight[EDGE_NUM_FILTERS * EDGE_CHAR_EMBED_DIM * EDGE_WORD_CONV_KERNEL] = {",
        ", ".join(f"{v:.8f}f" for v in w_flat.tolist()),
        "};",
        "",
        "static const float kEdgeWordConvBias[EDGE_NUM_FILTERS] = {",
        ", ".join(f"{v:.8f}f" for v in bias.tolist()),
        "};",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_embed_table(
    checkpoint: Path,
    out_path: Path,
    vocab_size: int,
    embed_dim: int,
    input_exponent: int,
) -> None:
    ckpt = torch.load(checkpoint, map_location="cpu")
    weight = ckpt["model_state"]["word_encoder.embedding.weight"].detach().cpu().numpy()
    if weight.shape != (vocab_size, embed_dim):
        raise ValueError(f"unexpected embedding shape {weight.shape}")

    inv_scale = float(1 << -input_exponent) if input_exponent < 0 else float(1.0 / (1 << input_exponent))
    quant = np.clip(np.round(weight * inv_scale), -128, 127).astype(np.int8)
    flat = quant.reshape(-1)
    lines = [
        "#pragma once",
        "",
        "#include <stdint.h>",
        "",
        f"#define EDGE_CHAR_VOCAB_SIZE {vocab_size}",
        f"#define EDGE_CHAR_EMBED_DIM {embed_dim}",
        f"#define EDGE_CHAR_EMBED_INPUT_EXP {input_exponent}",
        "",
        "static const int8_t kEdgeCharEmbedInt8[EDGE_CHAR_VOCAB_SIZE * EDGE_CHAR_EMBED_DIM] = {",
        ", ".join(str(int(v)) for v in flat.tolist()),
        "};",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ESP32 deploy headers")
    parser.add_argument("--deploy-dir", default=str(DEFAULT_DEPLOY))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--checkpoint", default="outputs/runs/conll2003_full/best.pt")
    args = parser.parse_args()

    deploy = Path(args.deploy_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((deploy / "deploy_manifest.json").read_text(encoding="utf-8"))
    num_tags = int(manifest["num_tags"])
    model_cfg = manifest["model_config"]

    write_char_table(deploy / "char_vocab.json", out / "edge_char_table.h")
    write_transitions(deploy / "crf_transitions.npy", out / "edge_crf_transitions.h", num_tags)
    write_embed_table(
        ROOT / args.checkpoint,
        out / "edge_char_embed.h",
        int(model_cfg["vocab_size"]),
        int(model_cfg["embed_dim"]),
        _read_embed_input_exponent(deploy),
    )
    feature_exp = _read_tensor_exponent(
        deploy / "edgefs_model.info",
        "word_features[INT8",
        -5,
    )
    write_word_conv(
        ROOT / args.checkpoint,
        out / "edge_word_conv.h",
        int(model_cfg["num_filters"]),
        int(model_cfg["embed_dim"]),
        feature_exp,
    )
    write_blocks_weights(
        ROOT / args.checkpoint,
        out / "edge_blocks_weights.h",
        int(model_cfg["num_filters"]),
        num_tags,
    )
    espdl = deploy / "edgefs_model.espdl"
    if espdl.exists():
        write_model_array(espdl, out / "edge_model_data.h")
        manifest_path = write_manifest(
            deploy,
            checkpoint=args.checkpoint,
        )
        print(f"Wrote artifact manifest {manifest_path}")
    print(f"Generated headers in {out}")


if __name__ == "__main__":
    main()
