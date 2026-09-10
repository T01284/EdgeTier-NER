"""Firmware-aligned INT8 character embedding for ESP32 / PPQ."""

from __future__ import annotations

import numpy as np
import torch
from edgefs.models.char_cnn_crf import CharCNNCRF

DEFAULT_EMBED_EXP = -5


def embed_exponent(deploy_dir) -> int:
    from pathlib import Path

    info = Path(deploy_dir) / "edgefs_model.info"
    if info.exists():
        for line in info.read_text(encoding="utf-8").splitlines():
            if "embedded[INT8" in line and "exponents:" in line:
                part = line.split("exponents:")[1].strip()
                start = part.index("[") + 1
                end = part.index("]", start)
                return int(part[start:end].split(",")[0].strip())
    return DEFAULT_EMBED_EXP


def quantize_embedding_table(
    model: CharCNNCRF,
    exp: int = DEFAULT_EMBED_EXP,
) -> np.ndarray:
    inv = float(1 << -exp)
    weight = model.word_encoder.embedding.weight.detach().cpu().numpy()
    return np.clip(np.round(weight * inv), -128, 127).astype(np.int8)


def build_int8_embed_array(
    char_ids: np.ndarray,
    quant_table: np.ndarray,
    max_len: int = 128,
    max_word_len: int = 20,
) -> np.ndarray:
    """Shape [1, max_len, embed_dim, max_word_len] int8, firmware layout."""
    embed_dim = quant_table.shape[1]
    out = np.zeros((1, max_len, embed_dim, max_word_len), dtype=np.int8)
    for t in range(max_len):
        for j in range(max_word_len):
            out[0, t, :, j] = quant_table[int(char_ids[t, j])]
    return out


def dequantize_embed(emb_int8: np.ndarray, exp: int = DEFAULT_EMBED_EXP) -> np.ndarray:
    return emb_int8.astype(np.float32) * float(2**exp)


def firmware_calib_embed(
    model: CharCNNCRF,
    char_ids: torch.Tensor,
    exp: int = DEFAULT_EMBED_EXP,
) -> torch.Tensor:
    """Float embedded input matching firmware int8 table + dequant (PPQ calib path)."""
    quant = torch.from_numpy(quantize_embedding_table(model, exp))
    b, t, w = char_ids.shape
    flat_idx = char_ids.reshape(b * t, w).long()
    rows = quant[flat_idx]
    emb = rows.reshape(b, t, w, -1).permute(0, 1, 3, 2).contiguous()
    return emb.float() * float(2**exp)
