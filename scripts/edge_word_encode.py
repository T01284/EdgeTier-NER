"""Firmware-aligned word conv + max-pool producing [B, C, T] word features."""

from __future__ import annotations

import numpy as np
import torch
from edgefs.models.char_cnn_crf import CharCNNCRF

from esp32_embed import firmware_calib_embed


def word_features_from_char_ids(
    model: CharCNNCRF,
    char_ids: torch.Tensor,
    embed_exp: int = -5,
) -> torch.Tensor:
    """Return float word features [B, C, T] matching firmware layout."""
    emb = firmware_calib_embed(model, char_ids, embed_exp)
    b, t, c, w = emb.shape
    flat = emb.reshape(b * t, c, w)
    x = torch.relu(model.word_encoder.conv(flat))
    x = x.max(dim=2).values.reshape(b, t, -1).transpose(1, 2).contiguous()
    return x


def word_features_int8(
    model: CharCNNCRF,
    char_ids: torch.Tensor,
    embed_exp: int = -5,
    feature_exp: int = -5,
) -> np.ndarray:
    scale = float(2**feature_exp)
    feats = word_features_from_char_ids(model, char_ids, embed_exp)
    return np.clip(np.round(feats.detach().cpu().numpy() / scale), -128, 127).astype(np.int8)
