#!/usr/bin/env python3
"""Mirror firmware word-feature int8 path on host for parity checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from edge_word_encode import word_features_from_char_ids, word_features_int8
from esp32_embed import embed_exponent
from esp32_int8_parity import encode_sentence, load_model


def firmware_mirror_int8(model, char_ids: np.ndarray, feature_exp: int, embed_exp: int) -> np.ndarray:
    """Python mirror of edge_word_encoder.cpp (pad cache + float conv + int8 quant)."""
    from esp32_embed import quantize_embedding_table

    pad_id = 0
    num_filters = model.config.num_filters
    max_len, max_word_len = char_ids.shape
    out = np.zeros((num_filters, max_len), dtype=np.int8)
    scale = float(2**feature_exp)
    embed_scale = float(2**embed_exp)
    quant_table = quantize_embedding_table(model, embed_exp)

    def conv_token_torch(token_row: np.ndarray) -> np.ndarray:
        emb = (
            torch.from_numpy(quant_table[token_row.astype(np.int64)]).float().T.unsqueeze(0)
            * embed_scale
        )
        x = torch.relu(model.word_encoder.conv(emb))
        return x.max(dim=2).values.squeeze(0).detach().cpu().numpy()

    pad_row = np.full(max_word_len, pad_id, dtype=np.int64)
    pad_feat = np.clip(np.round(conv_token_torch(pad_row) / scale), -128, 127).astype(np.int8)

    for t in range(max_len):
        row = char_ids[t]
        if np.all(row == pad_id):
            out[:, t] = pad_feat
        else:
            out[:, t] = np.clip(np.round(conv_token_torch(row) / scale), -128, 127).astype(np.int8)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="John works in Paris")
    args = parser.parse_args()

    deploy = ROOT / "deploy" / "esp32-s3" / "models"
    vocab = json.loads((deploy / "char_vocab.json").read_text(encoding="utf-8"))
    model = load_model(ROOT / "outputs/runs/conll2003_full/best.pt")
    embed_exp = embed_exponent(deploy)
    feature_exp = -5
    for line in (deploy / "edgefs_model.info").read_text(encoding="utf-8").splitlines():
        if "word_features[INT8" in line and "exponents:" in line:
            part = line.split("exponents:")[1].strip()
            feature_exp = int(part[part.index("[") + 1 : part.index("]")].split(",")[0])

    char_ids = encode_sentence(args.text, vocab)
    char_t = torch.from_numpy(char_ids).long().unsqueeze(0)
    host_f = word_features_from_char_ids(model, char_t, embed_exp)[0].detach().cpu().numpy()
    host_i = word_features_int8(model, char_t, embed_exp, feature_exp)[0]
    mirror_i = firmware_mirror_int8(model, char_ids, feature_exp, embed_exp)

    diff = np.abs(host_i.astype(np.int16) - mirror_i.astype(np.int16))
    print(
        json.dumps(
            {
                "text": args.text,
                "max_int8_diff": int(diff.max()),
                "mean_int8_diff": float(diff.mean()),
                "host_t0_int8": host_i[:, 0].tolist()[:8],
                "mirror_t0_int8": mirror_i[:, 0].tolist()[:8],
                "host_t0_float": host_f[:, 0].tolist()[:8],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
