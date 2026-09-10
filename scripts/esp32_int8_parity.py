#!/usr/bin/env python3
"""Host parity: firmware INT8 embedded path vs torch/ONNX."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "deploy" / "pi3b"))

from edgefs.data.conll import read_conll
from edgefs.models.char_cnn_crf import CharCNNCRF, CharCNNCRFConfig
from edgefs_pi.crf_decode import viterbi_decode
from quantize_to_espdl import PostEmbedEmissionWrapper
def load_model(checkpoint: Path) -> CharCNNCRF:
    ckpt = torch.load(checkpoint, map_location="cpu")
    model = CharCNNCRF(CharCNNCRFConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def encode_sentence(text: str, vocab: dict, max_len: int = 128, max_word_len: int = 20) -> np.ndarray:
    char2id = vocab["char2id"]
    unk_id = vocab["unk_id"]
    pad_id = vocab["pad_id"]
    words = text.split()[:max_len]
    arr = np.full((max_len, max_word_len), pad_id, dtype=np.int64)
    for i, word in enumerate(words):
        for j, ch in enumerate(word.lower()[:max_word_len]):
            arr[i, j] = char2id.get(ch, unk_id)
    return arr


def firmware_build_embedded(
    char_ids: np.ndarray,
    quant_weight: np.ndarray,
    seq_len: int,
    fill_tail: str,
) -> np.ndarray:
    """Simulate inference_engine.cpp build_embedded_input."""
    max_len, max_word_len, embed_dim = char_ids.shape[0], char_ids.shape[1], quant_weight.shape[1]
    out = np.zeros((max_len, embed_dim, max_word_len), dtype=np.int8)
    tokens = min(seq_len, max_len)
    for t in range(tokens):
        for w in range(max_word_len):
            cid = int(char_ids[t, w])
            out[t, :, w] = quant_weight[cid]
    if fill_tail == "pad":
        pad_id = int(char_ids[0, 0])  # wrong - use vocab pad
        for t in range(tokens, max_len):
            for w in range(max_word_len):
                out[t, :, w] = quant_weight[pad_id]
    return out


def main() -> None:
    deploy = ROOT / "deploy" / "esp32-s3" / "models"
    vocab = json.loads((deploy / "char_vocab.json").read_text(encoding="utf-8"))
    tags = json.loads((deploy / "tags.json").read_text(encoding="utf-8"))["tags"]
    trans = np.load(deploy / "crf_transitions.npy")
    exp = -5
    inv = float(1 << -exp)

    ckpt = ROOT / "outputs/runs/conll2003_full/best.pt"
    model = load_model(ckpt)
    weight = model.word_encoder.embedding.weight.detach().numpy()
    quant = np.clip(np.round(weight * inv), -128, 127).astype(np.int8)

    sent = read_conll("data/processed_remote/conll2003/test.conll")[0]
    text = " ".join(sent.tokens)
    char_ids = encode_sentence(text, vocab)
    seq_len = len(sent.tokens)

    sess = ort.InferenceSession(str(deploy / "edgefs_model.onnx"), providers=["CPUExecutionProvider"])

    def run_path(name: str, embedded_int8: np.ndarray) -> list[str]:
        emb_f = embedded_int8.astype(np.float32) * (2**exp)
        em = sess.run(None, {"embedded": emb_f[np.newaxis]})[0][0][:seq_len]
        path = viterbi_decode(em, trans)
        pred = [tags[i] for i in path]
        print(f"{name}: {pred[:12]}")
        return pred

    # Correct full-length pad fill (host export path)
    full_pad = firmware_build_embedded(char_ids, quant, seq_len=128, fill_tail="none")
    pad_id = vocab["pad_id"]
    for t in range(seq_len, 128):
        full_pad[t, :, :] = quant[pad_id][:, None]
    for t in range(seq_len):
        for w in range(20):
            full_pad[t, :, w] = quant[int(char_ids[t, w])]

    buggy = firmware_build_embedded(char_ids, quant, seq_len=seq_len, fill_tail="none")

    print("gold:", sent.tags[:12])
    run_path("full_128_pad", full_pad)
    run_path("buggy_tail_zero", buggy)

    with torch.no_grad():
        emb = torch.from_numpy(full_pad.astype(np.float32) * (2**exp)).unsqueeze(0)
        torch_em = PostEmbedEmissionWrapper(model)(emb).numpy()[0][:seq_len]
    path = viterbi_decode(torch_em, trans)
    print("torch_post_embed:", [tags[i] for i in path[:12]])


if __name__ == "__main__":
    main()
