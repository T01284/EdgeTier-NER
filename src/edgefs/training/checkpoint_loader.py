#!/usr/bin/env python3
"""Load a teacher checkpoint for distillation."""

from __future__ import annotations

from pathlib import Path

import torch

from edgefs.data.glove import build_embedding_matrix, ensure_glove
from edgefs.models.bert_crf import BertCRF, BertCRFConfig
from edgefs.models.bilstm_crf import (
    BiLSTMCRF,
    BiLSTMCRFConfig,
    BiLSTMCharCRF,
    BiLSTMCharCRFConfig,
)
from edgefs.models.char_cnn_crf import CharCNNCRF, CharCNNCRFConfig
from edgefs.data.prepare import NERCorpus


def load_teacher_from_checkpoint(
    checkpoint: str | Path,
    corpus: NERCorpus,
    embed_cfg: dict | None = None,
) -> tuple[torch.nn.Module, str, list[str] | None]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    model_type = ckpt.get("model_type", "char_cnn_crf")
    model_cfg = ckpt["model_config"]
    teacher_tagset = ckpt.get("tagset")
    num_tags = len(corpus.tagset.tags)

    if model_type == "bert_crf":
        config = BertCRFConfig(**model_cfg)
        model = BertCRF(config)
    elif model_type == "bilstm_char_crf":
        config = BiLSTMCharCRFConfig(**model_cfg)
        embed_cfg = embed_cfg or {}
        glove_path = ensure_glove(embed_cfg.get("glove_dir", "data/embeddings"), dim=config.embed_dim)
        pretrained = build_embedding_matrix(corpus.word_vocab, glove_path, embed_dim=config.embed_dim)
        model = BiLSTMCharCRF(config, pretrained_embeddings=pretrained)
    elif model_type == "bilstm_crf":
        config = BiLSTMCRFConfig(**model_cfg)
        embed_cfg = embed_cfg or {}
        glove_path = ensure_glove(embed_cfg.get("glove_dir", "data/embeddings"), dim=config.embed_dim)
        pretrained = build_embedding_matrix(corpus.word_vocab, glove_path, embed_dim=config.embed_dim)
        model = BiLSTMCRF(config, pretrained_embeddings=pretrained)
    else:
        model_cfg.setdefault("num_tags", num_tags)
        config = CharCNNCRFConfig(**model_cfg)
        model = CharCNNCRF(config)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, model_type, teacher_tagset
