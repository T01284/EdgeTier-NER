#!/usr/bin/env python3
"""Full-data supervised training (local CPU smoke or remote GPU)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from edgefs.config import load_config
from edgefs.data.glove import build_embedding_matrix, ensure_glove
from edgefs.data.prepare import load_processed_corpus
from edgefs.models.bilstm_crf import (
    BiLSTMCRF,
    BiLSTMCRFConfig,
    BiLSTMCharCRF,
    BiLSTMCharCRFConfig,
)
from edgefs.models.char_cnn_crf import CharCNNCRF, CharCNNCRFConfig
from edgefs.models.bert_crf import BertCRF, BertCRFConfig
from edgefs.training.trainer import FullTrainer, TrainConfig
from edgefs.training.checkpoint_loader import load_teacher_from_checkpoint
from edgefs.utils.hf_mirror import use_hf_mirror


def _build_bilstm_model(corpus, model_cfg_raw: dict, embed_cfg: dict | None, model_type: str):
    if corpus.word_vocab is None:
        raise ValueError(f"{model_type} requires word_vocab in processed corpus")
    embed_dim = int(model_cfg_raw.get("embed_dim", 100))
    embed_cfg = embed_cfg or {}
    glove_dir = embed_cfg.get("glove_dir", "data/embeddings")
    glove_path = ensure_glove(glove_dir, dim=embed_dim)
    pretrained = build_embedding_matrix(corpus.word_vocab, glove_path, embed_dim=embed_dim)
    num_tags = len(corpus.tagset.tags)

    if model_type == "bilstm_char_crf":
        model_cfg = BiLSTMCharCRFConfig(
            word_vocab_size=len(corpus.word_vocab),
            char_vocab_size=len(corpus.char_vocab),
            num_tags=num_tags,
            embed_dim=embed_dim,
            char_embed_dim=int(model_cfg_raw.get("char_embed_dim", 64)),
            char_num_filters=int(model_cfg_raw.get("char_num_filters", 128)),
            char_kernel_size=int(model_cfg_raw.get("char_kernel_size", 3)),
            hidden_dim=int(model_cfg_raw.get("hidden_dim", 256)),
            num_layers=int(model_cfg_raw.get("num_layers", 1)),
            dropout=float(model_cfg_raw.get("dropout", 0.5)),
            freeze_embeddings=bool(model_cfg_raw.get("freeze_embeddings", False)),
        )
        return BiLSTMCharCRF(model_cfg, pretrained_embeddings=pretrained), model_type, model_cfg

    model_cfg = BiLSTMCRFConfig(
        vocab_size=len(corpus.word_vocab),
        num_tags=num_tags,
        embed_dim=embed_dim,
        hidden_dim=int(model_cfg_raw.get("hidden_dim", 256)),
        num_layers=int(model_cfg_raw.get("num_layers", 1)),
        dropout=float(model_cfg_raw.get("dropout", 0.5)),
        freeze_embeddings=bool(model_cfg_raw.get("freeze_embeddings", False)),
    )
    return BiLSTMCRF(model_cfg, pretrained_embeddings=pretrained), model_type, model_cfg


def build_model(corpus, model_cfg_raw: dict, embed_cfg: dict | None):
    model_type = model_cfg_raw.get("model", "char_cnn_crf")
    num_tags = len(corpus.tagset.tags)

    if model_type == "bert_crf":
        model_cfg = BertCRFConfig(
            pretrained_name=model_cfg_raw.get("pretrained_name", "bert-base-cased"),
            num_tags=num_tags,
            dropout=float(model_cfg_raw.get("dropout", 0.1)),
            freeze_backbone=bool(model_cfg_raw.get("freeze_backbone", False)),
        )
        return BertCRF(model_cfg), model_type, model_cfg

    if model_type in ("bilstm_crf", "bilstm_char_crf"):
        return _build_bilstm_model(corpus, model_cfg_raw, embed_cfg, model_type)

    model_cfg = CharCNNCRFConfig(
        vocab_size=len(corpus.char_vocab),
        num_tags=num_tags,
        embed_dim=model_cfg_raw.get("embed_dim", 64),
        num_filters=model_cfg_raw.get("num_filters", 128),
        kernel_size=model_cfg_raw.get("kernel_size", 3),
        dilations=tuple(model_cfg_raw.get("dilations", [1, 2, 4])),
        dropout=model_cfg_raw.get("dropout", 0.1),
        use_residual=model_cfg_raw.get("use_residual", True),
        max_word_len=int(model_cfg_raw.get("max_word_len", 20)),
    )
    return CharCNNCRF(model_cfg), "char_cnn_crf", model_cfg


def main() -> None:
    use_hf_mirror()
    parser = argparse.ArgumentParser(description="Train NER model (full supervision)")
    parser.add_argument("--config", type=str, default="configs/train/full_toy.yaml")
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cpu/cuda")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds_cfg = load_config(cfg.get("dataset_config"))
    model_cfg_raw = load_config(cfg.get("model_config"))
    train_raw = cfg.get("train", {})
    embed_cfg = cfg.get("embeddings")

    processed_dir = ds_cfg.get("processed_dir")
    if not Path(processed_dir).exists():
        raise FileNotFoundError(
            f"Processed data not found at {processed_dir}. "
            f"Run: python scripts/prepare_datasets.py --config {cfg.get('dataset_config')}"
        )

    corpus = load_processed_corpus(processed_dir)
    model, model_type, _model_cfg = build_model(corpus, model_cfg_raw, embed_cfg)

    teacher = None
    teacher_model_type = None
    teacher_tagset = None
    distill_cfg = cfg.get("distill") or {}
    teacher_ckpt = distill_cfg.get("teacher_checkpoint")

    device = args.device or train_raw.get("device", "cpu")
    if teacher_ckpt:
        teacher, teacher_model_type, teacher_tagset = load_teacher_from_checkpoint(
            teacher_ckpt, corpus, embed_cfg
        )
        teacher = teacher.to(device)
    max_word_len = int(
        train_raw.get(
            "max_word_len",
            model_cfg_raw.get("max_word_len", 20),
        )
    )
    train_cfg = TrainConfig(
        epochs=int(train_raw.get("epochs", 20)),
        batch_size=int(train_raw.get("batch_size", 32)),
        lr=float(train_raw.get("lr", 1e-3)),
        weight_decay=float(train_raw.get("weight_decay", 1e-4)),
        max_len=int(train_raw.get("max_len", 128)),
        max_word_len=max_word_len,
        device=device,
        output_dir=str(train_raw.get("output_dir", "outputs/runs/default")),
        seed=int(train_raw.get("seed", 42)),
        distill_alpha=float(
            distill_cfg.get("alpha", train_raw.get("distill_alpha", 0.5))
        ),
        distill_temperature=float(
            distill_cfg.get("temperature", train_raw.get("distill_temperature", 2.0))
        ),
        lambda_transition=float(
            distill_cfg.get("lambda_transition", train_raw.get("lambda_transition", 0.1))
        ),
        lr_scheduler=train_raw.get("lr_scheduler"),
        scheduler_patience=int(train_raw.get("scheduler_patience", 3)),
        scheduler_factor=float(train_raw.get("scheduler_factor", 0.5)),
        grad_clip_norm=(
            float(train_raw["grad_clip_norm"])
            if train_raw.get("grad_clip_norm") is not None
            else None
        ),
        early_stop_patience=(
            int(train_raw["early_stop_patience"])
            if train_raw.get("early_stop_patience") is not None
            else None
        ),
        min_epochs=int(train_raw.get("min_epochs", 1)),
        eval_scheme=str(train_raw.get("eval_scheme", "IOB2")),
        log_test_each_epoch=bool(train_raw.get("log_test_each_epoch", False)),
    )

    trainer = FullTrainer(
        corpus,
        train_cfg,
        model=model,
        model_type=model_type,
        teacher=teacher,
        teacher_model_type=teacher_model_type,
        teacher_tagset=teacher_tagset,
        bert_tokenizer_name=(
            model_cfg_raw.get("pretrained_name") if model_type == "bert_crf" else None
        ),
    )
    metrics = trainer.train()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
