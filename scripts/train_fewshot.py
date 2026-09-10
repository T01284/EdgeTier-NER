#!/usr/bin/env python3
"""Few-shot episodic training: Ours (char-CNN) and baselines (ProtoBERT, NNShot, StructShot)."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from edgefs.config import load_config
from edgefs.data.fewshot import FewShotEpisodeSampler
from edgefs.data.prepare import load_processed_corpus
from edgefs.training.fewshot_methods import build_fewshot_trainer
from edgefs.training.trainer import TrainConfig
from edgefs.utils.hf_mirror import use_hf_mirror


def main() -> None:
    use_hf_mirror()
    parser = argparse.ArgumentParser(description="Few-shot NER training")
    parser.add_argument("--config", type=str, default="configs/train/fewshot_conll2003.yaml")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds_cfg = load_config(cfg.get("dataset_config"))
    model_cfg_raw = load_config(cfg.get("model_config"))
    train_raw = cfg.get("train", {})
    fs_raw = cfg.get("fewshot", {})

    processed_dir = ds_cfg.get("processed_dir")
    corpus = load_processed_corpus(processed_dir)
    method = str(fs_raw.get("method", "ours")).lower()
    sampler = FewShotEpisodeSampler(
        corpus.train,
        n_way=int(fs_raw.get("n_way", 5)),
        k_shot=int(fs_raw.get("k_shot", 1)),
        query_size=int(fs_raw.get("query_size", 200)),
        seed=int(train_raw.get("seed", 42)),
    )

    episode_metrics = []
    num_episodes = int(fs_raw.get("num_episodes", 5))
    for ep in range(num_episodes):
        episode = sampler.sample(seed=int(train_raw.get("seed", 42)) + ep)
        ep_corpus = replace(corpus, train=episode.support + episode.query)

        train_cfg = TrainConfig(
            epochs=int(train_raw.get("epochs", 10)),
            batch_size=int(train_raw.get("batch_size", 16)),
            lr=float(train_raw.get("lr", 1e-3)),
            weight_decay=float(train_raw.get("weight_decay", 1e-4)),
            max_len=int(train_raw.get("max_len", 128)),
            max_word_len=int(
                train_raw.get("max_word_len", model_cfg_raw.get("max_word_len", 20))
            ),
            device=args.device or train_raw.get("device", "cpu"),
            output_dir=str(Path(train_raw.get("output_dir", "outputs/runs/fewshot")) / f"ep{ep}"),
            seed=int(train_raw.get("seed", 42)) + ep,
            lambda_proto=float(train_raw.get("lambda_proto", 0.5)),
            lambda_contrastive=float(train_raw.get("lambda_contrastive", 0.1)),
            lambda_struct=float(train_raw.get("lambda_struct", 0.0)),
            proto_temperature=float(train_raw.get("proto_temperature", 10.0)),
            nn_k=int(train_raw.get("nn_k", 1)),
            support_only=bool(train_raw.get("support_only", False)),
            lr_scheduler=train_raw.get("lr_scheduler"),
            scheduler_patience=int(train_raw.get("scheduler_patience", 2)),
            scheduler_factor=float(train_raw.get("scheduler_factor", 0.5)),
            grad_clip_norm=(
                float(train_raw["grad_clip_norm"])
                if train_raw.get("grad_clip_norm") is not None
                else None
            ),
            eval_scheme=str(train_raw.get("eval_scheme", "IOB2")),
        )

        trainer = build_fewshot_trainer(
            method,
            ep_corpus,
            train_cfg,
            model_cfg_raw,
            episode.support,
        )
        metrics = trainer.train()
        metrics["episode"] = ep
        metrics["entity_types"] = episode.entity_types
        metrics["method"] = method
        episode_metrics.append(metrics)

    out_dir = Path(train_raw.get("output_dir", "outputs/runs/fewshot"))
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "fewshot_summary.json"
    summary_path.write_text(json.dumps(episode_metrics, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
