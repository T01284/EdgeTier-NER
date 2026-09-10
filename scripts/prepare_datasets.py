#!/usr/bin/env python3
"""Prepare datasets into unified CoNLL + char vocab."""

from __future__ import annotations

import argparse
from pathlib import Path

from edgefs.config import load_config
from edgefs.data.prepare import prepare_cluener, prepare_conll2003, prepare_ontonotes, prepare_toy
from edgefs.utils.hf_mirror import use_hf_mirror


def main() -> None:
    use_hf_mirror()
    parser = argparse.ArgumentParser(description="Prepare NER datasets")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset/toy.yaml",
        help="Dataset config YAML",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset = cfg.get("dataset")
    processed_dir = cfg.get("processed_dir")

    if dataset == "toy":
        corpus = prepare_toy(processed_dir)
    elif dataset == "conll2003":
        corpus = prepare_conll2003(
            cfg.get("raw_dir"),
            processed_dir,
            lowercase=bool(cfg.get("lowercase", False)),
        )
    elif dataset == "cluener":
        corpus = prepare_cluener(
            cfg.get("raw_dir"),
            processed_dir,
            encoding=str(cfg.get("encoding", "char")),
        )
    elif dataset == "ontonotes":
        corpus = prepare_ontonotes(cfg.get("raw_dir"), processed_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    print(f"Prepared {dataset} -> {processed_dir}")
    print(
        f"train={len(corpus.train)} dev={len(corpus.dev)} test={len(corpus.test)} "
        f"tags={len(corpus.tagset.tags)} vocab={len(corpus.char_vocab)}"
    )


if __name__ == "__main__":
    main()
