#!/usr/bin/env python3
"""Generate YAML configs for cloud paper-gap batch A–E."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs" / "train"
REMOTE = "/root/autodl-tmp/EdgeFS_NER"

SEEDS = [42, 43, 44]
DISTILL_GRID = [
    # teacher, alpha, temperature, lambda_transition, tag
    ("bilstm", 0.5, 2.0, 0.1, "bilstm_a05_t2_b01"),
    ("bilstm", 0.3, 2.0, 0.1, "bilstm_a03_t2_b01"),
    ("bilstm", 0.7, 2.0, 0.1, "bilstm_a07_t2_b01"),
    ("bilstm", 0.5, 1.0, 0.1, "bilstm_a05_t1_b01"),
    ("bilstm", 0.5, 4.0, 0.1, "bilstm_a05_t4_b01"),
    ("bilstm", 0.5, 2.0, 0.0, "bilstm_a05_t2_b00"),
    ("bert", 0.5, 2.0, 0.1, "bert_a05_t2_b01"),
    ("bert", 0.5, 4.0, 0.1, "bert_a05_t4_b01"),
]
TEACHERS = {
    "bilstm": f"{REMOTE}/outputs/runs/conll2003_bilstm/best.pt",
    "bert": f"{REMOTE}/outputs/runs/conll2003_bert/best.pt",
}


def write_yaml(name: str, body: str) -> Path:
    path = OUT / name
    path.write_text(body.strip() + "\n", encoding="utf-8")
    print(f"wrote {path.name}")
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # --- A: multi-seed wordlevel / distill / full ---
    for seed in SEEDS:
        write_yaml(
            f"ablation_conll2003_wordlevel_s{seed}_autodl.yaml",
            f"""
name: conll2003_ablation_wordlevel_s{seed}
dataset_config: configs/dataset/conll2003_autodl.yaml
model_config: configs/model/bilstm_crf_glove.yaml
embeddings:
  glove_dir: {REMOTE}/data/embeddings
  dim: 100
train:
  epochs: 30
  batch_size: 32
  lr: 0.001
  weight_decay: 0.0001
  max_len: 128
  max_word_len: 20
  device: cuda
  seed: {seed}
  output_dir: {REMOTE}/outputs/runs/conll2003_ablation_wordlevel_s{seed}
  lr_scheduler: plateau
  scheduler_patience: 3
  scheduler_factor: 0.5
""",
        )
        write_yaml(
            f"full_conll2003_distill_s{seed}_autodl.yaml",
            f"""
name: conll2003_distill_s{seed}
dataset_config: configs/dataset/conll2003_autodl.yaml
model_config: configs/model/char_cnn_crf_small.yaml
distill:
  teacher_checkpoint: {TEACHERS['bilstm']}
  alpha: 0.5
  temperature: 2.0
  lambda_transition: 0.1
train:
  epochs: 30
  batch_size: 32
  lr: 0.001
  weight_decay: 0.0001
  max_len: 128
  device: cuda
  seed: {seed}
  output_dir: {REMOTE}/outputs/runs/conll2003_distill_s{seed}
""",
        )
        write_yaml(
            f"full_conll2003_s{seed}_autodl.yaml",
            f"""
name: conll2003_full_s{seed}
dataset_config: configs/dataset/conll2003_autodl.yaml
model_config: configs/model/char_cnn_crf_small.yaml
train:
  epochs: 30
  batch_size: 32
  lr: 0.001
  weight_decay: 0.0001
  max_len: 128
  device: cuda
  seed: {seed}
  output_dir: {REMOTE}/outputs/runs/conll2003_full_s{seed}
""",
        )

    # --- B: distill hyperparameter × teacher grid ---
    for teacher, alpha, temp, beta, tag in DISTILL_GRID:
        write_yaml(
            f"full_conll2003_distill_grid_{tag}_autodl.yaml",
            f"""
name: conll2003_distill_grid_{tag}
dataset_config: configs/dataset/conll2003_autodl.yaml
model_config: configs/model/char_cnn_crf_small.yaml
distill:
  teacher_checkpoint: {TEACHERS[teacher]}
  alpha: {alpha}
  temperature: {temp}
  lambda_transition: {beta}
train:
  epochs: 30
  batch_size: 32
  lr: 0.001
  weight_decay: 0.0001
  max_len: 128
  device: cuda
  seed: 42
  output_dir: {REMOTE}/outputs/runs/conll2003_distill_grid_{tag}
""",
        )

    # --- C: frozen BERT + OntoNotes/CLUENER few-shot baselines ---
    write_yaml(
        "model_bert_crf_base_frozen.yaml",
        """
name: bert_crf_base_frozen
model: bert_crf
pretrained_name: bert-base-cased
dropout: 0.1
freeze_backbone: true
""",
    )
    # Move model yaml under configs/model/
    model_src = OUT / "model_bert_crf_base_frozen.yaml"
    model_dst = ROOT / "configs" / "model" / "bert_crf_base_frozen.yaml"
    model_dst.write_text(model_src.read_text(encoding="utf-8"), encoding="utf-8")
    model_src.unlink()
    print(f"wrote {model_dst.name}")

    bert_fs = """name: {run}
dataset_config: {ds_cfg}
model_config: {model_cfg}
fewshot:
  method: {method}
  n_way: 4
  k_shot: {k}
  num_episodes: 5
  query_size: 200
train:
  epochs: {epochs}
  batch_size: 16
  lr: {lr}
  weight_decay: 0.01
  max_len: 128
  device: cuda
  seed: 42
  lambda_proto: {lp}
  lambda_contrastive: {lc}
  lambda_struct: {ls}
  support_only: {support_only}
  nn_k: 1
  lr_scheduler: plateau
  scheduler_patience: 2
  scheduler_factor: 0.5
  output_dir: {REMOTE}/outputs/runs/{run}
"""
    for k, run_suf, epochs in [(1, "4w1s", 5), (5, "4w5s", 5)]:
        write_yaml(
            f"fewshot_conll2003_frozenbert_{run_suf}_autodl.yaml",
            bert_fs.format(
                run=f"conll2003_fewshot_frozenbert_{run_suf}",
                ds_cfg="configs/dataset/conll2003_autodl.yaml",
                model_cfg="configs/model/bert_crf_base_frozen.yaml",
                method="protobert",
                k=k,
                epochs=epochs,
                lr=0.0001,
                lp=1.0,
                lc=0.0,
                ls=0.0,
                support_only="false",
                REMOTE=REMOTE,
            ),
        )

    for ds, ds_cfg, model_cfg in [
        ("ontonotes", "configs/dataset/ontonotes_autodl.yaml", "configs/model/bert_crf_base.yaml"),
        ("cluener", "configs/dataset/cluener_autodl.yaml", "configs/model/bert_crf_chinese.yaml"),
    ]:
        for method, k, run_suf, epochs, lp, lc, ls, support_only in [
            ("protobert", 1, "protobert_4w1s", 5, 1.0, 0.0, 0.0, "false"),
            ("protobert", 5, "protobert_4w5s", 5, 1.0, 0.0, 0.0, "false"),
            ("nnshot", 1, "nnshot_4w1s", 3, 0.0, 0.0, 0.0, "true"),
            ("nnshot", 5, "nnshot_4w5s", 3, 0.0, 0.0, 0.0, "true"),
            ("structshot", 1, "structshot_4w1s", 5, 0.5, 0.0, 0.2, "false"),
            ("structshot", 5, "structshot_4w5s", 5, 0.5, 0.0, 0.2, "false"),
        ]:
            write_yaml(
                f"fewshot_{ds}_{run_suf}_autodl.yaml",
                bert_fs.format(
                    run=f"{ds}_fewshot_{run_suf}",
                    ds_cfg=ds_cfg,
                    model_cfg=model_cfg,
                    method=method,
                    k=k,
                    epochs=epochs,
                    lr=0.00005,
                    lp=lp,
                    lc=lc,
                    ls=ls,
                    support_only=support_only,
                    REMOTE=REMOTE,
                ),
            )

    # CLUENER Ours few-shot
    for k, suf in [(1, "4w1s"), (5, "4w5s")]:
        write_yaml(
            f"fewshot_cluener_{suf}_autodl.yaml",
            f"""
name: cluener_fewshot_{suf}
dataset_config: configs/dataset/cluener_autodl.yaml
model_config: configs/model/char_cnn_crf_small.yaml
fewshot:
  method: ours
  n_way: 4
  k_shot: {k}
  num_episodes: 5
  query_size: 200
train:
  epochs: 10
  batch_size: 16
  lr: 0.001
  max_len: 128
  device: cuda
  seed: 42
  lambda_proto: 0.5
  lambda_contrastive: 0.1
  output_dir: {REMOTE}/outputs/runs/cluener_fewshot_{suf}
""",
        )

    # --- D: byte-level CLUENER ---
    ds_byte = ROOT / "configs" / "dataset" / "cluener_byte_autodl.yaml"
    ds_byte.write_text(
        f"""
name: cluener_byte
dataset: cluener
encoding: byte
raw_dir: {REMOTE}/data/raw/cluener
processed_dir: {REMOTE}/data/processed/cluener_byte
""".strip()
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {ds_byte.name}")
    write_yaml(
        "full_cluener_byte_autodl.yaml",
        f"""
name: cluener_byte_full
dataset_config: configs/dataset/cluener_byte_autodl.yaml
model_config: configs/model/char_cnn_crf_small.yaml
train:
  epochs: 30
  batch_size: 32
  lr: 0.001
  weight_decay: 0.0001
  max_len: 128
  max_word_len: 24
  device: cuda
  seed: 42
  output_dir: {REMOTE}/outputs/runs/cluener_byte_full
""",
    )

    print("done")


if __name__ == "__main__":
    main()
