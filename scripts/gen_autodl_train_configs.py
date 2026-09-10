#!/usr/bin/env python3
"""Generate autodl train configs for multi-dataset GPU experiments."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs" / "train"
REMOTE = "/root/autodl-tmp/EdgeFS_NER"

DATASETS = {
    "conll2003": "configs/dataset/conll2003_autodl.yaml",
    "ontonotes": "configs/dataset/ontonotes_autodl.yaml",
    "cluener": "configs/dataset/cluener_autodl.yaml",
}

OURS = "configs/model/char_cnn_crf_small.yaml"
BILSTM = "configs/model/bilstm_crf_glove.yaml"
BILSTM_V3 = "configs/model/bilstm_char_crf_glove.yaml"
BERT_MODELS = {
    "bert": "configs/model/bert_crf_base.yaml",
    "distilbert": "configs/model/bert_crf_distil.yaml",
    "tinybert": "configs/model/bert_crf_tiny.yaml",
}
CLUENER_BERT_MODELS = {
    "bert": "configs/model/bert_crf_chinese.yaml",
    "distilbert": "configs/model/bert_crf_chinese_distil.yaml",
    "tinybert": "configs/model/bert_crf_chinese_tiny.yaml",
}


def write_yaml(name: str, body: str) -> None:
    path = OUT / name
    path.write_text(body.strip() + "\n", encoding="utf-8")
    print(f"wrote {path.name}")


def full_body(ds: str, model_cfg: str, run: str, epochs: int = 30, extra: str = "") -> str:
    return f"""name: {run}
dataset_config: {DATASETS[ds]}
model_config: {model_cfg}
{extra}train:
  epochs: {epochs}
  batch_size: 32
  lr: 0.001
  weight_decay: 0.0001
  max_len: 128
  device: cuda
  seed: 42
  output_dir: {REMOTE}/outputs/runs/{run}
"""


def bilstm_body(ds: str, run: str, model_cfg: str = BILSTM, epochs: int = 50) -> str:
    return f"""name: {run}
dataset_config: {DATASETS[ds]}
model_config: {model_cfg}
embeddings:
  glove_dir: {REMOTE}/data/embeddings
  dim: 100
train:
  epochs: {epochs}
  batch_size: 32
  lr: 0.001
  weight_decay: 0.0001
  max_len: 128
  max_word_len: 20
  device: cuda
  seed: 42
  output_dir: {REMOTE}/outputs/runs/{run}
  lr_scheduler: plateau
  scheduler_patience: 3
  scheduler_factor: 0.5
"""


def bert_body(ds: str, model_key: str, run: str, epochs: int = 5) -> str:
    models = CLUENER_BERT_MODELS if ds == "cluener" else BERT_MODELS
    return f"""name: {run}
dataset_config: {DATASETS[ds]}
model_config: {models[model_key]}
train:
  epochs: {epochs}
  batch_size: 16
  lr: 0.00005
  weight_decay: 0.01
  max_len: 128
  device: cuda
  seed: 42
  output_dir: {REMOTE}/outputs/runs/{run}
  lr_scheduler: plateau
  scheduler_patience: 2
  scheduler_factor: 0.5
"""


def main() -> None:
    for ds in DATASETS:
        write_yaml(f"full_{ds}_autodl.yaml", full_body(ds, OURS, f"{ds}_full"))
        write_yaml(f"full_{ds}_bilstm_autodl.yaml", bilstm_body(ds, f"{ds}_bilstm"))
        for key in BERT_MODELS:
            write_yaml(f"full_{ds}_{key}_autodl.yaml", bert_body(ds, key, f"{ds}_{key}"))

    write_yaml(
        "full_conll2003_bilstm_v3_autodl.yaml",
        bilstm_body("conll2003", "conll2003_bilstm_v3", BILSTM_V3, epochs=50),
    )
    write_yaml(
        "full_conll2003_distill_autodl.yaml",
        full_body(
            "conll2003",
            OURS,
            "conll2003_distill",
            extra=(
                f"distill:\n"
                f"  teacher_checkpoint: {REMOTE}/outputs/runs/conll2003_bilstm/best.pt\n"
            ),
        ),
    )
    write_yaml(
        "ablation_conll2003_wordlevel_autodl.yaml",
        bilstm_body("conll2003", "conll2003_ablation_wordlevel", BILSTM, epochs=30),
    )

    fewshot_tpl = """name: {run}
dataset_config: configs/dataset/conll2003_autodl.yaml
model_config: configs/model/char_cnn_crf_small.yaml
fewshot:
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
  lambda_proto: {lp}
  lambda_contrastive: {lc}
  output_dir: {REMOTE}/outputs/runs/{run}
"""
    write_yaml(
        "fewshot_conll2003_proto_only_autodl.yaml",
        fewshot_tpl.format(run="conll2003_fewshot_proto_only", k=1, lp=1.0, lc=0.0, REMOTE=REMOTE),
    )
    write_yaml(
        "fewshot_conll2003_contrastive_only_autodl.yaml",
        fewshot_tpl.format(run="conll2003_fewshot_contrastive_only", k=1, lp=0.0, lc=1.0, REMOTE=REMOTE),
    )
  # OntoNotes few-shot: 4-way (sample 4 entity types per episode; aligns with Table 2)
    write_yaml(
        "fewshot_ontonotes_4w1s_autodl.yaml",
        fewshot_tpl.format(run="ontonotes_fewshot_4w1s", k=1, lp=0.5, lc=0.1, REMOTE=REMOTE)
        .replace("configs/dataset/conll2003_autodl.yaml", "configs/dataset/ontonotes_autodl.yaml")
        .replace("fewshot:\n  n_way:", "fewshot:\n  method: ours\n  n_way:"),
    )
    write_yaml(
        "fewshot_ontonotes_4w5s_autodl.yaml",
        fewshot_tpl.format(run="ontonotes_fewshot_4w5s", k=5, lp=0.5, lc=0.1, REMOTE=REMOTE)
        .replace("configs/dataset/conll2003_autodl.yaml", "configs/dataset/ontonotes_autodl.yaml")
        .replace("fewshot:\n  n_way:", "fewshot:\n  method: ours\n  n_way:"),
    )

    bert_fewshot_tpl = """name: {run}
dataset_config: configs/dataset/conll2003_autodl.yaml
model_config: configs/model/bert_crf_base.yaml
fewshot:
  method: {method}
  n_way: 4
  k_shot: {k}
  num_episodes: 5
  query_size: 200
train:
  epochs: {epochs}
  batch_size: 16
  lr: 0.00005
  weight_decay: 0.01
  max_len: 128
  device: cuda
  seed: 42
  lambda_proto: {lp}
  lambda_contrastive: {lc}
  lambda_struct: {ls}
  support_only: {support_only}
  nn_k: {nn_k}
  lr_scheduler: plateau
  scheduler_patience: 2
  scheduler_factor: 0.5
  output_dir: {REMOTE}/outputs/runs/{run}
"""
    for method, k, run_suffix, epochs, lp, lc, ls, support_only, nn_k in [
        ("protobert", 1, "protobert_4w1s", 5, 1.0, 0.0, 0.0, "false", 1),
        ("protobert", 5, "protobert_4w5s", 5, 1.0, 0.0, 0.0, "false", 1),
        ("nnshot", 1, "nnshot_4w1s", 3, 0.0, 0.0, 0.0, "true", 1),
        ("nnshot", 5, "nnshot_4w5s", 3, 0.0, 0.0, 0.0, "true", 1),
        ("structshot", 1, "structshot_4w1s", 5, 0.5, 0.0, 0.2, "false", 1),
        ("structshot", 5, "structshot_4w5s", 5, 0.5, 0.0, 0.2, "false", 1),
    ]:
        write_yaml(
            f"fewshot_conll2003_{run_suffix}_autodl.yaml",
            bert_fewshot_tpl.format(
                run=f"conll2003_fewshot_{run_suffix}",
                method=method,
                k=k,
                epochs=epochs,
                lp=lp,
                lc=lc,
                ls=ls,
                support_only=support_only,
                nn_k=nn_k,
                REMOTE=REMOTE,
            ),
        )


if __name__ == "__main__":
    main()
