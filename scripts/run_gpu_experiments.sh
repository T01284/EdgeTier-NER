#!/usr/bin/env bash
# Full GPU experiment pipeline: multi-dataset, baselines, ablations, few-shot.
set -euo pipefail
cd /root/EdgeFS_NER
source .venv/bin/activate

# HuggingFace mirror (mainland China / AutoDL)
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HUGGINGFACE_HUB_ENDPOINT="${HF_ENDPOINT}"
export HF_HUB_ENABLE_HF_TRANSFER=0

mkdir -p /root/autodl-tmp/EdgeFS_NER/{data,outputs/runs,outputs/export,logs}

LOG=/root/autodl-tmp/EdgeFS_NER/logs/gpu_experiments.log
exec > >(tee -a "$LOG") 2>&1
echo "=== GPU experiments start $(date -u +%FT%TZ) HF_ENDPOINT=$HF_ENDPOINT ==="

pip install -q transformers accelerate

run_prepare() {
  local cfg="$1"
  local marker="$2"
  if [ ! -f "$marker" ]; then
    echo "== prepare $(basename "$cfg") =="
    python scripts/prepare_datasets.py --config "$cfg"
  else
    echo "== skip prepare $(basename "$cfg") =="
  fi
}

run_train() {
  local metrics="$1"
  shift
  if [ ! -f "$metrics" ]; then
    echo "== train $* =="
    python scripts/train_full.py "$@" --device cuda
  else
    echo "== skip train (exists $metrics) =="
  fi
}

run_fewshot() {
  local summary="$1"
  shift
  if [ ! -f "$summary" ]; then
    echo "== fewshot $* =="
    python scripts/train_fewshot.py "$@" --device cuda
  else
    echo "== skip fewshot (exists $summary) =="
  fi
}

BASE=/root/autodl-tmp/EdgeFS_NER

# --- CoNLL dataset (required for first batch of experiments) ---
run_prepare configs/dataset/conll2003_autodl.yaml "$BASE/data/processed/conll2003/train.conll"

# --- CoNLL-2003: run before multi-dataset prep (avoids HF blocking BERT baselines) ---
run_train "$BASE/outputs/runs/conll2003_full/metrics.json" --config configs/train/full_conll2003_autodl.yaml
run_train "$BASE/outputs/runs/conll2003_bilstm/metrics.json" --config configs/train/full_conll2003_bilstm_autodl.yaml
run_train "$BASE/outputs/runs/conll2003_bilstm_v3/metrics.json" --config configs/train/full_conll2003_bilstm_v3_autodl.yaml
run_train "$BASE/outputs/runs/conll2003_ablation_wordlevel/metrics.json" --config configs/train/ablation_conll2003_wordlevel_autodl.yaml
run_train "$BASE/outputs/runs/conll2003_distill/metrics.json" --config configs/train/full_conll2003_distill_autodl.yaml
run_train "$BASE/outputs/runs/conll2003_bert/metrics.json" --config configs/train/full_conll2003_bert_autodl.yaml
run_train "$BASE/outputs/runs/conll2003_distilbert/metrics.json" --config configs/train/full_conll2003_distilbert_autodl.yaml
run_train "$BASE/outputs/runs/conll2003_tinybert/metrics.json" --config configs/train/full_conll2003_tinybert_autodl.yaml

# --- Other datasets (HF mirror + GitHub mirrors) ---
run_prepare configs/dataset/ontonotes_autodl.yaml "$BASE/data/processed/ontonotes/train.conll"
run_prepare configs/dataset/cluener_autodl.yaml "$BASE/data/processed/cluener/train.conll"

# --- CoNLL few-shot ablations (skip if done) ---
run_fewshot "$BASE/outputs/runs/conll2003_fewshot_5w1s/fewshot_summary.json" --config configs/train/fewshot_conll2003_autodl.yaml
run_fewshot "$BASE/outputs/runs/conll2003_fewshot_5w5s/fewshot_summary.json" --config configs/train/fewshot_conll2003_5w5s_autodl.yaml
run_fewshot "$BASE/outputs/runs/conll2003_fewshot_proto_only/fewshot_summary.json" --config configs/train/fewshot_conll2003_proto_only_autodl.yaml
run_fewshot "$BASE/outputs/runs/conll2003_fewshot_contrastive_only/fewshot_summary.json" --config configs/train/fewshot_conll2003_contrastive_only_autodl.yaml

# --- Phase 2 gate: require approval marker before heavy OntoNotes/CLUENER runs ---
PHASE2_OK="$BASE/logs/phase2_approved"
if [ ! -f "$PHASE2_OK" ]; then
  echo "== PHASE1 complete; pausing before OntoNotes/CLUENER =="
  echo "== To continue: touch $PHASE2_OK and re-run this script =="
  exit 0
fi

# --- OntoNotes full ---
run_train "$BASE/outputs/runs/ontonotes_full/metrics.json" --config configs/train/full_ontonotes_autodl.yaml
run_train "$BASE/outputs/runs/ontonotes_bilstm/metrics.json" --config configs/train/full_ontonotes_bilstm_autodl.yaml
run_train "$BASE/outputs/runs/ontonotes_bert/metrics.json" --config configs/train/full_ontonotes_bert_autodl.yaml
run_train "$BASE/outputs/runs/ontonotes_distilbert/metrics.json" --config configs/train/full_ontonotes_distilbert_autodl.yaml
run_train "$BASE/outputs/runs/ontonotes_tinybert/metrics.json" --config configs/train/full_ontonotes_tinybert_autodl.yaml

# --- CLUENER full ---
run_train "$BASE/outputs/runs/cluener_full/metrics.json" --config configs/train/full_cluener_autodl.yaml
run_train "$BASE/outputs/runs/cluener_bilstm/metrics.json" --config configs/train/full_cluener_bilstm_autodl.yaml
run_train "$BASE/outputs/runs/cluener_bert/metrics.json" --config configs/train/full_cluener_bert_autodl.yaml
run_train "$BASE/outputs/runs/cluener_distilbert/metrics.json" --config configs/train/full_cluener_distilbert_autodl.yaml
run_train "$BASE/outputs/runs/cluener_tinybert/metrics.json" --config configs/train/full_cluener_tinybert_autodl.yaml

# --- Few-shot CoNLL (skip if done) ---
# (moved above, before OntoNotes)

# --- Few-shot OntoNotes (5-way) ---
run_fewshot "$BASE/outputs/runs/ontonotes_fewshot_5w1s/fewshot_summary.json" --config configs/train/fewshot_ontonotes_5w1s_autodl.yaml
run_fewshot "$BASE/outputs/runs/ontonotes_fewshot_5w5s/fewshot_summary.json" --config configs/train/fewshot_ontonotes_5w5s_autodl.yaml

# --- export ---
CKPT="$BASE/outputs/runs/conll2003_full/best.pt"
EXPORT_DIR="$BASE/outputs/export/conll2003_full"
if [ -f "$CKPT" ] && [ ! -f "$EXPORT_DIR/deploy/model_emissions.onnx" ]; then
  echo "== export best checkpoint =="
  python scripts/export_model.py --checkpoint "$CKPT" --output-dir "$EXPORT_DIR"
else
  echo "== skip export =="
fi

echo "=== GPU experiments done $(date -u +%FT%TZ) ==="
