#!/usr/bin/env bash
# Run GPU experiment pipeline on SeetaCloud (AutoDL).
set -euo pipefail
cd /root/EdgeFS_NER
source .venv/bin/activate
mkdir -p /root/autodl-tmp/EdgeFS_NER/{data,outputs/runs,logs}

LOG=/root/autodl-tmp/EdgeFS_NER/logs/gpu_pipeline.log
exec > >(tee -a "$LOG") 2>&1
echo "=== GPU pipeline start $(date -u +%FT%TZ) ==="

echo "== prepare CoNLL-2003 =="
python scripts/prepare_datasets.py --config configs/dataset/conll2003_autodl.yaml

if [ ! -f /root/autodl-tmp/EdgeFS_NER/outputs/runs/conll2003_full/metrics.json ]; then
  echo "== train Ours (char-cnn-crf) =="
  python scripts/train_full.py --config configs/train/full_conll2003_autodl.yaml --device cuda
else
  echo "== skip Ours full (metrics exist) =="
fi

if [ ! -f /root/autodl-tmp/EdgeFS_NER/outputs/runs/conll2003_bilstm/metrics.json ]; then
  echo "== train BiLSTM-CRF baseline =="
  python scripts/train_full.py --config configs/train/full_conll2003_bilstm_autodl.yaml --device cuda
else
  echo "== skip BiLSTM (metrics exist) =="
fi

if [ ! -f /root/autodl-tmp/EdgeFS_NER/outputs/runs/conll2003_fewshot_5w1s/fewshot_summary.json ]; then
  echo "== few-shot 5w1s =="
  python scripts/train_fewshot.py --config configs/train/fewshot_conll2003_autodl.yaml --device cuda
else
  echo "== skip few-shot 5w1s (summary exist) =="
fi

if [ ! -f /root/autodl-tmp/EdgeFS_NER/outputs/runs/conll2003_fewshot_5w5s/fewshot_summary.json ]; then
  echo "== few-shot 5w5s =="
  python scripts/train_fewshot.py --config configs/train/fewshot_conll2003_5w5s_autodl.yaml --device cuda
else
  echo "== skip few-shot 5w5s (summary exist) =="
fi

echo "== export best checkpoint =="
CKPT=/root/autodl-tmp/EdgeFS_NER/outputs/runs/conll2003_full/best.pt
EXPORT_DIR=/root/autodl-tmp/EdgeFS_NER/outputs/export/conll2003_full
if [ -f "$CKPT" ] && [ ! -f "$EXPORT_DIR/deploy/model_emissions.onnx" ]; then
  python scripts/export_model.py \
    --checkpoint "$CKPT" \
    --output-dir "$EXPORT_DIR"
elif [ -f "$EXPORT_DIR/deploy/model_emissions.onnx" ]; then
  echo "== skip export (model_emissions.onnx exist) =="
fi

echo "=== GPU pipeline done $(date -u +%FT%TZ) ==="
