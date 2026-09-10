#!/usr/bin/env bash
# Run on remote GPU server after sync_to_remote.ps1
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/prepare_datasets.py --config configs/dataset/conll2003.yaml
python scripts/train_full.py --config configs/train/full_conll2003.yaml --device cuda
python scripts/train_fewshot.py --config configs/train/fewshot_conll2003.yaml --device cuda
