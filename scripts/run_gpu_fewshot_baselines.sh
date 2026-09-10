#!/usr/bin/env bash
# Few-shot baselines: ProtoBERT, NNShot, StructShot on CoNLL-2003.
set -euo pipefail
cd /root/EdgeFS_NER
source .venv/bin/activate

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HUGGINGFACE_HUB_ENDPOINT="${HF_ENDPOINT}"
export HF_HUB_ENABLE_HF_TRANSFER=0

BASE=/root/autodl-tmp/EdgeFS_NER
mkdir -p "$BASE/logs"
LOG="$BASE/logs/gpu_fewshot_baselines.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== Few-shot baselines start $(date -u +%FT%TZ) ==="

pip install -q transformers accelerate

CONFIGS=(
  configs/train/fewshot_conll2003_protobert_4w1s_autodl.yaml
  configs/train/fewshot_conll2003_protobert_4w5s_autodl.yaml
  configs/train/fewshot_conll2003_nnshot_4w1s_autodl.yaml
  configs/train/fewshot_conll2003_nnshot_4w5s_autodl.yaml
  configs/train/fewshot_conll2003_structshot_4w1s_autodl.yaml
  configs/train/fewshot_conll2003_structshot_4w5s_autodl.yaml
)

for cfg in "${CONFIGS[@]}"; do
  summary=$(python - <<PY
import yaml
from pathlib import Path
c = yaml.safe_load(Path("$cfg").read_text(encoding="utf-8"))
print(Path(c["train"]["output_dir"]) / "fewshot_summary.json")
PY
)
  if [ ! -f "$summary" ]; then
    echo "== fewshot $cfg =="
    python scripts/train_fewshot.py --config "$cfg" --device cuda
  else
    echo "== skip fewshot (exists $summary) =="
  fi
done

echo "=== Few-shot baselines done $(date -u +%FT%TZ) ==="
