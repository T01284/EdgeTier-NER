#!/usr/bin/env bash
# Run once on Raspberry Pi 3B (64-bit Raspberry Pi OS)
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

mkdir -p assets/deploy results
echo "Copy outputs/export/deploy/* to assets/deploy/ before running benchmarks."
