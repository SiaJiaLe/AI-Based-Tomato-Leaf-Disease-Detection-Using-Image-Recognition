#!/usr/bin/env bash
# Full training pipeline — run inside model_trainer container:
#   bash scripts/run_training.sh
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> prepare_dataset"
python scripts/prepare_dataset.py

echo "==> train"
python src/train.py

echo "==> evaluate"
python src/evaluate.py

echo "==> export"
python src/export.py

echo "==> Done. Check outputs/ for artifacts."
