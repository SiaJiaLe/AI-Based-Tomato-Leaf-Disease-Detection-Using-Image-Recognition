#!/usr/bin/env bash
# Full training pipeline — scaffold
set -euo pipefail
python scripts/prepare_dataset.py
python src/train.py
python src/evaluate.py
python src/export.py
