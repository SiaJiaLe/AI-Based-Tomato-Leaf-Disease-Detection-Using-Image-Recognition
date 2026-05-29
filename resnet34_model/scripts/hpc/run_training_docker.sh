#!/usr/bin/env bash
# Singularity/Docker training wrapper — scaffold
set -euo pipefail
IMAGE="${TOMATO_TRAINER_IMAGE:-tomato-trainer.sif}"
# singularity exec --nv "$IMAGE" python src/train.py
echo "Set TOMATO_TRAINER_IMAGE and uncomment singularity exec for Sunway HPC."
