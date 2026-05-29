#!/bin/bash
#SBATCH --job-name=tomato-train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
# TODO: set partition for Sunway HPC
#SBATCH --partition=gpu

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p logs

# Path A: container (uncomment and set image path)
# bash scripts/hpc/run_training_docker.sh

# Path B: venv fallback (uncomment)
# source ~/venvs/tomato-ml/bin/activate
# module load cuda
# python src/train.py

echo "Configure train_slurm.sh for Sunway HPC before submitting."
