#!/bin/bash
#SBATCH --job-name=tomato-train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:l4:1
#SBATCH --partition=gpu-24c-l4-4g

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p logs

# Activate the tomato-ml conda environment
eval "$(conda shell.bash hook)"
conda activate tomato-ml

echo "Starting ResNet34 Training..."
python src/train.py
echo "Training Complete!"

echo "Evaluating on Standard Test Set..."
python src/evaluate.py

echo "All evaluations complete!"
