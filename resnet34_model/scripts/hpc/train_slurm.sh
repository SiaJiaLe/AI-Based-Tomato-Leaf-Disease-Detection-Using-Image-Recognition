#!/bin/bash
#SBATCH --job-name=tomato-train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
# TODO: If your HPC requires a specific partition, run 'sinfo' to find it and uncomment/edit the line below:
# #SBATCH --partition=gpu

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
