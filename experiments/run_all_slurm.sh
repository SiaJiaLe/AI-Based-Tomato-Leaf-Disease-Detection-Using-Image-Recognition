#!/bin/bash
#SBATCH --job-name=cp2-ablation
#SBATCH --output=experiments/results/slurm_%j.out
#SBATCH --error=experiments/results/slurm_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:l4:1
#SBATCH --partition=gpu-24c-l4-4g

# Trains + evaluates all 12 ablation runs (6 OFF + 6 ON) in one GPU batch job.
# Submit from the repo root:  sbatch experiments/run_all_slurm.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p experiments/results

eval "$(conda shell.bash hook)"
conda activate tomato-ml

echo "Starting CP2 ablation — all 12 runs..."
python -m experiments.run --all
echo "Ablation complete. Building comparison tables + figure..."
python -m experiments.compare
echo "Done."
