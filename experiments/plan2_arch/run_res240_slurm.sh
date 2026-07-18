#!/bin/bash
#SBATCH --job-name=plan2-tier2-res240
#SBATCH --output=experiments/results/plan2_tier2_%j.out
#SBATCH --error=experiments/results/plan2_tier2_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:l4:1
#SBATCH --partition=gpu-8c-l4-1g

# Plan 2 Tier 2 — input resolution 224 -> 240 for EfficientNetB0.
# Single row (the plan names ..._res240 and warns to keep the increment small),
# so there is no sweep and no val-selection step: train, then read the test sets
# once, then compare against efficientnetb0_on.
#
# Submit from the repo root:
#     sbatch experiments/plan2_arch/run_res240_slurm.sh
#
# Evaluation runs at 240 too (evaluate_res.py) — common.evaluate hardcodes 224,
# and evaluating a 240-trained model at 224 would silently degrade it.
# ~1.15x the compute of the 224 baseline; batch 32 fits an L4.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p experiments/results

eval "$(conda shell.bash hook)"
conda activate tomato-ml

CFG=experiments/plan2_arch/configs/efficientnetb0_on_res240.yaml

echo "=== Training + evaluating efficientnetb0_on_res240 ==="
python -m experiments.plan2_arch.run_arch --config "$CFG"

echo "=== Comparing efficientnetb0_on_res240 against efficientnetb0_on ==="
python -m experiments.plan2_arch.compare_arch --run efficientnetb0_on_res240
echo "Done. Results in experiments/results/efficientnetb0_on_res240/."
