#!/bin/bash
#SBATCH --job-name=plan1-bgrand-real
#SBATCH --output=experiments/results/plan1_bgrand_real_%j.out
#SBATCH --error=experiments/results/plan1_bgrand_real_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:l4:1
#SBATCH --partition=gpu-24c-l4-4g

# REAL-BACKGROUND variant of Plan 1: same pipeline as run_bgrand_slurm.sh but
# backgrounds come from data/backgrounds_generic_real (real CC0 photos you dropped in),
# and the run_name is efficientnetb0_on_bgrand_real so the synthetic-backdrop
# result is preserved. The mask cache (data/mask_cache) is reused, so this run
# is fast from epoch 1. Submit from the repo root:
#     sbatch experiments/plan1_bgrand/run_bgrand_real_slurm.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p experiments/results

eval "$(conda shell.bash hook)"
conda activate tomato-ml

CFG=experiments/plan1_bgrand/configs/efficientnetb0_on_bgrand_real.yaml

# Preflight the env before spending the GPU allocation. This script never
# installs anything — see run_bgrand_slurm.sh's note and
# experiments/plan1_bgrand/requirements.txt for why that rule exists.
echo "Preflight: environment..."
python -m experiments.preflight_env

if ! python -c "import rembg" 2>/dev/null; then
  echo "ERROR: rembg is not importable, but this config uses segmentation: pretrained." >&2
  echo "       Install it ON THE LOGIN NODE:" >&2
  echo "           pip install -r experiments/plan1_bgrand/requirements.txt" >&2
  exit 1
fi

# Fail early with a clear message if the real-background folder is empty.
if [ -z "$(ls -A data/backgrounds_generic_real 2>/dev/null | grep -Ei '\.(jpg|jpeg|png|bmp|tif|tiff|webp)$' || true)" ]; then
  echo "ERROR: data/backgrounds_generic_real has no images. Drop 30-100 real CC0 photos there first."
  exit 1
fi

# Sanity grids with the real backgrounds (uses the same rembg masks).
echo "Writing mask sanity grids (real backgrounds)..."
python -m experiments.plan1_bgrand.sanity_check_masks --per-class 3 \
  --segmentation pretrained --background-dir data/backgrounds_generic_real

echo "Training efficientnetb0_on_bgrand_real..."
python -m experiments.plan1_bgrand.run_bgrand --config "$CFG"

echo "Comparing real-background run against baseline..."
python -m experiments.plan1_bgrand.compare_bgrand --bgrand efficientnetb0_on_bgrand_real
echo "Done. Results in experiments/results/efficientnetb0_on_bgrand_real/."
