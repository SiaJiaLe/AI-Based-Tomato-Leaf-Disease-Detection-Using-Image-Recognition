#!/bin/bash
#SBATCH --job-name=plan1-bgrand
#SBATCH --output=experiments/results/plan1_bgrand_%j.out
#SBATCH --error=experiments/results/plan1_bgrand_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:l4:1
#SBATCH --partition=gpu-24c-l4-4g

# Trains + evaluates the single Plan 1 run (EfficientNetB0 Stack-ON + background
# randomization) and compares it against the efficientnetb0_on baseline.
# Submit from the repo root:  sbatch experiments/plan1_bgrand/run_bgrand_slurm.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p experiments/results

eval "$(conda shell.bash hook)"
conda activate tomato-ml

CFG=experiments/plan1_bgrand/configs/efficientnetb0_on_bgrand.yaml

# 0. rembg (U^2-Net) for segmentation: pretrained. BEST DONE ON THE LOGIN NODE
#    BEFOREHAND (needs internet). Left here guarded in case the node has access.
python -c "import rembg" 2>/dev/null || pip install -r experiments/plan1_bgrand/requirements.txt || \
  echo "WARN: rembg install failed — run it on the login node, or set segmentation: hsv_threshold."
python -c "from rembg import new_session; new_session('u2net')" 2>/dev/null || \
  echo "WARN: could not pre-download u2net — run the pre-download on the login node."

# 1. Generate synthetic backgrounds if the folder is empty (idempotent).
if [ -z "$(ls -A data/backgrounds_generic 2>/dev/null | grep -Ei '\.(jpg|jpeg|png|bmp|tif|tiff|webp)$' || true)" ]; then
  echo "No backgrounds found — generating synthetic textures..."
  python -m experiments.plan1_bgrand.make_synthetic_backgrounds
fi

# 2. Mask/composite sanity grids (quick; inspect the PNGs afterwards).
#    Uses the same segmentation as the config (pretrained/rembg by default).
echo "Writing mask sanity grids..."
python -m experiments.plan1_bgrand.sanity_check_masks --per-class 3 --segmentation pretrained

# 3. Train + evaluate the Plan 1 run.
echo "Training efficientnetb0_on_bgrand..."
python -m experiments.plan1_bgrand.run_bgrand --config "$CFG"

# 4. Compare against the baseline efficientnetb0_on.
echo "Comparing against baseline..."
python -m experiments.plan1_bgrand.compare_bgrand
echo "Done. Results in experiments/results/efficientnetb0_on_bgrand/."
