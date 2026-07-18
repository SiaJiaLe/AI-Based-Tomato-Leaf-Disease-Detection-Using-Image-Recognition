#!/bin/bash
#SBATCH --job-name=seedrep
#SBATCH --output=experiments/results/seedrep_%A_%a.out
#SBATCH --error=experiments/results/seedrep_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:l4:1
#SBATCH --partition=gpu-8c-l4-1g
#SBATCH --array=0-6

# 3-seed replication of the realism result (Plan 6). One array task per config:
# baseline (prob 0) x {42,43,44} + synthetic/real x {43,44}. Seed-42 synthetic/real
# already exist (efficientnetb0_on_bgrand / _bgrand_real) and are NOT retrained here.
#
# All 7 runs go through the SAME run_bgrand.py -> evaluate_run path, so the only
# variable across the matrix is the background treatment (prob / background_dir).
# Each run bakes real_world_dir=data/real_environment_dataset into its checkpoint, so
# it evaluates on the cleaned 333-image real set — same set as everything else.
#
# Submit from the repo root:  sbatch experiments/plan1_bgrand/run_seedrep_slurm.sh
# After all tasks finish:      python -m experiments.plan1_bgrand.compare_seeds

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p experiments/results

eval "$(conda shell.bash hook)"
conda activate tomato-ml

# One config per array index. Order is fixed; do not reorder without updating 0-6.
CFGS=(
  experiments/plan1_bgrand/configs/seedrep/base_s42.yaml
  experiments/plan1_bgrand/configs/seedrep/base_s43.yaml
  experiments/plan1_bgrand/configs/seedrep/base_s44.yaml
  experiments/plan1_bgrand/configs/seedrep/bgrand_s43.yaml
  experiments/plan1_bgrand/configs/seedrep/bgrand_s44.yaml
  experiments/plan1_bgrand/configs/seedrep/bgrandreal_s43.yaml
  experiments/plan1_bgrand/configs/seedrep/bgrandreal_s44.yaml
)
CFG="${CFGS[$SLURM_ARRAY_TASK_ID]}"
echo "=== task $SLURM_ARRAY_TASK_ID -> $CFG ==="

# Preflight. NOTHING is installed from inside this job (see run_bgrand_slurm.sh for
# why: installing into the shared conda env from a batch job corrupts it). Install
# rembg + pre-download u2net on the LOGIN NODE beforehand.
echo "Preflight: environment..."
python -m experiments.preflight_env

if ! python -c "import rembg" 2>/dev/null; then
  echo "ERROR: rembg is not importable, but the bgrand/bgrand_real configs use" >&2
  echo "       segmentation: pretrained. Install it ON THE LOGIN NODE:" >&2
  echo "           pip install -r experiments/plan1_bgrand/requirements.txt" >&2
  echo "           python -c \"from rembg import new_session; new_session('u2net')\"" >&2
  exit 1
fi
python -c "from rembg import new_session; new_session('u2net')" 2>/dev/null || {
  echo "ERROR: u2net model is not available offline. Pre-download it on the login node:" >&2
  echo "           python -c \"from rembg import new_session; new_session('u2net')\"" >&2
  exit 1
}

# Background folders. Real CC0 photos must already be present; synthetic is generated
# on demand (idempotent). Baseline configs point at the real folder but never sample it.
if [ -z "$(ls -A data/backgrounds_generic_real 2>/dev/null | grep -Ei '\.(jpg|jpeg|png|bmp|tif|tiff|webp)$' || true)" ]; then
  echo "ERROR: data/backgrounds_generic_real is empty. Put the real CC0 backgrounds there first." >&2
  exit 1
fi
if [ -z "$(ls -A data/backgrounds_generic_synthetic 2>/dev/null | grep -Ei '\.(jpg|jpeg|png|bmp|tif|tiff|webp)$' || true)" ]; then
  echo "No synthetic backgrounds — generating..."
  python -m experiments.plan1_bgrand.make_synthetic_backgrounds
fi

# Train + evaluate this seed/recipe. Masks are reused from data/mask_cache (keyed by
# leaf image), so seeds 43/44 do not re-run U^2-Net; baseline does no segmentation.
python -m experiments.plan1_bgrand.run_bgrand --config "$CFG"
echo "=== task $SLURM_ARRAY_TASK_ID done ==="
