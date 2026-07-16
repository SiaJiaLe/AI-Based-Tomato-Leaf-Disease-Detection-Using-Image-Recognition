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

# 0. Preflight. NOTHING is installed from inside this job any more.
#    The old version ran `pip install -r experiments/plan1_bgrand/requirements.txt`
#    here. Unpinned, that resolved rembg to 2.0.76, which demands numpy>=2.3.0
#    while torch 2.3.0 requires numpy<2 — pip moved numpy, collided with torch,
#    and left a half-written torch behind. Every later job then died on
#    `import torch.nn`. Installing into a shared conda env from a batch job is
#    how envs get destroyed; do it on the login node, from pinned requirements.
echo "Preflight: environment..."
python -m experiments.preflight_env

if ! python -c "import rembg" 2>/dev/null; then
  echo "ERROR: rembg is not importable, but this config uses segmentation: pretrained." >&2
  echo "       Install it ON THE LOGIN NODE (it needs internet anyway):" >&2
  echo "           pip install -r experiments/plan1_bgrand/requirements.txt" >&2
  echo "           python -c \"from rembg import new_session; new_session('u2net')\"" >&2
  exit 1
fi
# Pre-download check only — the model download needs internet, which compute
# nodes generally lack. A cached u2net makes this a no-op.
python -c "from rembg import new_session; new_session('u2net')" 2>/dev/null || {
  echo "ERROR: u2net model is not available offline. Pre-download it on the login node:" >&2
  echo "           python -c \"from rembg import new_session; new_session('u2net')\"" >&2
  exit 1
}

# 1. Generate synthetic backgrounds if the folder is empty (idempotent).
if [ -z "$(ls -A data/backgrounds_generic_synthetic 2>/dev/null | grep -Ei '\.(jpg|jpeg|png|bmp|tif|tiff|webp)$' || true)" ]; then
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
