#!/bin/bash
# Re-split the NEW dataset and retrain EVERYTHING in parallel, then post-process.
# RUN THIS ON THE LOGIN NODE (it splits, archives, and submits jobs - no GPU itself):
#
#     bash experiments/retrain_all.sh
#     bash experiments/retrain_all.sh --clear-mask-cache   # also wipe old bgrand masks
#
# What it does, in order:
#   1. Preflight (fail fast BEFORE touching anything): data/raw populated; real+synthetic
#      backgrounds present; rembg + u2net importable (bgrand/seedrep need them).
#   2. Re-split repo-root data/raw -> data/processed (70/15/15, seed 42, symlinks).
#   3. Archive experiments/results -> experiments/results_archive_<timestamp>/ (the old
#      results are gitignored and exist only here), then start a clean results/.
#   4. sbatch every training job so they queue and run CONCURRENTLY:
#        ablation array (12) + bgrand (2) + seedrep array (7) + Plan 2 tiers (3).
#   5. sbatch the post-processing job with an afterok dependency on all of them, so the
#      master tables + aggregate confusion matrices + seed summary build automatically
#      once training finishes.

set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"

CLEAR_MASKS=0
[ "${1:-}" = "--clear-mask-cache" ] && CLEAR_MASKS=1

eval "$(conda shell.bash hook)"
conda activate tomato-ml

echo "=== [1/5] Preflight ==="
python -m experiments.preflight_env

if [ -z "$(ls -A data/raw 2>/dev/null || true)" ]; then
  echo "ERROR: data/raw is empty. Put the new dataset's 10 class folders there first." >&2
  exit 1
fi
for bg in data/backgrounds_generic_real data/backgrounds_generic_synthetic; do
  if [ -z "$(ls -A "$bg" 2>/dev/null | grep -Ei '\.(jpg|jpeg|png|bmp|tif|tiff|webp)$' || true)" ]; then
    echo "ERROR: $bg is empty - bgrand/seedrep need it. (Synthetic can be generated with" >&2
    echo "       python -m experiments.plan1_bgrand.make_synthetic_backgrounds.)" >&2
    exit 1
  fi
done
if ! python -c "import rembg; from rembg import new_session; new_session('u2net')" 2>/dev/null; then
  echo "ERROR: rembg / u2net not available. On the LOGIN NODE (has internet):" >&2
  echo "       pip install -r experiments/plan1_bgrand/requirements.txt" >&2
  echo "       python -c \"from rembg import new_session; new_session('u2net')\"" >&2
  exit 1
fi
echo "Preflight OK."

echo ""; echo "=== [2/5] Re-split data/raw -> data/processed ==="
python -m experiments.split_dataset

echo ""; echo "=== [3/5] Archive old results ==="
if [ -d experiments/results ] && [ -n "$(ls -A experiments/results 2>/dev/null || true)" ]; then
  STAMP="$(date +%Y%m%d_%H%M%S)"
  ARCHIVE="experiments/results_archive_${STAMP}"
  mv experiments/results "$ARCHIVE"
  echo "Old results moved to $ARCHIVE/ (seed-replication + confusion analysis preserved there)."
fi
mkdir -p experiments/results
if [ "$CLEAR_MASKS" -eq 1 ]; then
  rm -rf data/mask_cache && echo "Cleared data/mask_cache (bgrand/seedrep will re-segment)."
fi

echo ""; echo "=== [4/5] Submit training jobs (parallel) ==="
ABL=$(sbatch --parsable experiments/run_ablation_array_slurm.sh)
BG1=$(sbatch --parsable experiments/plan1_bgrand/run_bgrand_slurm.sh)
BG2=$(sbatch --parsable experiments/plan1_bgrand/run_bgrand_real_slurm.sh)
SEED=$(sbatch --parsable experiments/plan1_bgrand/run_seedrep_slurm.sh)
DP=$(sbatch --parsable experiments/plan2_arch/run_droppath_slurm.sh)
RES=$(sbatch --parsable experiments/plan2_arch/run_res240_slurm.sh)
MIX=$(sbatch --parsable experiments/plan2_arch/run_mixstyle_slurm.sh)
echo "  ablation array : $ABL"
echo "  bgrand         : $BG1"
echo "  bgrand_real    : $BG2"
echo "  seedrep array  : $SEED"
echo "  droppath (T1)  : $DP"
echo "  res240   (T2)  : $RES"
echo "  mixstyle (T3)  : $MIX"

echo ""; echo "=== [5/5] Submit post-processing (runs after all training) ==="
DEP="afterok:${ABL}:${BG1}:${BG2}:${SEED}:${DP}:${RES}:${MIX}"
POST=$(sbatch --parsable --dependency="$DEP" experiments/run_postprocess_slurm.sh)
echo "  postprocess    : $POST  (waits for all training to succeed)"

echo ""
echo "All submitted. Watch progress:"
echo "    squeue -u \$USER"
echo "If a training job fails, afterok holds postprocess back - fix that job, then:"
echo "    sbatch experiments/run_postprocess_slurm.sh"
