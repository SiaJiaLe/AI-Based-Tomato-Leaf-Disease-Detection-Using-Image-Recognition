#!/bin/bash
#SBATCH --job-name=postprocess
#SBATCH --output=experiments/results/postprocess_%j.out
#SBATCH --error=experiments/results/postprocess_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:l4:1
#SBATCH --partition=gpu-24c-l4-4g

# Runs AFTER all training/eval jobs finish (retrain_all.sh submits this with an
# afterok dependency). Regenerates the cross-cutting analysis from the freshly
# written eval_results*.json:
#   - compile_results      : the master + ablation metric tables
#   - confusion_matrices   : per-run raw-count matrices, the side-by-side grids, and
#                            confusion_counts.txt (compute mode -> needs GPU; re-infers
#                            each run and verifies it reproduces its own published F1)
#   - compare_seeds        : the 3-seed mean +/- std matrix and the paired contrasts
# Each run's OWN confusion PNG + per-class metrics are already written by evaluate_run
# during training; this step adds the aggregate views.
#
# If a training job failed, its rows are simply skipped; fix it and re-run this script.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

eval "$(conda shell.bash hook)"
conda activate tomato-ml

fail=0
run() {  # run a step, keep going if it errors so the others still produce output
  echo ""; echo "=== $* ==="
  if ! python -m "$@"; then
    echo "!! $* FAILED - re-run it manually after checking the training jobs." >&2
    fail=1
  fi
}

run experiments.compile_results
run experiments.confusion_matrices          # compute mode (GPU): counts + grids + confusion_counts.txt
run experiments.plan1_bgrand.compare_seeds

echo ""
if [ "$fail" -eq 0 ]; then
  echo "Post-processing complete. Tables + confusion matrices in experiments/results/ and experiments/results/confusion/."
else
  echo "Post-processing finished WITH ERRORS above - some analysis was skipped."
fi
echo "Per-model confusion matrices: experiments/results/<run>/cm_real_world_test.png"
echo "Aggregate counts (text):      experiments/results/confusion/confusion_counts.txt"
