#!/bin/bash
#SBATCH --job-name=reeval-real-world
#SBATCH --output=experiments/results/reeval_%j.out
#SBATCH --error=experiments/results/reeval_%j.err
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:l4:1
#SBATCH --partition=gpu-8c-l4-1g

# Re-measure every evaluated row against the NEW real-world set
# (data/real_environment_dataset), retiring data/processed/real_environment_test.
#
# NO TRAINING. Frozen checkpoints, inference over the real-world set only.
#
# BEFORE YOU SUBMIT THIS, run the check on the login node — it needs no GPU:
#
#     python -m experiments.reevaluate_real_world --check
#
# It counts images per class and refuses an incomplete set. This matters because a
# partially-downloaded set does NOT fail loudly: seven of ten classes at zero
# support still produces a number, and that number looks like a catastrophic
# modelling result rather than missing files. The job runs the same check first and
# exits before touching anything, so a bad set costs you nothing but the queue.
#
# The old results are ARCHIVED, not overwritten, into
# experiments/results/<run>/archive_real_environment_test/. Keep them: if the
# bgrand_real vs bgrand contrast (+0.0365 on the old set) reproduces on this new
# independent set, it stops being a single-seed curiosity and becomes a replicated
# finding. Two sets agreeing is a result; two disagreeing is also a result.
#
# Submit from the repo root:
#     sbatch experiments/run_reeval_slurm.sh
#
# Any arguments after the script name are forwarded to the re-evaluation step. Use
# this to REFRESH the current set after re-processing (cleaning) its images:
#
#     sbatch experiments/run_reeval_slurm.sh --refresh
#
# --refresh overwrites the new-set results in place, backs up the previous pass per
# run into archive_<set>_<timestamp>/, and does NOT touch the retired archive. The
# --check step below never receives these args, so it stays a plain set check.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p experiments/results

eval "$(conda shell.bash hook)"
conda activate tomato-ml

echo "=== [1/3] Preflight: environment ==="
python -m experiments.preflight_env

echo "=== [2/3] Checking the new real-world set ==="
# reevaluate_real_world runs this check itself and refuses to proceed on failure;
# running it here too means a bad set is reported before the GPU work starts.
python -m experiments.reevaluate_real_world --check

echo "=== [3/3] Re-evaluating every row on the new set ==="
# "$@" forwards any sbatch script args (e.g. --refresh) to the re-evaluation step.
python -m experiments.reevaluate_real_world "$@"

echo
echo "Done. The real-world column now refers to data/real_environment_dataset."
echo "Old numbers: experiments/results/<run>/archive_real_environment_test/"
echo
echo "Rebuild the tables (login node, no GPU needed):"
echo "    python -m experiments.compile_results"
echo "    python -m experiments.confusion_matrices --report-only"
