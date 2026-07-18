#!/bin/bash
#SBATCH --job-name=confusion-matrices
#SBATCH --output=experiments/results/confusion_%j.out
#SBATCH --error=experiments/results/confusion_%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:l4:1
#SBATCH --partition=gpu-8c-l4-1g

# Recover the real-world confusion matrix of every EVALUATED row as numbers.
#
# NO TRAINING. This job loads frozen checkpoints and runs inference over the
# real-world test set only, so it costs minutes, not hours. 30 minutes of wall is
# already generous.
#
# Why it is needed at all: common/evaluate.py draws cm_real_world_test.png for
# every row but throws the COUNTS away (evaluate.py:85 returns them; nobody
# catches it). The pictures cannot be compared, ranked, or tabulated.
#
# Why it does not break the read-once rule: frozen weights, deterministic
# inference, no selection of any kind — these are the same predictions whose
# aggregate is already published. Each row must reproduce its published
# real-world macro-F1 to 1e-6 or the job aborts. Rows that were never evaluated
# (the val-only sweep losers) have no published results and are therefore never
# touched. See experiments/confusion_matrices.py's docstring for the full argument.
#
# Submit from the repo root:
#     sbatch experiments/run_confusion_slurm.sh
#
# Re-running is free after the first pass: the counts are cached as JSON, so
#     python -m experiments.confusion_matrices --report-only
# rebuilds every table and figure on a login node with no GPU and no inference.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p experiments/results

eval "$(conda shell.bash hook)"
conda activate tomato-ml

# Fail in seconds rather than after queueing for a GPU and dying on an import.
echo "=== [1/2] Preflight: environment ==="
python -m experiments.preflight_env

echo "=== [2/2] Recomputing real-world confusion matrices ==="
python -m experiments.confusion_matrices

echo
echo "Done. Look at:"
echo "  experiments/results/confusion/confusion_report.txt   (aligned tables)"
echo "  experiments/results/confusion/confusion_report.md    (paste into the report)"
echo "  experiments/results/confusion/grid_efficientnetb0.png (the thesis figure)"
echo "  experiments/results/confusion/<run>.png               (one matrix per row)"
