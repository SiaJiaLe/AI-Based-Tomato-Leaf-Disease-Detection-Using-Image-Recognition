#!/bin/bash
#SBATCH --job-name=plan2-tier1-droppath
#SBATCH --output=experiments/results/plan2_tier1_%j.out
#SBATCH --error=experiments/results/plan2_tier1_%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:l4:1
#SBATCH --partition=gpu-8c-l4-1g

# Plan 2 Tier 1 — stochastic depth (drop-path) for EfficientNetB0.
# Sweeps drop_path_rate in {0.2, 0.3}, selects the winner on PlantVillage VAL
# macro-F1, then reads the test sets ONCE for that winner only.
#
# Submit from the repo root:
#     sbatch experiments/plan2_arch/run_droppath_slurm.sh
#
# No rembg / no background randomization here — this row is the plain
# efficientnetb0_on recipe plus drop-path. Only the standard tomato-ml env.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p experiments/results

eval "$(conda shell.bash hook)"
conda activate tomato-ml

CFG_DIR=experiments/plan2_arch/configs

# --- Sweep: train both rates WITHOUT touching any test set ---
for RATE in 02 03; do
  echo "=== Training efficientnetb0_on_droppath${RATE} (train-only, val selection) ==="
  python -m experiments.plan2_arch.run_arch \
    --config "$CFG_DIR/efficientnetb0_on_droppath${RATE}.yaml" --train-only
done

# --- Select on VAL macro-F1 only ---
echo "=== Selecting drop_path_rate on validation ==="
python -m experiments.plan2_arch.select_on_val \
  efficientnetb0_on_droppath02 efficientnetb0_on_droppath03

WINNER=$(python -m experiments.plan2_arch.select_on_val \
  efficientnetb0_on_droppath02 efficientnetb0_on_droppath03 --print-winner)
echo "Winner on val: $WINNER"

# --- Read the test sets ONCE, for the winner only ---
echo "=== Evaluating $WINNER (real-world read once) ==="
python -m experiments.plan2_arch.run_arch --config "$CFG_DIR/${WINNER}.yaml" --eval-only

echo "=== Comparing $WINNER against efficientnetb0_on ==="
python -m experiments.plan2_arch.compare_arch --run "$WINNER"
echo "Done. Results in experiments/results/${WINNER}/."
