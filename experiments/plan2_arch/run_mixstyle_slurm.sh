#!/bin/bash
#SBATCH --job-name=plan2-tier3-mixstyle
#SBATCH --output=experiments/results/plan2_tier3_%j.out
#SBATCH --error=experiments/results/plan2_tier3_%j.err
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:l4:1
#SBATCH --partition=gpu-24c-l4-4g

# Plan 2 Tier 3 — MixStyle, in two parts, in one job:
#
#   PART 1  MixStyle ALONE (standalone tier row, vs efficientnetb0_on)
#           sweep insertion depth {layers 1-2, layers 1-3} -> select on VAL ->
#           read the test sets ONCE for the winner -> compare.
#
#   PART 2  MixStyle + background randomization (COMBINATION row)
#           uses the depth that won Part 1's val sweep, plus Plan 1's bgrand
#           settings verbatim. Completes a 2x2 factorial:
#               baseline / bgrand-only / mixstyle-only / both.
#           NOT attributable to either factor alone — it is only interpretable
#           against the two single-factor rows.
#
# Part 2 deliberately inherits Part 1's VAL winner, so no selection ever touches
# the real-world set. The real-world set is read exactly twice in this job: once
# for the tier winner, once for the combination row. Both are distinct rows, so
# that is one read per row — the rule holds.
#
# Submit from the repo root:
#     sbatch experiments/plan2_arch/run_mixstyle_slurm.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p experiments/results

eval "$(conda shell.bash hook)"
conda activate tomato-ml

CFGDIR=experiments/plan2_arch/configs
L12=efficientnetb0_on_mixstyle_l12
L123=efficientnetb0_on_mixstyle_l123

# ---------------------------------------------------------------------------
# PART 1 — MixStyle alone
# ---------------------------------------------------------------------------
echo "=== [1/5] Training ${L12} (--train-only, real-world NOT read) ==="
python -m experiments.plan2_arch.run_arch --config "$CFGDIR/${L12}.yaml" --train-only

echo "=== [2/5] Training ${L123} (--train-only, real-world NOT read) ==="
python -m experiments.plan2_arch.run_arch --config "$CFGDIR/${L123}.yaml" --train-only

echo "=== [3/5] Selecting the MixStyle depth on PlantVillage VAL macro-F1 ==="
python -m experiments.plan2_arch.select_on_val "$L12" "$L123"
WINNER=$(python -m experiments.plan2_arch.select_on_val "$L12" "$L123" --print-winner)
echo "Val winner: ${WINNER}"

echo "=== [4/5] Evaluating ${WINNER} (real-world read ONCE) ==="
python -m experiments.plan2_arch.run_arch --config "$CFGDIR/${WINNER}.yaml" --eval-only
python -m experiments.plan2_arch.compare_arch --run "$WINNER"

# ---------------------------------------------------------------------------
# PART 2 — MixStyle + background randomization (combination row)
# ---------------------------------------------------------------------------
COMBO="${WINNER}_bgrand"
echo "=== [5/5] Training + evaluating the COMBINATION row ${COMBO} ==="

# rembg (U^2-Net) drives bgrand's segmentation. Masks from the earlier Plan 1 run
# are cached in data/mask_cache, so this is usually a no-op — guarded anyway.
python -c "import rembg" 2>/dev/null || pip install -r experiments/plan1_bgrand/requirements.txt || \
  echo "WARN: rembg install failed — run it on the login node, or set segmentation: hsv_threshold."

# Synthetic backgrounds must be the SAME pool the bgrand-alone row used, or the
# factorial breaks. Regenerate only if the folder is empty (idempotent).
if [ -z "$(ls -A data/backgrounds_generic_synthetic 2>/dev/null | grep -Ei '\.(jpg|jpeg|png|bmp|tif|tiff|webp)$' || true)" ]; then
  echo "No synthetic backgrounds found — generating..."
  python -m experiments.plan1_bgrand.make_synthetic_backgrounds
fi

python -m experiments.plan2_arch.run_arch --config "$CFGDIR/${COMBO}.yaml"
python -m experiments.plan2_arch.compare_arch --run "$COMBO"

echo
echo "Done. The 2x2 factorial now reads:"
echo "  efficientnetb0_on             neither"
echo "  efficientnetb0_on_bgrand      bgrand only"
echo "  ${WINNER}   mixstyle only"
echo "  ${COMBO}   both"
echo "Results in experiments/results/{${WINNER},${COMBO}}/."
