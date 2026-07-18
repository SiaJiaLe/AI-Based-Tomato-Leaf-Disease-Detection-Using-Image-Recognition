#!/bin/bash
#SBATCH --job-name=ablation-array
#SBATCH --output=experiments/results/ablation_%A_%a.out
#SBATCH --error=experiments/results/ablation_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:l4:1
#SBATCH --partition=gpu-8c-l4-1g
#SBATCH --array=0-11

# Trains + evaluates the 12 ablation runs IN PARALLEL, one config per array task
# (the sequential run_all_slurm.sh does all 12 in one job; this fans them out).
# Each task trains AND evaluates on both the controlled and real-world sets, so each
# run folder ends with eval_results*.json (metrics + per-class classification_report)
# and cm_controlled_test.png / cm_real_world_test.png (the confusion matrices).
#
# Submit from the repo root:  sbatch experiments/run_ablation_array_slurm.sh
# (retrain_all.sh submits this for you.)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p experiments/results

eval "$(conda shell.bash hook)"
conda activate tomato-ml

# One config per array index; order fixed (keep 0-11 in sync with this list).
CFGS=(
  experiments/configs/alexnet_off.yaml
  experiments/configs/alexnet_on.yaml
  experiments/configs/vgg16_off.yaml
  experiments/configs/vgg16_on.yaml
  experiments/configs/resnet34_off.yaml
  experiments/configs/resnet34_on.yaml
  experiments/configs/resnet50_off.yaml
  experiments/configs/resnet50_on.yaml
  experiments/configs/mobilenetv2_off.yaml
  experiments/configs/mobilenetv2_on.yaml
  experiments/configs/efficientnetb0_off.yaml
  experiments/configs/efficientnetb0_on.yaml
)
CFG="${CFGS[$SLURM_ARRAY_TASK_ID]}"
echo "=== ablation task $SLURM_ARRAY_TASK_ID -> $CFG ==="

python -m experiments.preflight_env
python -m experiments.run --config "$CFG"
echo "=== ablation task $SLURM_ARRAY_TASK_ID done ==="
