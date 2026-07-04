#!/bin/bash
#SBATCH --job-name=tomato-baselines
#SBATCH --output=logs/baselines_%j.out
#SBATCH --error=logs/baselines_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
# TODO: set partition for Sunway HPC (see resnet34_model/scripts/hpc/README.md)
#SBATCH --partition=gpu

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs

module load cuda
source ~/miniconda3/etc/profile.d/conda.sh   # TODO: adjust to your conda install path
conda activate tomato-ml                     # TODO: set to your HPC conda env name

bash run_all_baselines.sh
