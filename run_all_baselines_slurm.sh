#!/bin/bash
#SBATCH --job-name=tomato-baselines
#SBATCH --output=logs/baselines_%j.out
#SBATCH --error=logs/baselines_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:l4:1
#SBATCH --partition=gpu-8c-l4-1g

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs

# No explicit "module load cuda" — the pip-installed torch wheel bundles
# its own CUDA runtime. Uncomment below only if `nvidia-smi` fails inside
# the job and your HPC requires an environment-modules CUDA load.
# module load cuda

source /app/application/shared/miniconda/25.5.1/etc/profile.d/conda.sh
conda activate tomato-ml

bash run_all_baselines.sh
