#!/bin/bash
# Runs train.py + evaluate.py (or train_evaluate.py for classical ML)
# for every baseline model folder in sequence, then compare_models.py.
#
# Assumes: dataset already prepared at resnet34_model/data/processed/,
# and requirements-baselines.txt already installed in the active
# conda/venv environment.
#
# Usage:
#   bash run_all_baselines.sh

set -uo pipefail
cd "$(dirname "$0")"

mkdir -p logs

DL_MODELS=("AlexNet" "VGG16" "MobileNetV2" "EfficientNetB0" "ResNet50")
CLASSICAL_MODELS=("KNN" "SVM" "RandomForest")

FAILED=()

run_step() {
    local model="$1"
    local script="$2"
    local log_file="logs/${model}_$(basename "$script" .py).log"

    echo "=== ${model}: python ${script} ==="
    if (cd "$model" && python "$script" 2>&1 | tee "../${log_file}"); then
        echo "=== ${model}: ${script} OK ==="
    else
        echo "=== ${model}: ${script} FAILED (see ${log_file}) ==="
        FAILED+=("${model}/${script}")
    fi
}

for model in "${DL_MODELS[@]}"; do
    run_step "$model" "src/train.py"
    run_step "$model" "src/evaluate.py"
done

for model in "${CLASSICAL_MODELS[@]}"; do
    run_step "$model" "src/train_evaluate.py"
done

echo ""
echo "=== All baselines finished. Running compare_models.py ==="
python compare_models.py

if [ "${#FAILED[@]}" -gt 0 ]; then
    echo ""
    echo "The following steps failed (check logs/ for details):"
    printf '  - %s\n' "${FAILED[@]}"
    exit 1
fi
