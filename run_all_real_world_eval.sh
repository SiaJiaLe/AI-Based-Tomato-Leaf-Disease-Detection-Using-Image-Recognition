#!/bin/bash
# Runs evaluate_real_world.py for all 9 models (proposed ResNet34 + 8
# baselines) against the held-out real-world field photo test set, then
# compare_models.py to produce the generalization-gap comparison table.
#
# Assumes: each model has already been trained/evaluated on the
# controlled processed/ test set (best_model.pth or classifier.joblib
# must already exist per model), and
# resnet34_model/data/real_environment_test/ is populated with one
# subfolder per disease class.
#
# Usage:
#   bash run_all_real_world_eval.sh

set -uo pipefail
cd "$(dirname "$0")"

mkdir -p logs

MODELS=("resnet34_model" "AlexNet" "VGG16" "MobileNetV2" "EfficientNetB0" "ResNet50" "KNN" "SVM" "RandomForest")

FAILED=()

for model in "${MODELS[@]}"; do
    log_file="logs/${model}_evaluate_real_world.log"
    echo "=== ${model}: python src/evaluate_real_world.py ==="
    if (cd "$model" && python src/evaluate_real_world.py 2>&1 | tee "../${log_file}"); then
        echo "=== ${model}: evaluate_real_world.py OK ==="
    else
        echo "=== ${model}: evaluate_real_world.py FAILED (see ${log_file}) ==="
        FAILED+=("${model}")
    fi
done

echo ""
echo "=== All real-world evaluations finished. Running compare_models.py ==="
python compare_models.py

if [ "${#FAILED[@]}" -gt 0 ]; then
    echo ""
    echo "The following models failed real-world evaluation (check logs/ for details):"
    printf '  - %s\n' "${FAILED[@]}"
    exit 1
fi
