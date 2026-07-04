"""
Reads eval_results.json from every baseline model folder's outputs/
directory, plus the proposed ResNet34 model's evaluation report, and
prints/saves a single comparison table.

Run after training + evaluating each model:
    python AlexNet/src/train.py && python AlexNet/src/evaluate.py
    ... (repeat for each folder) ...
    python compare_models.py
"""
import json
import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# (display name, path to eval_results.json)
MODEL_RESULT_PATHS = [
    ("ResNet34 (proposed)", os.path.join(REPO_ROOT, "resnet34_model", "outputs", "evaluation_report", "eval_results.json")),
    ("AlexNet", os.path.join(REPO_ROOT, "AlexNet", "outputs", "eval_results.json")),
    ("VGG16", os.path.join(REPO_ROOT, "VGG16", "outputs", "eval_results.json")),
    ("MobileNetV2", os.path.join(REPO_ROOT, "MobileNetV2", "outputs", "eval_results.json")),
    ("EfficientNetB0", os.path.join(REPO_ROOT, "EfficientNetB0", "outputs", "eval_results.json")),
    ("ResNet50", os.path.join(REPO_ROOT, "ResNet50", "outputs", "eval_results.json")),
    ("KNN", os.path.join(REPO_ROOT, "KNN", "outputs", "eval_results.json")),
    ("SVM", os.path.join(REPO_ROOT, "SVM", "outputs", "eval_results.json")),
    ("RandomForest", os.path.join(REPO_ROOT, "RandomForest", "outputs", "eval_results.json")),
]


def load_results():
    rows = []
    for display_name, path in MODEL_RESULT_PATHS:
        if not os.path.exists(path):
            rows.append({"model": display_name, "accuracy": None, "macro_f1": None, "weighted_f1": None, "status": "not evaluated"})
            continue
        with open(path) as f:
            data = json.load(f)
        rows.append({
            "model": display_name,
            "accuracy": data.get("accuracy"),
            "macro_f1": data.get("macro_f1"),
            "weighted_f1": data.get("weighted_f1"),
            "status": "ok",
        })
    return rows


def format_table(rows):
    header = f"{'Model':<22} {'Accuracy':>10} {'Macro F1':>10} {'Weighted F1':>12}  Status"
    lines = [header, "-" * len(header)]
    for row in rows:
        acc = f"{row['accuracy']:.4f}" if row["accuracy"] is not None else "-"
        mf1 = f"{row['macro_f1']:.4f}" if row["macro_f1"] is not None else "-"
        wf1 = f"{row['weighted_f1']:.4f}" if row["weighted_f1"] is not None else "-"
        lines.append(f"{row['model']:<22} {acc:>10} {mf1:>10} {wf1:>12}  {row['status']}")
    return "\n".join(lines)


def main():
    rows = load_results()
    table = format_table(rows)
    print(table)

    output_path = os.path.join(REPO_ROOT, "model_comparison.json")
    with open(output_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved comparison data to {output_path}")


if __name__ == "__main__":
    main()
