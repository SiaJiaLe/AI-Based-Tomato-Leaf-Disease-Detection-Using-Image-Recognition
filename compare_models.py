"""
Reads eval_results.json (controlled PlantVillage-style test set) and
eval_results_real_world.json (held-out real-world field photo test set,
see real_world_generalization_plan.md) from every model folder's
outputs/, plus the proposed ResNet34 model's evaluation report, and
prints/saves a single comparison table including the generalization
gap between the two test sets.

Run after training + evaluating each model:
    python AlexNet/src/train.py && python AlexNet/src/evaluate.py
    python AlexNet/src/evaluate_real_world.py   # optional, once real_environment_test/ is populated
    ... (repeat for each folder) ...
    python compare_models.py
"""
import json
import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

RESNET34_OUTPUT_DIR = os.path.join(REPO_ROOT, "resnet34_model", "outputs", "evaluation_report")

# (display name, controlled-test eval_results.json, real-world eval_results_real_world.json)
MODEL_RESULT_PATHS = [
    ("ResNet34 (proposed)",
     os.path.join(RESNET34_OUTPUT_DIR, "eval_results.json"),
     os.path.join(RESNET34_OUTPUT_DIR, "eval_results_real_world.json")),
    ("AlexNet",
     os.path.join(REPO_ROOT, "AlexNet", "outputs", "eval_results.json"),
     os.path.join(REPO_ROOT, "AlexNet", "outputs", "eval_results_real_world.json")),
    ("VGG16",
     os.path.join(REPO_ROOT, "VGG16", "outputs", "eval_results.json"),
     os.path.join(REPO_ROOT, "VGG16", "outputs", "eval_results_real_world.json")),
    ("MobileNetV2",
     os.path.join(REPO_ROOT, "MobileNetV2", "outputs", "eval_results.json"),
     os.path.join(REPO_ROOT, "MobileNetV2", "outputs", "eval_results_real_world.json")),
    ("EfficientNetB0",
     os.path.join(REPO_ROOT, "EfficientNetB0", "outputs", "eval_results.json"),
     os.path.join(REPO_ROOT, "EfficientNetB0", "outputs", "eval_results_real_world.json")),
    ("ResNet50",
     os.path.join(REPO_ROOT, "ResNet50", "outputs", "eval_results.json"),
     os.path.join(REPO_ROOT, "ResNet50", "outputs", "eval_results_real_world.json")),
    ("KNN",
     os.path.join(REPO_ROOT, "KNN", "outputs", "eval_results.json"),
     os.path.join(REPO_ROOT, "KNN", "outputs", "eval_results_real_world.json")),
    ("SVM",
     os.path.join(REPO_ROOT, "SVM", "outputs", "eval_results.json"),
     os.path.join(REPO_ROOT, "SVM", "outputs", "eval_results_real_world.json")),
    ("RandomForest",
     os.path.join(REPO_ROOT, "RandomForest", "outputs", "eval_results.json"),
     os.path.join(REPO_ROOT, "RandomForest", "outputs", "eval_results_real_world.json")),
]


def _load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_results():
    rows = []
    for display_name, controlled_path, real_world_path in MODEL_RESULT_PATHS:
        controlled = _load_json(controlled_path)
        real_world = _load_json(real_world_path)

        controlled_acc = controlled.get("accuracy") if controlled else None
        real_world_acc = real_world.get("accuracy") if real_world else None

        gap = None
        if controlled_acc is not None and real_world_acc is not None:
            gap = controlled_acc - real_world_acc

        if controlled is None:
            status = "not evaluated"
        elif real_world is None:
            status = "no real-world data"
        else:
            status = "ok"

        rows.append({
            "model": display_name,
            "accuracy": controlled_acc,
            "macro_f1": controlled.get("macro_f1") if controlled else None,
            "weighted_f1": controlled.get("weighted_f1") if controlled else None,
            "real_world_accuracy": real_world_acc,
            "real_world_macro_f1": real_world.get("macro_f1") if real_world else None,
            "generalization_gap": gap,
            "status": status,
        })
    return rows


def format_table(rows):
    header = f"{'Model':<22} {'Acc':>8} {'MacroF1':>8}  {'RealWorldAcc':>13} {'RealWorldF1':>12}  {'Gap':>8}  Status"
    lines = [header, "-" * len(header)]
    for row in rows:
        acc = f"{row['accuracy']:.4f}" if row["accuracy"] is not None else "-"
        mf1 = f"{row['macro_f1']:.4f}" if row["macro_f1"] is not None else "-"
        rw_acc = f"{row['real_world_accuracy']:.4f}" if row["real_world_accuracy"] is not None else "-"
        rw_f1 = f"{row['real_world_macro_f1']:.4f}" if row["real_world_macro_f1"] is not None else "-"
        gap = f"{row['generalization_gap']:+.4f}" if row["generalization_gap"] is not None else "-"
        lines.append(f"{row['model']:<22} {acc:>8} {mf1:>8}  {rw_acc:>13} {rw_f1:>12}  {gap:>8}  {row['status']}")
    lines.append("")
    lines.append("Gap = controlled-test accuracy - real-world-test accuracy (higher gap = worse field generalization)")
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
