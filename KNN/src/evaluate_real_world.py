"""
Evaluates the already-fitted classifier (classifier.joblib) on the
held-out real-world field photo test set
(resnet34_model/data/real_environment_test/), separate from the
controlled PlantVillage-style test set train_evaluate.py reports on.

Requires train_evaluate.py to already have been run (classifier.joblib
and class_labels.json must exist). Does not refit anything.
"""
import os
import json
import joblib
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from config import OUTPUT_DIR, REAL_WORLD_DIR
from extract_features import extract_folder

MODEL_NAME = "KNN"


def main():
    if not os.path.isdir(REAL_WORLD_DIR):
        print(f"Real-world test folder not found at {REAL_WORLD_DIR}")
        print("Populate it with class subfolders (one per disease) before running this script.")
        return

    classifier_path = os.path.join(OUTPUT_DIR, "classifier.joblib")
    labels_path = os.path.join(OUTPUT_DIR, "class_labels.json")
    if not os.path.exists(classifier_path) or not os.path.exists(labels_path):
        print(f"Error: Could not find a fitted classifier at {classifier_path}")
        print("Please run train_evaluate.py first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Extracting real-world features on device: {device}")

    # idx_to_class from training time — the classifier's labels are in
    # this index space, which may not match the real-world folder's own
    # (alphabetical) ImageFolder indexing if class subfolders differ.
    with open(labels_path) as f:
        train_idx_to_class = {int(k): v for k, v in json.load(f).items()}
    train_class_to_idx = {v: k for k, v in train_idx_to_class.items()}

    X_real, y_real_local, real_class_to_idx = extract_folder(REAL_WORLD_DIR, device)
    real_idx_to_class = {v: k for k, v in real_class_to_idx.items()}

    # Remap real-world local label indices to the training label space.
    y_real = np.array([train_class_to_idx[real_idx_to_class[i]] for i in y_real_local])

    class_names = [train_idx_to_class[i] for i in range(len(train_idx_to_class))]

    print("Loading fitted classifier...")
    clf = joblib.load(classifier_path)

    print("Evaluating on real-world test set...")
    y_pred = clf.predict(X_real)

    print("\n--- Real-World Classification Report ---")
    report_text = classification_report(y_real, y_pred, target_names=class_names, digits=4)
    print(report_text)

    with open(os.path.join(OUTPUT_DIR, "classification_report_real_world.txt"), "w") as f:
        f.write(f"{MODEL_NAME} Baseline - Real-World Test Set Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(report_text)

    accuracy = float((y_real == y_pred).mean())
    report_dict = classification_report(y_real, y_pred, target_names=class_names, digits=4, output_dict=True)
    eval_results = {
        "model": MODEL_NAME,
        "accuracy": accuracy,
        "macro_f1": report_dict["macro avg"]["f1-score"],
        "weighted_f1": report_dict["weighted avg"]["f1-score"],
        "classification_report": report_dict,
    }
    results_path = os.path.join(OUTPUT_DIR, "eval_results_real_world.json")
    with open(results_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Saved eval results to {results_path}")

    cm = confusion_matrix(y_real, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - Real-World Test Set ({MODEL_NAME})')
    plt.ylabel('True Disease')
    plt.xlabel('Predicted Disease')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cm_real_world_test.png"), dpi=300)
    plt.close()

    print(f"Saved results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
