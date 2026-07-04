import os
import json
import joblib
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

from config import OUTPUT_DIR
from extract_features import extract_split

MODEL_NAME = "SVM"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Extracting features on device: {device}")

    print("Extracting train features (frozen pretrained ResNet34 backbone)...")
    X_train, y_train, class_to_idx = extract_split("train", device)
    print(f"Extracting test features...")
    X_test, y_test, _ = extract_split("test", device)

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    # scikit-learn standard defaults: RBF kernel, C=1.0, gamma='scale' —
    # not tuned to match the proposed ResNet34's hyperparameters.
    clf = SVC(kernel="rbf", C=1.0, gamma="scale")
    print("Fitting SVC(kernel='rbf', C=1.0, gamma='scale')...")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Persist the fitted classifier + label mapping so
    # evaluate_real_world.py can reuse it without refitting.
    joblib.dump(clf, os.path.join(OUTPUT_DIR, "classifier.joblib"))
    with open(os.path.join(OUTPUT_DIR, "class_labels.json"), "w") as f:
        json.dump(idx_to_class, f, indent=2)

    report_text = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    print(report_text)
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write(f"{MODEL_NAME} Baseline - Test Set Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(report_text)

    accuracy = float((y_test == y_pred).mean())
    report_dict = classification_report(y_test, y_pred, target_names=class_names, digits=4, output_dict=True)
    eval_results = {
        "model": MODEL_NAME,
        "accuracy": accuracy,
        "macro_f1": report_dict["macro avg"]["f1-score"],
        "weighted_f1": report_dict["weighted avg"]["f1-score"],
        "classification_report": report_dict,
    }
    with open(os.path.join(OUTPUT_DIR, "eval_results.json"), "w") as f:
        json.dump(eval_results, f, indent=2)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - Test Set ({MODEL_NAME})')
    plt.ylabel('True Disease')
    plt.xlabel('Predicted Disease')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cm_processed_test.png"), dpi=300)
    plt.close()

    print(f"Saved results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
