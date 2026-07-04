"""
Evaluates this already-trained model on the held-out real-world field
photo test set (resnet34_model/data/real_environment_test/), separate
from the controlled PlantVillage-style test set evaluate.py reports on.

Requires training to already be complete (best_model.pth must exist).
Does not retrain anything.
"""
import os
import json
import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import OUTPUT_DIR, REAL_WORLD_DIR, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, BATCH_SIZE
from model import build_model

MODEL_NAME = "VGG16"


class AlbumentationsDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            image = np.array(sample)
            sample = self.transform(image=image)['image']
        return sample, target


def _eval_transform():
    return A.Compose([
        A.Resize(height=256, width=256),
        A.CenterCrop(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def main():
    if not os.path.isdir(REAL_WORLD_DIR):
        print(f"Real-world test folder not found at {REAL_WORLD_DIR}")
        print("Populate it with class subfolders (one per disease) before running this script.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = AlbumentationsDataset(REAL_WORLD_DIR, transform=_eval_transform())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    num_classes = len(dataset.class_to_idx)
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(num_classes)]
    print(f"Loaded {len(dataset)} real-world images across {num_classes} classes.")

    weights_path = os.path.join(OUTPUT_DIR, "best_model.pth")
    if not os.path.exists(weights_path):
        print(f"Error: Could not find model weights at {weights_path}")
        print("Please ensure training is complete and best_model.pth exists.")
        return

    print("Loading best model weights...")
    model = build_model(num_classes).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    print("Evaluating on real-world test set...")
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_true, y_pred = np.array(all_labels), np.array(all_preds)

    print("\n--- Real-World Classification Report ---")
    report_text = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report_text)

    report_path = os.path.join(OUTPUT_DIR, "classification_report_real_world.txt")
    with open(report_path, "w") as f:
        f.write(f"{MODEL_NAME} Baseline - Real-World Test Set Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(report_text)
    print(f"Saved classification report to {report_path}")

    accuracy = float((y_true == y_pred).mean())
    report_dict = classification_report(y_true, y_pred, target_names=class_names, digits=4, output_dict=True)
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

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - Real-World Test Set ({MODEL_NAME})')
    plt.ylabel('True Disease')
    plt.xlabel('Predicted Disease')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    cm_path = os.path.join(OUTPUT_DIR, "cm_real_world_test.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to {cm_path}")


if __name__ == "__main__":
    main()
