import os
import json
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from config import OUTPUT_DIR
from dataset import get_dataloaders
from model import TomatoResNet34

def evaluate_model(model, dataloader, device, class_names):
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("Evaluating model on test dataset...")
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return np.array(all_labels), np.array(all_preds)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load test dataloader
    _, _, test_loader, class_to_idx = get_dataloaders()
    num_classes = len(class_to_idx)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(num_classes)]
    
    # Initialize model
    print("Loading best model weights...")
    model = TomatoResNet34(num_classes).to(device)
    
    # Load weights
    weights_path = os.path.join(OUTPUT_DIR, "best_model.pth")
    if not os.path.exists(weights_path):
        print(f"Error: Could not find model weights at {weights_path}")
        print("Please ensure training is complete and best_model.pth exists.")
        return
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    
    # Run evaluation
    y_true, y_pred = evaluate_model(model, test_loader, device, class_names)
    
    # Generate Classification Report
    print("\n--- Classification Report ---")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)
    
    # Save Classification Report
    report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("Tomato Leaf Disease Detection - Final Evaluation Report\n")
        f.write("="*60 + "\n\n")
        f.write(report)
    print(f"Saved classification report to {report_path}")

    # Save eval_results.json — same schema as the baseline models'
    # evaluate.py, so compare_models.py can read this model's results
    # alongside the 8 baselines.
    accuracy = float((y_true == y_pred).mean())
    report_dict = classification_report(y_true, y_pred, target_names=class_names, digits=4, output_dict=True)
    eval_results = {
        "model": "ResNet34 (proposed)",
        "accuracy": accuracy,
        "macro_f1": report_dict["macro avg"]["f1-score"],
        "weighted_f1": report_dict["weighted avg"]["f1-score"],
        "classification_report": report_dict,
    }
    evaluation_report_dir = os.path.join(OUTPUT_DIR, "evaluation_report")
    os.makedirs(evaluation_report_dir, exist_ok=True)
    results_path = os.path.join(evaluation_report_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Saved eval results to {results_path}")

    # Generate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Test Set')
    plt.ylabel('True Disease')
    plt.xlabel('Predicted Disease')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save Confusion Matrix
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to {cm_path}")

if __name__ == "__main__":
    main()
