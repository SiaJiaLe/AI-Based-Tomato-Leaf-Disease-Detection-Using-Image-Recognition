"""Evaluation for a trained run on BOTH test sets.

Loads a run's canonical best_model.pth (selected on val macro-F1), then
computes on the PlantVillage test set and the held-out real-world test set:
accuracy, macro precision/recall/F1, per-class report, and confusion matrices
(raw counts + row-normalized heatmap). The generalization gap is
controlled-minus-real-world for both accuracy and macro-F1.

Real-world folder label indices are remapped into the training label space by
class name, so a different class ordering on disk cannot silently scramble the
metrics. The real-world set is read here for the first and only time.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support)

from .backbones import build_backbone
from .data import build_eval_transform, build_loaders, AlbumentationsImageFolder
from torch.utils.data import DataLoader


def _load_model(results_dir, device):
    ckpt = torch.load(os.path.join(results_dir, "best_model.pth"), map_location=device)
    cfg = ckpt["config"]
    class_to_idx = ckpt["class_to_idx"]
    num_classes = len(class_to_idx)
    built = build_backbone(cfg["backbone"], num_classes,
                           cfg["stack"]["strong_head"], cfg["stack"]["cbam"])
    built.module.to(device)
    built.warm_up(device)
    built.module.load_state_dict(ckpt["state_dict"])
    built.module.eval()
    return built.module, cfg, class_to_idx


@torch.no_grad()
def _predict(model, loader, device):
    y_true, y_pred = [], []
    for inputs, labels in loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, pred = outputs.max(1)
        y_true.extend(labels.tolist())
        y_pred.extend(pred.cpu().tolist())
    return np.array(y_true), np.array(y_pred)


def _metrics(y_true, y_pred, class_names):
    acc = float((y_true == y_pred).mean())
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, labels=list(range(len(class_names))),
                                   target_names=class_names, digits=4,
                                   output_dict=True, zero_division=0)
    return {"accuracy": acc, "macro_precision": float(p), "macro_recall": float(r),
            "macro_f1": float(f1), "weighted_f1": report["weighted avg"]["f1-score"],
            "classification_report": report}


def _plot_confusion(y_true, y_pred, class_names, out_path, title):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return cm.tolist()


def evaluate_run(results_dir, device):
    model, cfg, class_to_idx = _load_model(results_dir, device)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    # --- PlantVillage controlled test set ---
    _, _, test_loader, _ = build_loaders(
        cfg["data_dir"], 224, cfg["training"]["batch_size"],
        advanced_augmentation=False, seed=cfg["seed"])
    yt, yp = _predict(model, test_loader, device)
    controlled = _metrics(yt, yp, class_names)
    controlled["model"] = cfg["run_name"]
    _plot_confusion(yt, yp, class_names,
                    os.path.join(results_dir, "cm_controlled_test.png"),
                    f"{cfg['run_name']} — controlled test (row-normalized)")
    with open(os.path.join(results_dir, "eval_results.json"), "w") as f:
        json.dump(controlled, f, indent=2)

    # --- Held-out real-world test set (touched once) ---
    real_dir = cfg["real_world_dir"]
    real_world = None
    if os.path.isdir(real_dir):
        eval_tf = build_eval_transform(224)
        real_ds = AlbumentationsImageFolder(real_dir, transform=eval_tf)
        real_loader = DataLoader(real_ds, batch_size=cfg["training"]["batch_size"],
                                 shuffle=False, num_workers=4, pin_memory=True)
        real_idx_to_class = {v: k for k, v in real_ds.class_to_idx.items()}
        yt_local, yp = _predict(model, real_loader, device)
        # Remap real-world local indices into training label space by name.
        yt = np.array([class_to_idx[real_idx_to_class[i]] for i in yt_local])
        real_world = _metrics(yt, yp, class_names)
        real_world["model"] = cfg["run_name"]
        real_world["generalization_gap_accuracy"] = controlled["accuracy"] - real_world["accuracy"]
        real_world["generalization_gap_macro_f1"] = controlled["macro_f1"] - real_world["macro_f1"]
        _plot_confusion(yt, yp, class_names,
                        os.path.join(results_dir, "cm_real_world_test.png"),
                        f"{cfg['run_name']} — real-world test (row-normalized)")
        with open(os.path.join(results_dir, "eval_results_real_world.json"), "w") as f:
            json.dump(real_world, f, indent=2)
    else:
        print(f"  Real-world dir not found ({real_dir}); skipping real-world eval.", flush=True)

    print(f"  {cfg['run_name']}: controlled acc {controlled['accuracy']:.4f} f1 {controlled['macro_f1']:.4f}"
          + (f" | real-world acc {real_world['accuracy']:.4f} f1 {real_world['macro_f1']:.4f}"
             f" | gap {real_world['generalization_gap_accuracy']:+.4f}" if real_world else ""),
          flush=True)
    return controlled, real_world
