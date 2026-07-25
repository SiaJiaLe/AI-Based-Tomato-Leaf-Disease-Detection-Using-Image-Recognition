"""Train + evaluate EfficientViT-B0 on the tomato split — TRY-OUT ONLY.

This is a standalone exploration, NOT part of the CP2 ablation/thesis. It reuses
the study's shared code READ-ONLY so preprocessing and metrics are identical:

  * data pipeline   -> experiments.common.data  (Resize256->Crop224->ImageNet-norm,
                       the same 4 basic train augmentations; advanced block toggled
                       by the config, exactly like the study's OFF/ON stacks)
  * metrics + plots -> experiments.common.evaluate (_metrics, _plot_confusion)
  * seeding         -> experiments.common.seeding

Why not the shared engine? experiments/common/engine.py fine-tunes through a
CNN-specific stage-group interface (freeze deepest 1-2 conv stages). EfficientViT
is a transformer with no such decomposition, so this file implements a small
two-phase fine-tune of its own (head warm-up -> full fine-tune) while leaving the
study engine untouched.

Usage (from the repo root):
    python -m experiments.tryout_efficientvit.train_efficientvit \
        --config experiments/tryout_efficientvit/config_off.yaml
    python -m experiments.tryout_efficientvit.train_efficientvit \
        --config experiments/tryout_efficientvit/config_on.yaml
    # ... --eval-only to re-score an existing checkpoint.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from experiments.common.data import (AlbumentationsImageFolder, build_eval_transform,
                                      build_loaders)
from experiments.common.evaluate import _metrics, _plot_confusion
from experiments.common.seeding import seed_everything

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(REPO_ROOT, "experiments", "results")
IMAGE_SIZE = 224


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["data_dir"] = os.path.join(REPO_ROOT, cfg["data_dir"])
    cfg["real_world_dir"] = os.path.join(REPO_ROOT, cfg["real_world_dir"])
    return cfg


def build_model(timm_model: str, num_classes: int, device):
    """timm EfficientViT with a fresh classifier head sized to our classes."""
    model = timm.create_model(timm_model, pretrained=True, num_classes=num_classes)
    return model.to(device)


def _run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    running_loss, correct, total = 0.0, 0, 0
    y_true, y_pred = [], []
    torch.set_grad_enabled(train)
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        if train:
            optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        if train:
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(predicted.cpu().tolist())
    torch.set_grad_enabled(True)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return running_loss / total, correct / total, macro_f1


def train_run(cfg: dict, results_dir: str, device):
    os.makedirs(results_dir, exist_ok=True)
    tr = cfg["training"]
    seed_everything(cfg["seed"])

    train_loader, val_loader, _, class_to_idx = build_loaders(
        data_dir=cfg["data_dir"], image_size=IMAGE_SIZE, batch_size=tr["batch_size"],
        advanced_augmentation=cfg["advanced_augmentation"], seed=cfg["seed"])
    num_classes = len(class_to_idx)

    model = build_model(cfg["timm_model"], num_classes, device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    history = []

    # --- Phase A: classifier head only (backbone frozen) ---
    head = model.get_classifier()
    for p in model.parameters():
        p.requires_grad = False
    for p in head.parameters():
        p.requires_grad = True
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=tr["phase_a_lr"])
    for epoch in range(tr["phase_a_epochs"]):
        t0 = time.time()
        t_loss, t_acc, t_f1 = _run_epoch(model, train_loader, criterion, optimizer, device, True)
        v_loss, v_acc, v_f1 = _run_epoch(model, val_loader, criterion, None, device, False)
        history.append({"phase": "A", "epoch": epoch + 1, "train_loss": t_loss,
                        "train_acc": t_acc, "val_loss": v_loss, "val_acc": v_acc,
                        "val_macro_f1": v_f1})
        print(f"[A {epoch+1}/{tr['phase_a_epochs']}] train_loss {t_loss:.4f} acc {t_acc:.4f} | "
              f"val_loss {v_loss:.4f} acc {v_acc:.4f} f1 {v_f1:.4f} | {time.time()-t0:.1f}s",
              flush=True)

    # --- Phase B: full fine-tune, cosine LR, macro-F1 checkpoint + early stop ---
    for p in model.parameters():
        p.requires_grad = True
    optimizer = optim.AdamW(model.parameters(), lr=tr["phase_b_lr"],
                            weight_decay=tr["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tr["phase_b_epochs"])

    best_f1, best_loss = -1.0, float("inf")
    epochs_no_improve = 0
    for epoch in range(tr["phase_b_epochs"]):
        t0 = time.time()
        t_loss, t_acc, t_f1 = _run_epoch(model, train_loader, criterion, optimizer, device, True)
        v_loss, v_acc, v_f1 = _run_epoch(model, val_loader, criterion, None, device, False)
        scheduler.step()
        history.append({"phase": "B", "epoch": epoch + 1, "train_loss": t_loss,
                        "train_acc": t_acc, "val_loss": v_loss, "val_acc": v_acc,
                        "val_macro_f1": v_f1})
        marker = ""
        if v_f1 > best_f1:
            best_f1 = v_f1
            epochs_no_improve = 0
            torch.save({"state_dict": model.state_dict(), "class_to_idx": class_to_idx,
                        "config": cfg, "val_macro_f1": v_f1, "val_loss": v_loss},
                       os.path.join(results_dir, "best_model.pth"))
            marker += " *best_f1"
        else:
            epochs_no_improve += 1
        if v_loss < best_loss:
            best_loss = v_loss
            torch.save({"state_dict": model.state_dict(), "class_to_idx": class_to_idx,
                        "config": cfg, "val_macro_f1": v_f1, "val_loss": v_loss},
                       os.path.join(results_dir, "best_by_loss.pth"))
            marker += " *best_loss"
        print(f"[B {epoch+1}/{tr['phase_b_epochs']}] train_loss {t_loss:.4f} acc {t_acc:.4f} | "
              f"val_loss {v_loss:.4f} acc {v_acc:.4f} f1 {v_f1:.4f} | {time.time()-t0:.1f}s{marker}",
              flush=True)
        if epochs_no_improve >= tr["patience"]:
            print(f"Early stopping after {epoch+1} Phase-B epochs (no val macro-F1 improvement).",
                  flush=True)
            break

    metrics = {"run_name": cfg["run_name"], "timm_model": cfg["timm_model"],
               "tryout": True, "seed": cfg["seed"],
               "advanced_augmentation": cfg["advanced_augmentation"],
               "best_val_macro_f1": best_f1, "best_val_loss": best_loss,
               "class_to_idx": class_to_idx, "history": history}
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(results_dir, "resolved_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved best_model.pth (val macro-F1 {best_f1:.4f}) to {results_dir}", flush=True)
    return metrics


def _load_model(results_dir, device):
    ckpt = torch.load(os.path.join(results_dir, "best_model.pth"), map_location=device)
    cfg = ckpt["config"]
    class_to_idx = ckpt["class_to_idx"]
    model = build_model(cfg["timm_model"], len(class_to_idx), device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg, class_to_idx


@torch.no_grad()
def _predict(model, loader, device):
    y_true, y_pred = [], []
    for inputs, labels in loader:
        outputs = model(inputs.to(device))
        _, pred = outputs.max(1)
        y_true.extend(labels.tolist())
        y_pred.extend(pred.cpu().tolist())
    return np.array(y_true), np.array(y_pred)


def evaluate_run(cfg: dict, results_dir: str, device):
    """Score the best checkpoint on the controlled test set and the real-world
    set, using the study's shared _metrics/_plot_confusion so the numbers and
    the confusion PNGs match the other runs exactly."""
    model, cfg_ckpt, class_to_idx = _load_model(results_dir, device)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    # Controlled PlantVillage test set.
    _, _, test_loader, _ = build_loaders(
        cfg["data_dir"], IMAGE_SIZE, cfg["training"]["batch_size"],
        advanced_augmentation=False, seed=cfg["seed"])
    yt, yp = _predict(model, test_loader, device)
    controlled = _metrics(yt, yp, class_names)
    controlled["model"] = cfg["run_name"]
    _plot_confusion(yt, yp, class_names,
                    os.path.join(results_dir, "cm_controlled_test.png"),
                    f"{cfg['run_name']} (tryout) — controlled test (row-normalized)")
    with open(os.path.join(results_dir, "eval_results.json"), "w") as f:
        json.dump(controlled, f, indent=2)

    # Held-out real-world set (labels remapped into training space by name).
    real_dir = cfg["real_world_dir"]
    real_world = None
    if os.path.isdir(real_dir):
        eval_tf = build_eval_transform(IMAGE_SIZE)
        real_ds = AlbumentationsImageFolder(real_dir, transform=eval_tf)
        real_loader = DataLoader(real_ds, batch_size=cfg["training"]["batch_size"],
                                 shuffle=False, num_workers=4, pin_memory=True)
        real_idx_to_class = {v: k for k, v in real_ds.class_to_idx.items()}
        yt_local, yp = _predict(model, real_loader, device)
        yt = np.array([class_to_idx[real_idx_to_class[i]] for i in yt_local])
        real_world = _metrics(yt, yp, class_names)
        real_world["model"] = cfg["run_name"]
        real_world["generalization_gap_accuracy"] = controlled["accuracy"] - real_world["accuracy"]
        real_world["generalization_gap_macro_f1"] = controlled["macro_f1"] - real_world["macro_f1"]
        _plot_confusion(yt, yp, class_names,
                        os.path.join(results_dir, "cm_real_world_test.png"),
                        f"{cfg['run_name']} (tryout) — real-world test (row-normalized)")
        with open(os.path.join(results_dir, "eval_results_real_world.json"), "w") as f:
            json.dump(real_world, f, indent=2)
    else:
        print(f"  Real-world dir not found ({real_dir}); skipping real-world eval.", flush=True)

    print(f"  {cfg['run_name']}: controlled acc {controlled['accuracy']:.4f} f1 {controlled['macro_f1']:.4f}"
          + (f" | real-world acc {real_world['accuracy']:.4f} f1 {real_world['macro_f1']:.4f}"
             f" | gap {real_world['generalization_gap_accuracy']:+.4f}" if real_world else ""),
          flush=True)
    return controlled, real_world


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a tryout config YAML.")
    parser.add_argument("--eval-only", action="store_true", help="Skip training; only evaluate.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    cfg = load_config(args.config)
    results_dir = os.path.join(RESULTS_DIR, cfg["run_name"])
    print(f"\n===== {cfg['run_name']} (EfficientViT-B0 TRY-OUT) =====", flush=True)
    if not args.eval_only:
        train_run(cfg, results_dir, device)
    evaluate_run(cfg, results_dir, device)


if __name__ == "__main__":
    main()
