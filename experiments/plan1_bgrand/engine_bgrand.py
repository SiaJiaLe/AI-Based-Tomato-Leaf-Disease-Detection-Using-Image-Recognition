"""Two-stage training for the Plan 1 run.

Identical to experiments.common.engine.train_run in every respect — seeding,
backbone build, warm-up, Stage A / Stage B schedule, macro-F1 checkpointing
(best_model.pth) + best_by_loss.pth, metrics.json — EXCEPT the train loader
comes from data_bgrand (background randomization prepended). The per-epoch loop
itself is reused verbatim via common.engine._run_epoch, so training behaviour
is not re-implemented.

The checkpoint stores the same cfg/class_to_idx layout, so common.evaluate.
evaluate_run consumes it unchanged.
"""
import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from experiments.common.backbones import build_backbone
from experiments.common.engine import _run_epoch
from experiments.common.seeding import seed_everything

from .data_bgrand import build_loaders_bgrand


def train_run_bgrand(cfg: dict, results_dir: str, device: torch.device) -> dict:
    os.makedirs(results_dir, exist_ok=True)
    stack = cfg["stack"]
    tr = cfg["training"]

    seed_everything(cfg["seed"])

    train_loader, val_loader, class_to_idx = build_loaders_bgrand(
        data_dir=cfg["data_dir"],
        image_size=224,
        batch_size=tr["batch_size"],
        advanced_augmentation=stack["advanced_augmentation"],
        seed=cfg["seed"],
        bg_cfg=cfg["background_randomization"],
    )
    num_classes = len(class_to_idx)

    built = build_backbone(
        name=cfg["backbone"],
        num_classes=num_classes,
        strong_head=stack["strong_head"],
        cbam=stack["cbam"],
    )
    built.module.to(device)
    built.warm_up(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=stack["label_smoothing"])
    history = []

    # --- Stage A: head only ---
    head_params = built.freeze_to_head()
    optimizer = optim.Adam(head_params, lr=tr["stage_a_lr"])
    for epoch in range(tr["stage_a_epochs"]):
        t0 = time.time()
        t_loss, t_acc, t_f1 = _run_epoch(built.module, train_loader, criterion, optimizer, device, True)
        v_loss, v_acc, v_f1 = _run_epoch(built.module, val_loader, criterion, None, device, False)
        history.append({"stage": "A", "epoch": epoch + 1, "train_loss": t_loss,
                        "train_acc": t_acc, "val_loss": v_loss, "val_acc": v_acc,
                        "val_macro_f1": v_f1})
        print(f"[A {epoch+1}/{tr['stage_a_epochs']}] "
              f"train_loss {t_loss:.4f} acc {t_acc:.4f} | "
              f"val_loss {v_loss:.4f} acc {v_acc:.4f} f1 {v_f1:.4f} | {time.time()-t0:.1f}s",
              flush=True)

    # --- Stage B: deep fine-tune, macro-F1 checkpointing + early stop ---
    param_dicts = built.configure_stage_b(two_group=(stack["stage_b"] == "two_group"))
    optimizer = optim.Adam(param_dicts)

    best_f1, best_loss = -1.0, float("inf")
    epochs_no_improve = 0
    for epoch in range(tr["stage_b_epochs"]):
        t0 = time.time()
        t_loss, t_acc, t_f1 = _run_epoch(built.module, train_loader, criterion, optimizer, device, True)
        v_loss, v_acc, v_f1 = _run_epoch(built.module, val_loader, criterion, None, device, False)
        history.append({"stage": "B", "epoch": epoch + 1, "train_loss": t_loss,
                        "train_acc": t_acc, "val_loss": v_loss, "val_acc": v_acc,
                        "val_macro_f1": v_f1})
        improved = v_f1 > best_f1
        marker = ""
        if improved:
            best_f1 = v_f1
            epochs_no_improve = 0
            torch.save({"state_dict": built.module.state_dict(),
                        "class_to_idx": class_to_idx, "config": cfg,
                        "val_macro_f1": v_f1, "val_loss": v_loss},
                       os.path.join(results_dir, "best_model.pth"))
            marker += " *best_f1"
        else:
            epochs_no_improve += 1
        if v_loss < best_loss:
            best_loss = v_loss
            torch.save({"state_dict": built.module.state_dict(),
                        "class_to_idx": class_to_idx, "config": cfg,
                        "val_macro_f1": v_f1, "val_loss": v_loss},
                       os.path.join(results_dir, "best_by_loss.pth"))
            marker += " *best_loss"
        print(f"[B {epoch+1}/{tr['stage_b_epochs']}] "
              f"train_loss {t_loss:.4f} acc {t_acc:.4f} | "
              f"val_loss {v_loss:.4f} acc {v_acc:.4f} f1 {v_f1:.4f} | "
              f"{time.time()-t0:.1f}s{marker}", flush=True)
        if epochs_no_improve >= tr["patience"]:
            print(f"Early stopping after {epoch+1} Stage-B epochs (no val macro-F1 improvement).", flush=True)
            break

    metrics = {"run_name": cfg["run_name"], "backbone": cfg["backbone"],
               "seed": cfg["seed"], "stack": stack,
               "background_randomization": cfg["background_randomization"],
               "best_val_macro_f1": best_f1, "best_val_loss": best_loss,
               "class_to_idx": class_to_idx, "history": history}
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(results_dir, "resolved_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved best_model.pth (val macro-F1 {best_f1:.4f}) and metrics.json to {results_dir}", flush=True)
    return metrics
