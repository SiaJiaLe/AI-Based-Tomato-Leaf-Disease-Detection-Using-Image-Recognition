"""Two-stage training for Plan 2 architecture rows.

Identical to `common.engine.train_run` except the backbone comes from
`backbones_droppath.build_arch_backbone` (so `drop_path_rate` can be passed).
`_run_epoch`, `build_loaders`, and `seed_everything` are imported from common
and reused unchanged, so the training loop, data, split, seed, and budget are
bit-for-bit the baseline's — the only difference is stochastic depth.

The checkpoint layout matches the baseline's exactly, so
`common.evaluate.evaluate_run` works on these runs verbatim.
"""
import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from experiments.common.engine import _run_epoch
from experiments.common.seeding import seed_everything

from .backbones_droppath import assert_eval_compatible, build_arch_backbone
from .data_res import build_loaders_res


def train_run_arch(cfg: dict, results_dir: str, device: torch.device) -> dict:
    os.makedirs(results_dir, exist_ok=True)
    stack = cfg["stack"]
    tr = cfg["training"]
    mod = cfg.get("architecture_mod", {})

    seed_everything(cfg["seed"])

    # Tier 2's one variable. Defaults to the baseline's 224, and at 224
    # build_loaders_res is identical to common.build_loaders (224/0.875 == 256),
    # so Tier 1 rows reproduce bit-for-bit through this path.
    image_size = int(mod.get("input_resolution", 224))

    train_loader, val_loader, _, class_to_idx = build_loaders_res(
        data_dir=cfg["data_dir"],
        image_size=image_size,
        batch_size=tr["batch_size"],
        advanced_augmentation=stack["advanced_augmentation"],
        seed=cfg["seed"],
    )
    num_classes = len(class_to_idx)

    # Fail in seconds, not after an hour, if drop-path broke eval compatibility.
    if "drop_path_rate" in mod:
        assert_eval_compatible(num_classes, stack["strong_head"], stack["cbam"],
                               float(mod["drop_path_rate"]))

    built = build_arch_backbone(cfg, num_classes)
    built.module.to(device)
    built.warm_up(device)  # materialize lazy CBAM params before optimizer
    if image_size != 224:
        # warm_up probes at 224; prove the real resolution forwards too, before
        # the optimizer is built and an hour of GPU time is spent.
        built.module.eval()
        with torch.no_grad():
            out = built.module(torch.zeros(2, 3, image_size, image_size, device=device))
        print(f"Forward OK at {image_size}px -> logits {tuple(out.shape)}.", flush=True)

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
               "architecture_mod": mod,
               "best_val_macro_f1": best_f1, "best_val_loss": best_loss,
               "class_to_idx": class_to_idx, "history": history}
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(results_dir, "resolved_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved best_model.pth (val macro-F1 {best_f1:.4f}) and metrics.json to {results_dir}", flush=True)
    return metrics
