"""Resolution-aware evaluation for Plan 2 Tier 2.

`common.evaluate.evaluate_run` hardcodes 224 (evaluate.py:95 `build_loaders(...,
224, ...)` and :110 `build_eval_transform(224)`). A model trained at 240 would
be evaluated at 224 — a silent train/eval preprocessing mismatch, exactly what
CLAUDE.md warns degrades accuracy invisibly. So Tier 2 cannot reuse it.

Only the loader construction differs. `_load_model`, `_predict`, `_metrics` and
`_plot_confusion` are imported from `common.evaluate` and reused UNCHANGED, so
the metrics, the class-name remapping, the confusion plots and the gap
definition are computed by exactly the same code as every other row — the
yardstick is identical, only the input resolution differs.

The output files are byte-compatible with the baseline's, so `compare_arch.py`
reads them without knowing which eval path produced them.
"""
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.common.data import AlbumentationsImageFolder
from experiments.common.evaluate import (_load_model, _metrics, _plot_confusion,
                                         _predict)

from .data_res import build_eval_transform_res, build_loaders_res, default_resize_to


def _resolution_of(cfg: dict) -> int:
    return int(cfg.get("architecture_mod", {}).get("input_resolution", 224))


def evaluate_run_res(results_dir: str, device: torch.device):
    """Same contract and outputs as common.evaluate.evaluate_run, at the run's
    own input resolution."""
    model, cfg, class_to_idx = _load_model(results_dir, device)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    image_size = _resolution_of(cfg)
    resize_to = default_resize_to(image_size)
    print(f"  Evaluating at {image_size}px (Resize {resize_to} -> CenterCrop {image_size}) "
          f"— matches training.", flush=True)

    # --- PlantVillage controlled test set ---
    _, _, test_loader, _ = build_loaders_res(
        cfg["data_dir"], image_size, cfg["training"]["batch_size"],
        advanced_augmentation=False, seed=cfg["seed"])
    yt, yp = _predict(model, test_loader, device)
    controlled = _metrics(yt, yp, class_names)
    controlled["model"] = cfg["run_name"]
    controlled["input_resolution"] = image_size
    _plot_confusion(yt, yp, class_names,
                    os.path.join(results_dir, "cm_controlled_test.png"),
                    f"{cfg['run_name']} — controlled test (row-normalized)")
    with open(os.path.join(results_dir, "eval_results.json"), "w") as f:
        json.dump(controlled, f, indent=2)

    # --- Held-out real-world test set (touched once) ---
    real_dir = cfg["real_world_dir"]
    real_world = None
    if os.path.isdir(real_dir):
        eval_tf = build_eval_transform_res(image_size, resize_to)
        real_ds = AlbumentationsImageFolder(real_dir, transform=eval_tf)
        real_loader = DataLoader(real_ds, batch_size=cfg["training"]["batch_size"],
                                 shuffle=False, num_workers=4, pin_memory=True)
        real_idx_to_class = {v: k for k, v in real_ds.class_to_idx.items()}
        yt_local, yp = _predict(model, real_loader, device)
        # Remap real-world local indices into training label space by name
        # (identical to common.evaluate).
        yt = np.array([class_to_idx[real_idx_to_class[i]] for i in yt_local])
        real_world = _metrics(yt, yp, class_names)
        real_world["model"] = cfg["run_name"]
        real_world["input_resolution"] = image_size
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
