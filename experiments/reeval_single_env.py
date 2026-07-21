"""Re-score every evaluated row's REAL-WORLD set on the current device, in ONE
environment, so the published macro-F1 and its confusion matrix come from the SAME
forward pass and reproduce by construction.

    python -m experiments.reeval_single_env               # re-score every evaluated row
    python -m experiments.reeval_single_env --dry-run      # infer + print old->new, write nothing
    python -m experiments.reeval_single_env --run <name>   # re-score one row only

WHY THIS EXISTS
---------------
The study ran across multiple Colab sessions on different GPUs (early runs on T4, the
final post-processing on an A100). cuDNN picks a different conv algorithm per device
(CUDNN_STATUS_NOT_SUPPORTED fallback on the A100), so a checkpoint scored on a T4 does
NOT re-infer bit-identically on the A100. `confusion_matrices.py` verifies each row
reproduces its published macro-F1 to 1e-6 and HARD-FAILS otherwise -- which is exactly
what it should do -- so `resnet50_on` (scored 0.232497 on a T4, re-inferred 0.232755 on
the A100) aborted the confusion report.

This is NOT a data problem: `resnet50_off` reproduced exactly on the same 261-image set,
and the split is deterministic (seed 42). It is pure cross-GPU float non-associativity.

THE FIX
-------
Re-score EVERY evaluated row on the current (A100) device and rewrite its published
numbers from that pass. This writes, from ONE forward pass per row, both the row's
`eval_results_real_world.json` AND its confusion counts (confusion/<run>_cm_real_world.json)
with `macro_f1 == macro_f1_published`, so `confusion_matrices --report-only` rebuilds all
matrices from those counts with no second inference and nothing to reconcile.

IS THIS A SECOND READ OF THE TEST SET / RE-SELECTION?
-----------------------------------------------------
No. Nothing is trained, tuned, or selected. Frozen checkpoints, deterministic inference.
ONLY rows that ALREADY have a published `eval_results_real_world.json` are touched -- the
same read-once boundary `confusion_matrices` respects. `droppath03` and `mixstyle_l123`
lost on validation and were never evaluated, so they have no such file and are skipped.

SAFETY
------
Inference runs FIRST; only after it succeeds are the old files backed up (into
`experiments/results/<run>/backup_pre_a100_<ts>/`) and overwritten. A mid-run failure
therefore leaves every file exactly as it was. Each rewritten JSON records provenance
(`scored_on`, the device, `real_world_n_images`, and `superseded_macro_f1` = the pre-A100
value), so the change is fully traceable. Controlled (PlantVillage) results are NOT
touched; the generalization gap is recomputed from the untouched `eval_results.json`, so
only the real-world term moves.

ISOLATION: additive. Imports from common/evaluate, confusion_matrices, compile_results,
and plan1_bgrand/compare_seeds; modifies none of them.
"""
import argparse
import json
import os
import shutil
import time

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from experiments.common.evaluate import (_load_model, _metrics, _plot_confusion,
                                         _predict)
from experiments.confusion_matrices import (OUT_DIR, _resolution_of,
                                            _real_world_loader, cm_path)
from experiments.compile_results import (BACKBONE_ORDER, RESULTS_DIR, STORY_ROWS,
                                         _load)
from experiments.plan1_bgrand.compare_seeds import MATRIX as SEEDREP_MATRIX


def _all_runs():
    """Every row this study evaluated, in a stable order: the 12 ablation rows, then
    the EfficientNetB0 story rows, then the 7 seedrep rows -- de-duplicated (seed-42
    synthetic/real appear in both the story and the seedrep matrix)."""
    runs = [f"{bb}_{t}" for bb in BACKBONE_ORDER for t in ("off", "on")]
    runs += [r for r, _ in STORY_ROWS]
    for recipe in SEEDREP_MATRIX.values():
        runs += list(recipe.values())
    seen, ordered = set(), []
    for r in runs:
        if r not in seen:
            seen.add(r)
            ordered.append(r)
    return ordered


def _score_one(run, device):
    """Run the real-world forward pass for one frozen checkpoint on `device`.

    Returns a dict of everything the caller needs to write, or None if the row was
    never evaluated (no published real-world file -> read-once: this script will not
    be the thing that reads its test set)."""
    published = _load(run, "eval_results_real_world.json")
    if published is None:
        return None

    results_dir = os.path.join(RESULTS_DIR, run)
    model, cfg, class_to_idx = _load_model(results_dir, device)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    image_size = _resolution_of(cfg)
    ds, loader = _real_world_loader(cfg, image_size)
    real_idx_to_class = {v: k for k, v in ds.class_to_idx.items()}

    yt_local, yp = _predict(model, loader, device)
    # Remap real-world folder indices into the training label space BY NAME -
    # identical to common.evaluate:117, so a different on-disk class ordering cannot
    # silently scramble the metrics or the matrix.
    yt = np.array([class_to_idx[real_idx_to_class[i]] for i in yt_local])

    real_world = _metrics(yt, yp, class_names)
    real_world["model"] = cfg["run_name"]

    # Recompute the gap from the UNTOUCHED controlled results, mirroring evaluate_run.
    controlled = _load(run, "eval_results.json")
    if controlled is not None:
        real_world["generalization_gap_accuracy"] = (
            controlled["accuracy"] - real_world["accuracy"])
        real_world["generalization_gap_macro_f1"] = (
            controlled["macro_f1"] - real_world["macro_f1"])

    return {"yt": yt, "yp": yp, "real_world": real_world, "class_names": class_names,
            "image_size": image_size, "run_name": cfg["run_name"],
            "old_f1": float(published["macro_f1"])}


def _backup(run, stamp, paths):
    """Copy each existing file in `paths` into results/<run>/backup_pre_a100_<stamp>/."""
    backup_dir = os.path.join(RESULTS_DIR, run, f"backup_pre_a100_{stamp}")
    os.makedirs(backup_dir, exist_ok=True)
    for p in paths:
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(backup_dir, os.path.basename(p)))
    return backup_dir


def reeval(run, device, stamp, dry_run=False):
    scored = _score_one(run, device)
    if scored is None:
        print(f"  SKIP  {run}: never evaluated (no published real-world file).", flush=True)
        return "skipped"

    yt, yp = scored["yt"], scored["yp"]
    real_world = scored["real_world"]
    class_names = scored["class_names"]
    image_size = scored["image_size"]
    new_f1 = real_world["macro_f1"]
    old_f1 = scored["old_f1"]
    n_images = int(len(yt))
    print(f"  {run}: macro-F1 {old_f1:.6f} -> {new_f1:.6f} ({new_f1 - old_f1:+.6f}) "
          f"at {image_size}px, {n_images} images.", flush=True)

    if dry_run:
        return "dry-run"

    results_dir = os.path.join(RESULTS_DIR, run)
    rw_json = os.path.join(results_dir, "eval_results_real_world.json")
    rw_png = os.path.join(results_dir, "cm_real_world_test.png")
    cm_json = cm_path(run)
    # Inference already succeeded above; only now do we disturb anything on disk.
    _backup(run, stamp, [rw_json, rw_png, cm_json])

    # Provenance: make the re-score fully traceable and never silently mistaken for
    # an original number.
    real_world["scored_on"] = "a100_single_env"
    real_world["scored_device"] = str(device)
    real_world["real_world_n_images"] = n_images
    real_world["superseded_macro_f1"] = old_f1
    with open(rw_json, "w", encoding="utf-8") as f:
        json.dump(real_world, f, indent=2)

    # Redraw the row's own real-world confusion PNG from the same predictions.
    _plot_confusion(yt, yp, class_names, rw_png,
                    f"{scored['run_name']} - real-world test (row-normalized)")

    # Write the confusion COUNTS in the exact shape confusion_matrices._read_cm_json
    # expects, so `confusion_matrices --report-only` rebuilds every matrix from these
    # numbers with no re-inference. macro_f1 == macro_f1_published by construction now.
    cm = confusion_matrix(yt, yp, labels=list(range(len(class_names))))
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(cm_json, "w", encoding="utf-8") as f:
        json.dump({"run": run, "class_names": class_names,
                   "input_resolution": image_size, "counts": cm.tolist(),
                   "macro_f1": new_f1, "macro_f1_published": new_f1,
                   "n_images": n_images}, f, indent=2)

    return "rewritten"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Infer and print old->new per row; write nothing.")
    parser.add_argument("--run", default=None,
                        help="Re-score just this one run name.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Re-scoring real-world predictions on {device}. Frozen checkpoints, "
          f"deterministic. Each row's published macro-F1 is rewritten from THIS pass so "
          f"the whole table is single-environment.\n", flush=True)
    if device.type != "cuda":
        print("  WARNING: no CUDA device found. The point of this pass is to score every "
              "row on ONE GPU; running on CPU will not fix the cross-GPU drift.\n", flush=True)

    runs = [args.run] if args.run else _all_runs()
    stamp = time.strftime("%Y%m%d_%H%M%S")

    counts = {"rewritten": 0, "skipped": 0, "dry-run": 0, "missing-dir": 0}
    for run in runs:
        if not os.path.isdir(os.path.join(RESULTS_DIR, run)):
            print(f"  MISS  {run}: no run directory.", flush=True)
            counts["missing-dir"] += 1
            continue
        status = reeval(run, device, stamp, dry_run=args.dry_run)
        counts[status] = counts.get(status, 0) + 1

    print("\nDone. " + ", ".join(f"{k}={v}" for k, v in counts.items() if v))
    if not args.dry_run and counts.get("rewritten"):
        print("\nNow rebuild the tables (no further inference):")
        print("  python -m experiments.compile_results")
        print("  python -m experiments.confusion_matrices --report-only")
        print("  python -m experiments.plan1_bgrand.compare_seeds")


if __name__ == "__main__":
    main()
