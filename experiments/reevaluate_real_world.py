"""Re-measure every evaluated row against a DIFFERENT real-world test set.

    python -m experiments.reevaluate_real_world --check   # counts only, no model, no GPU
    python -m experiments.reevaluate_real_world           # archive + re-evaluate every row
    python -m experiments.reevaluate_real_world --run efficientnetb0_on   # one row

WHY A SEPARATE SCRIPT — the naive approach silently does nothing
----------------------------------------------------------------
Editing `real_world_dir` in the YAML does NOT change evaluation. `run.py:36-41`
loads the config and then calls `evaluate_run(results_dir, device)` — passing only
the directory. `evaluate.py:30-31` does `cfg = ckpt["config"]`: evaluation reads
`real_world_dir` OUT OF THE CHECKPOINT, saved at training time. The YAML value is
used for training only.

So repointing the YAML and running `--eval-only` re-evaluates every model on the
OLD set and prints entirely plausible numbers. A silent no-op that produces
believable results is the worst failure mode available, so the new set must be
passed explicitly. That is this script.

WHAT IT PROTECTS
----------------
1. OVERWRITING. `evaluate_run` writes eval_results_real_world.json in place, so a
   re-evaluation would destroy the numbers the whole study currently reports. Here
   the old results are ARCHIVED first, into
   `experiments/results/<run>/archive_<old_set>/`. If an archive already exists the
   run REFUSES rather than overwrite it — the original can only ever be saved once.
   Inference happens BEFORE archiving, so a failure mid-run leaves everything as it
   was.

2. PROVENANCE. Nothing in eval_results_real_world.json records WHICH real-world set
   produced it. With one set that is untidy; with two it is unrecoverable
   ambiguity. Every file written here records real_world_set, real_world_dir and
   real_world_n_images.

3. PARTIAL DOWNLOADS. Ten empty folders make ImageFolder raise, which is fine. But
   three of ten populated evaluates HAPPILY, with seven classes at zero support,
   producing a catastrophic-looking macro-F1 caused purely by missing files. So
   --check counts every class and this script REFUSES an incomplete set.

The read-once discipline carries over: only rows that ALREADY have a published
eval_results_real_world.json are touched. `droppath03` and `mixstyle_l12` lost on
validation and their real-world set was never read; it will not be read on the new
set either. That exclusion is structural — those rows have no published file, so
there is nothing here to re-measure.

Controlled (PlantVillage) results are NOT touched. The gap is recomputed from the
untouched eval_results.json, so only its real-world term moves.
"""
import argparse
import json
import os
import shutil

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from experiments.common.evaluate import _load_model, _metrics, _plot_confusion, _predict
from experiments.compile_results import BACKBONE_ORDER, RESULTS_DIR, STORY_ROWS, _load
from experiments.confusion_matrices import (OUT_DIR, _real_world_loader, _resolution_of,
                                            cm_path)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = "data/real_environment_dataset"

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# The project's 10 classes (CLAUDE.md). The real-world folder names must match
# these EXACTLY: evaluate.py:117 remaps folder labels into the training label space
# BY NAME, so a mismatch is a KeyError. Listed here so --check can say precisely
# which folder is wrong, instead of leaving you to read a traceback.
EXPECTED_CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


def _abs(path):
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def _count_images(d):
    if not os.path.isdir(d):
        return 0
    return sum(1 for f in os.listdir(d)
               if os.path.splitext(f)[1].lower() in IMAGE_EXT)


def check_set(real_dir):
    """Report the new set's per-class counts. Returns True only if it is usable."""
    real_dir = _abs(real_dir)
    print(f"Checking real-world set: {real_dir}")
    if not os.path.isdir(real_dir):
        print(f"FAIL  directory does not exist.")
        return False

    found = sorted(d for d in os.listdir(real_dir)
                   if os.path.isdir(os.path.join(real_dir, d)))
    missing = [c for c in EXPECTED_CLASSES if c not in found]
    extra = [d for d in found if d not in EXPECTED_CLASSES]

    # A missing class counts 0 regardless of what the filesystem says. On a
    # case-insensitive filesystem (Windows, macOS) os.path.isdir("..._Target_Spot")
    # happily matches a folder actually named "..._Target_spot", so counting it
    # would print "12 <- FOLDER MISSING". `missing` is decided by exact string
    # comparison against os.listdir, which is right on every platform; the count
    # must agree with it.
    counts = {c: (0 if c in missing else _count_images(os.path.join(real_dir, c)))
              for c in EXPECTED_CLASSES}
    width = max(len(c) for c in EXPECTED_CLASSES)
    print()
    for c in EXPECTED_CLASSES:
        note = ""
        if c in missing:
            note = "  <- FOLDER MISSING"
        elif counts[c] == 0:
            note = "  <- EMPTY"
        print(f"  {c.ljust(width)}  {counts[c]:>5}{note}")
    total = sum(counts.values())
    print(f"  {'TOTAL'.ljust(width)}  {total:>5}")
    print()

    ok = True
    if missing:
        print(f"FAIL  {len(missing)} class folder(s) missing: {missing}")
        print("      Folder names must match the training classes EXACTLY "
              "(evaluate.py remaps by name).")
        ok = False
    if extra:
        # Not fatal for the 10 expected classes, but ImageFolder WOULD index these
        # as extra classes and shift every label. That is silent corruption.
        print(f"FAIL  unexpected folder(s) present: {extra}")
        print("      ImageFolder indexes every subfolder, so an extra one shifts the "
              "label space and silently corrupts every metric. Remove them.")
        ok = False
    empty = [c for c in EXPECTED_CLASSES if c not in missing and counts[c] == 0]
    if empty:
        print(f"FAIL  {len(empty)} class(es) have no images: {empty}")
        print("      A partially-downloaded set does NOT fail loudly: macro-F1 would be "
              "computed with these classes at zero support, and the result would look "
              "catastrophic for reasons that have nothing to do with the models.")
        ok = False

    if ok:
        print(f"Set OK: {total} images across {len(EXPECTED_CLASSES)} classes.")
        thin = [c for c in EXPECTED_CLASSES if counts[c] < 10]
        if thin:
            # Not a failure - just arithmetic worth knowing before you read the table.
            print(f"NOTE  {len(thin)} class(es) have fewer than 10 images: {thin}")
            print("      Per-class F1 on a handful of images is extremely noisy, and macro-F1 "
                  "weights every class equally regardless of size.")
    return ok


def _archive_dir(results_dir, old_set):
    return os.path.join(results_dir, f"archive_{old_set}")


def reevaluate(run, new_dir, device, dry_run=False):
    published = _load(run, "eval_results_real_world.json")
    if published is None:
        # Never evaluated => the real-world set was never read for this row, and
        # this script will not be the thing that reads it. See the module docstring.
        return None

    results_dir = os.path.join(RESULTS_DIR, run)
    model, cfg, class_to_idx = _load_model(results_dir, device)
    old_dir = cfg["real_world_dir"]
    old_set = os.path.basename(old_dir.rstrip("/\\"))
    archive = _archive_dir(results_dir, old_set)

    # Refuse rather than overwrite an existing archive: the original numbers can be
    # saved exactly once, and a careless second run must not be able to destroy them.
    if os.path.isdir(archive) and os.listdir(archive):
        raise RuntimeError(
            f"{run}: {archive} already exists and is not empty. The original results "
            f"have already been archived once; refusing to overwrite them. If you "
            f"really mean to re-run, move or delete that folder yourself first.")

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    image_size = _resolution_of(cfg)
    cfg = dict(cfg)
    cfg["real_world_dir"] = _abs(new_dir)   # THE OVERRIDE
    ds, loader = _real_world_loader(cfg, image_size)
    real_idx_to_class = {v: k for k, v in ds.class_to_idx.items()}

    # Inference FIRST, archive second, write third: a failure here must leave the
    # existing results exactly as they were.
    yt_local, yp = _predict(model, loader, device)
    yt = np.array([class_to_idx[real_idx_to_class[i]] for i in yt_local])

    real_world = _metrics(yt, yp, class_names)
    controlled = _load(run, "eval_results.json")
    real_world["model"] = run
    real_world["input_resolution"] = image_size
    if controlled:
        real_world["generalization_gap_accuracy"] = controlled["accuracy"] - real_world["accuracy"]
        real_world["generalization_gap_macro_f1"] = controlled["macro_f1"] - real_world["macro_f1"]
    # Provenance: with two real-world sets in existence, a result file that does not
    # name its source is ambiguous forever.
    real_world["real_world_set"] = os.path.basename(_abs(new_dir).rstrip("/\\"))
    real_world["real_world_dir"] = _abs(new_dir)
    real_world["real_world_n_images"] = int(len(yt))
    real_world["previous_real_world_set"] = old_set
    real_world["previous_macro_f1"] = published.get("macro_f1")

    if dry_run:
        print(f"  [dry-run] {run}: {published.get('macro_f1'):.4f} ({old_set}) -> "
              f"{real_world['macro_f1']:.4f} ({real_world['real_world_set']}) "
              f"— nothing written.", flush=True)
        return real_world

    os.makedirs(archive, exist_ok=True)
    for fname in ("eval_results_real_world.json", "cm_real_world_test.png"):
        src = os.path.join(results_dir, fname)
        if os.path.isfile(src):
            shutil.move(src, os.path.join(archive, fname))
    with open(os.path.join(archive, "README.txt"), "w") as f:
        f.write(f"Real-world results for run '{run}' measured on the RETIRED test set\n"
                f"'{old_set}' ({old_dir}).\n\n"
                f"Archived when the study switched to '{real_world['real_world_set']}'.\n"
                f"Kept because these are the numbers the earlier analysis reported, and\n"
                f"because a finding that reproduces on a second, independent field set is\n"
                f"far stronger than one measured once. Do not mix the two in one table.\n")

    with open(os.path.join(results_dir, "eval_results_real_world.json"), "w") as f:
        json.dump(real_world, f, indent=2)
    _plot_confusion(yt, yp, class_names,
                    os.path.join(results_dir, "cm_real_world_test.png"),
                    f"{run} - real-world test ({real_world['real_world_set']}, row-normalized)")

    # Also write the confusion counts, so confusion_matrices.py --report-only can
    # build every table and figure on the new set with no further inference.
    cm = confusion_matrix(yt, yp, labels=list(range(len(class_names))))
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(cm_path(run), "w") as f:
        json.dump({"run": run, "class_names": class_names, "input_resolution": image_size,
                   "counts": cm.tolist(), "macro_f1": real_world["macro_f1"],
                   "macro_f1_published": real_world["macro_f1"],
                   "real_world_set": real_world["real_world_set"],
                   "n_images": int(len(yt))}, f, indent=2)

    d = real_world["macro_f1"] - (published.get("macro_f1") or 0.0)
    print(f"  {run}: macro-F1 {published.get('macro_f1'):.4f} ({old_set}) -> "
          f"{real_world['macro_f1']:.4f} ({real_world['real_world_set']}), {d:+.4f} "
          f"on {len(yt)} images. Old results archived.", flush=True)
    return real_world


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=DEFAULT_DIR,
                        help=f"Real-world set to evaluate against (default: {DEFAULT_DIR}).")
    parser.add_argument("--check", action="store_true",
                        help="Only verify the set's class folders and image counts, then exit.")
    parser.add_argument("--run", help="Re-evaluate a single run instead of all of them.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Evaluate and report, but write and archive nothing.")
    args = parser.parse_args()

    if not check_set(args.dir):
        print("\nRefusing to evaluate: fix the set above first. Nothing was changed.")
        raise SystemExit(1)
    if args.check:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nRe-evaluating on {device}. Frozen checkpoints; no training, no selection.")
    print(f"New real-world set: {_abs(args.dir)}\n")

    if args.run:
        runs = [args.run]
    else:
        runs = ([f"{bb}_{t}" for bb in BACKBONE_ORDER for t in ("off", "on")]
                + [r for r, _ in STORY_ROWS])

    done, skipped = [], []
    for run in runs:
        if not os.path.isdir(os.path.join(RESULTS_DIR, run)):
            continue
        result = reevaluate(run, args.dir, device, dry_run=args.dry_run)
        (done if result else skipped).append(run)

    print(f"\nRe-evaluated {len(done)} run(s).")
    if skipped:
        print("Skipped (never evaluated - the real-world set was deliberately not read "
              "for them, and still is not): " + ", ".join(skipped))
    if args.dry_run:
        print("Dry run: nothing was written or archived.")
        return
    print("\nOld results are archived per run in "
          "experiments/results/<run>/archive_<old_set>/.")
    print("Next:")
    print("  python -m experiments.compile_results                    # new tables")
    print("  python -m experiments.confusion_matrices --report-only   # new matrices, no GPU")


if __name__ == "__main__":
    main()
