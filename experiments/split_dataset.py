"""Split repo-root data/raw into data/processed train/val/test (for the experiments/ pipeline).

WHY THIS EXISTS AND NOT resnet34_model/scripts/prepare_dataset.py: that script's
BASE_DIR is resnet34_model/, so it reads/writes resnet34_model/data/*. The experiments/
pipeline (run.py, run_bgrand.py, run_arch.py) resolves data_dir against the REPO ROOT
-> repo-root/data/processed. Running prepare_dataset.py would leave repo-root
data/processed empty and every training job would fail. This splitter targets the
repo-root tree the experiments actually use.

Same behaviour as prepare_dataset otherwise: 70/15/15, seed 42, SYMLINKS (0 extra
bytes), and it cleans ONLY train/val/test so data/real_environment_dataset (the
real-world test set) and anything else under data/ is never touched.

    python -m experiments.split_dataset
    python -m experiments.split_dataset --seed 42
"""
import argparse
import os
import random
import shutil

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
SPLITS = ("train", "val", "test")
TRAIN_RATIO, VAL_RATIO = 0.70, 0.15   # test = remainder (0.15)


def _images(d):
    return sorted(f for f in os.listdir(d)
                  if os.path.splitext(f)[1].lower() in IMAGE_EXT)


def _existing_split_matches(processed_dir, expected):
    """True only if train/val/test all already hold EXACTLY `expected` class folders
    and are non-empty. A match means re-splitting would reproduce the same class set,
    so it is safe to skip. A mismatch (e.g. an old 10-class split when we now expect 8)
    must NOT be skipped - that is the stale-split trap."""
    expected = set(expected)
    for sp in SPLITS:
        d = os.path.join(processed_dir, sp)
        if not os.path.isdir(d):
            return False, f"{sp}/ does not exist"
        found = set(x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x)))
        if found != expected:
            extra = sorted(found - expected)
            missing = sorted(expected - found)
            return False, f"{sp}/ class set differs (extra={extra}, missing={missing})"
    total = sum(len(os.listdir(os.path.join(processed_dir, sp, c)))
                for sp in SPLITS for c in expected)
    if total == 0:
        return False, "split dirs exist but hold no images"
    return True, ""


def _existing_counts(processed_dir, classes):
    return {c: {sp: len(os.listdir(os.path.join(processed_dir, sp, c))) for sp in SPLITS}
            for c in classes}


def split(raw_dir, processed_dir, seed=42, exclude=None, skip_if_exists=False):
    """Symlink every raw image into processed/{train,val,test}/<class>/. Returns counts.

    `exclude` is a list of class folder names to OMIT from the split (their raw images
    are left on disk, just not linked in), so a class can be turned off from training
    without deleting anything.

    `skip_if_exists` returns the existing split WITHOUT rebuilding, but ONLY when
    train/val/test already hold exactly the expected class set (raw minus exclude) and
    are non-empty. A mismatch always re-splits, so a stale split for the wrong classes
    can never be silently reused.
    """
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"raw dir does not exist: {raw_dir}")
    exclude = set(exclude or [])
    all_dirs = sorted(d for d in os.listdir(raw_dir)
                      if os.path.isdir(os.path.join(raw_dir, d)))
    classes = [d for d in all_dirs if d not in exclude]
    skipped = [d for d in all_dirs if d in exclude]
    if skipped:
        print(f"Excluding {len(skipped)} class(es) from the split (raw images kept, not linked): {skipped}")
    missing_excl = exclude - set(all_dirs)
    if missing_excl:
        print(f"WARNING: --exclude names not found in raw dir (ignored): {sorted(missing_excl)}")
    if not classes:
        raise FileNotFoundError(f"No class folders to split in {raw_dir} (after exclusions).")

    if skip_if_exists:
        ok, reason = _existing_split_matches(processed_dir, classes)
        if ok:
            print(f"Existing split in {processed_dir} already has the {len(classes)} expected "
                  f"class(es); skipping re-split (use no --skip-if-exists to force).")
            return _existing_counts(processed_dir, classes)
        print(f"Not skipping - existing split unusable: {reason}. Re-splitting.")

    # Clean ONLY the three splits - never touch real_environment_dataset or siblings.
    for sp in SPLITS:
        d = os.path.join(processed_dir, sp)
        if os.path.isdir(d):
            print(f"Cleaning old {sp}/ ...")
            shutil.rmtree(d)
    for sp in SPLITS:
        for c in classes:
            os.makedirs(os.path.join(processed_dir, sp, c), exist_ok=True)

    rng = random.Random(seed)
    counts, total = {}, 0
    empty = []
    print("\n--- Per-class split (train/val/test) ---")
    for c in classes:
        imgs = _images(os.path.join(raw_dir, c))
        if not imgs:
            empty.append(c)
        rng.shuffle(imgs)
        n = len(imgs)
        n_tr = int(n * TRAIN_RATIO)
        n_va = int(n * VAL_RATIO)
        parts = {"train": imgs[:n_tr], "val": imgs[n_tr:n_tr + n_va], "test": imgs[n_tr + n_va:]}
        for sp, files in parts.items():
            dst_dir = os.path.join(processed_dir, sp, c)
            for f in files:
                src = os.path.abspath(os.path.join(raw_dir, c, f))
                os.symlink(src, os.path.join(dst_dir, f))
        counts[c] = {sp: len(files) for sp, files in parts.items()}
        total += n
        print(f"  {c:48} {n:>6}  ->  train {len(parts['train']):>5}  "
              f"val {len(parts['val']):>4}  test {len(parts['test']):>4}")

    print(f"\n  {'TOTAL':48} {total:>6}")
    if empty:
        print(f"\nWARNING: {len(empty)} class(es) have NO images: {empty}")
        print("  ImageFolder will raise on an empty class, or train with zero support - check the raw data.")
    print(f"\nDone. Symlinked {total} images into {processed_dir}/{{train,val,test}}/ (0 bytes copied).")
    print("data/real_environment_dataset and everything else under data/ were left untouched.")
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=os.path.join(REPO_ROOT, "data", "raw"))
    parser.add_argument("--processed", default=os.path.join(REPO_ROOT, "data", "processed"))
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed (default 42).")
    parser.add_argument("--exclude", nargs="+", default=[],
                        help="Class folder name(s) to omit from the split (raw kept, not linked).")
    parser.add_argument("--skip-if-exists", action="store_true",
                        help="Skip re-splitting IF train/val/test already hold exactly the "
                             "expected class set (raw minus --exclude); a mismatch re-splits.")
    args = parser.parse_args()
    print(f"Splitting {args.raw}\n      into {args.processed}  (70/15/15, seed {args.seed})")
    split(args.raw, args.processed, seed=args.seed, exclude=args.exclude,
          skip_if_exists=args.skip_if_exists)


if __name__ == "__main__":
    main()
