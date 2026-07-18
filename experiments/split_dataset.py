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


def split(raw_dir, processed_dir, seed=42, exclude=None):
    """Symlink every raw image into processed/{train,val,test}/<class>/. Returns counts.

    `exclude` is a list of class folder names to OMIT from the split (their raw images
    are left on disk, just not linked in), so a class can be turned off from training
    without deleting anything.
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
    args = parser.parse_args()
    print(f"Splitting {args.raw}\n      into {args.processed}  (70/15/15, seed {args.seed})")
    split(args.raw, args.processed, seed=args.seed, exclude=args.exclude)


if __name__ == "__main__":
    main()
