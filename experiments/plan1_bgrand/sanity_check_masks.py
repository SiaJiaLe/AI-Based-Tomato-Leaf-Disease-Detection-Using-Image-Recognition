"""Eyeball segmentation + composite quality BEFORE the full run (plan §5.2).

Bad masks poison training, so look at these first. For a few images per class
it saves a grid [ original | leaf mask | composite-on-random-background ] to
experiments/results/plan1_bgrand/mask_sanity/<class>.png.

    python -m experiments.plan1_bgrand.sanity_check_masks
    python -m experiments.plan1_bgrand.sanity_check_masks --per-class 4
"""
import argparse
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .bg_randomize import (_load_backgrounds, composite, make_rembg_session,
                           segment_leaf, segment_leaf_rembg)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_TRAIN = os.path.join(REPO_ROOT, "data", "processed", "train")
DEFAULT_BG = os.path.join(REPO_ROOT, "data", "backgrounds_generic_synthetic")
OUT_DIR = os.path.join(REPO_ROOT, "experiments", "results", "plan1_bgrand", "mask_sanity")
_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def _read_rgb(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default=DEFAULT_TRAIN)
    parser.add_argument("--background-dir", default=DEFAULT_BG)
    parser.add_argument("--per-class", type=int, default=3)
    parser.add_argument("--segmentation", default="pretrained",
                        choices=["pretrained", "hsv_threshold"],
                        help="Match the config so the preview reflects the real pipeline.")
    parser.add_argument("--erode-px", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    backgrounds = _load_backgrounds(args.background_dir)
    session = make_rembg_session() if args.segmentation == "pretrained" else None
    os.makedirs(OUT_DIR, exist_ok=True)

    classes = sorted(d for d in os.listdir(args.train_dir)
                     if os.path.isdir(os.path.join(args.train_dir, d)))
    for cls in classes:
        cls_dir = os.path.join(args.train_dir, cls)
        files = [f for f in sorted(os.listdir(cls_dir)) if f.lower().endswith(_EXTS)]
        if not files:
            continue
        picks = rng.choice(len(files), size=min(args.per_class, len(files)), replace=False)

        n = len(picks)
        fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
        if n == 1:
            axes = axes[None, :]
        for row, i in enumerate(picks):
            img = _read_rgb(os.path.join(cls_dir, files[i]))
            if img is None:
                continue
            if args.segmentation == "pretrained":
                mask = segment_leaf_rembg(img, session)
            else:
                mask = segment_leaf(img)
            bg = backgrounds[rng.integers(len(backgrounds))]
            comp = composite(img, mask, bg, boundary_blur=True, erode_px=args.erode_px)
            fg = float((mask > 0).mean())
            for ax, im, title in zip(
                    axes[row],
                    [img, mask, comp],
                    ["original", f"mask (fg={fg:.0%})", "composite"]):
                ax.imshow(im, cmap="gray" if im.ndim == 2 else None)
                ax.set_title(title, fontsize=9)
                ax.axis("off")
        fig.suptitle(cls, fontsize=11)
        fig.tight_layout()
        out = os.path.join(OUT_DIR, f"{cls}.png")
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"  wrote {out}", flush=True)

    print(f"\nMask sanity grids in {OUT_DIR}. Inspect before launching the full run.", flush=True)


if __name__ == "__main__":
    main()
