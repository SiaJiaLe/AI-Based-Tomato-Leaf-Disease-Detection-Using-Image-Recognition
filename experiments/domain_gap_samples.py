"""Visual PlantVillage vs real-world comparison for the structural-zero classes.

Target_Spot (40 real images) and Tomato_mosaic_virus (32) score EXACTLY 0 correct in
every evaluated model - below the ~4/40 random guessing over 10 classes would give. So
the models are not "confused", they have a confident CONSISTENT wrong answer. This
script builds a figure that separates the two explanations:

  Row A  PlantVillage samples of the TRUE class      (what the model learned)
  Row B  real-world samples of the TRUE class        (what it is tested on, 0 correct)
  Row C  PlantVillage samples of the DOMINANT PREDICTED class
         (read from the saved confusion counts) - so you can judge whether the real
         Target_Spot leaves look more like PlantVillage Septoria than like PlantVillage
         Target_Spot. If B resembles C, it is a label/definition mismatch (a DATA
         problem); if B just looks unlike A, it is plain domain shift.

    python -m experiments.domain_gap_samples
    python -m experiments.domain_gap_samples --n 8 --pv-split train --seed 0
    python -m experiments.domain_gap_samples --predicted-from efficientnetb0_on

No model, no inference, no GPU - just reads images off disk. Additive; touches nothing
else. ASCII-only console output.
"""
import argparse
import json
import os
import random

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "experiments", "results")
CONFUSION_DIR = os.path.join(RESULTS_DIR, "confusion")

PV_ROOT = os.path.join(REPO_ROOT, "data", "processed")               # <split>/<class>
RW_ROOT = os.path.join(REPO_ROOT, "data", "real_environment_dataset")  # <class>

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

DEFAULT_CLASSES = ["Tomato___Target_Spot", "Tomato___Tomato_mosaic_virus"]

# Fallback dominant-prediction map if the confusion counts are unavailable; the script
# prefers the data-driven value read from the counts JSON.
FALLBACK_PREDICTED = {
    "Tomato___Target_Spot": "Tomato___Septoria_leaf_spot",
    "Tomato___Tomato_mosaic_virus": "Tomato___Early_blight",
}


def _short(name):
    return name.replace("Tomato___", "").replace("_", " ")


def _list_images(d):
    if not os.path.isdir(d):
        return []
    return sorted(os.path.join(d, f) for f in os.listdir(d)
                  if os.path.splitext(f)[1].lower() in IMAGE_EXT)


def _sample(paths, n, rng):
    if len(paths) <= n:
        return list(paths)
    return rng.sample(paths, n)


def dominant_predicted(class_name, ref_run):
    """The class most real-world images of `class_name` are predicted as, read from the
    saved confusion counts of `ref_run`. Falls back to FALLBACK_PREDICTED if unavailable."""
    path = os.path.join(CONFUSION_DIR, f"{ref_run}_cm_real_world.json")
    if os.path.isfile(path):
        with open(path) as f:
            data = json.load(f)
        names = data["class_names"]
        counts = data["counts"]
        if class_name in names:
            i = names.index(class_name)
            row = counts[i]
            # argmax over predicted columns EXCLUDING the true class (diagonal is 0 for
            # these classes anyway, but exclude it so this is correct in general).
            best_j, best_v = None, -1
            for j, v in enumerate(row):
                if j == i:
                    continue
                if v > best_v:
                    best_j, best_v = j, v
            if best_j is not None and best_v > 0:
                return names[best_j], best_v, sum(row)
    fb = FALLBACK_PREDICTED.get(class_name)
    return fb, None, None


def _load_thumb(path, size):
    from PIL import Image
    img = Image.open(path).convert("RGB")
    img.thumbnail((size, size))
    return img


def _caption(class_name, ref_run):
    """One-line note under a class block: real count, 0-correct, where the misses go."""
    pred, n_to_pred, n_total = dominant_predicted(class_name, ref_run)
    rw_n = len(_list_images(os.path.join(RW_ROOT, class_name)))
    base = f"{_short(class_name)}: {rw_n} real images, 0 correct in every evaluated model."
    if pred is None:
        return base
    if n_to_pred is not None and n_total:
        pct = 100.0 * n_to_pred / n_total
        return (base + f" Most often predicted as {_short(pred)} "
                f"({n_to_pred}/{n_total} = {pct:.0f}%, from {ref_run}).")
    return base + f" Most often predicted as {_short(pred)}."


def _rows_for_class(class_name, pv_split, n, rng, ref_run):
    """Build the three (label, image_paths) rows for one class."""
    pred, _, _ = dominant_predicted(class_name, ref_run)

    def need(d, what):
        imgs = _list_images(d)
        if not imgs:
            raise FileNotFoundError(
                f"No images for {what} in {d}. A silently empty folder would render a "
                f"blank row and read as a finding - refusing. Check the path/download.")
        return imgs

    a = need(os.path.join(PV_ROOT, pv_split, class_name), f"PlantVillage {pv_split} {_short(class_name)}")
    b = need(os.path.join(RW_ROOT, class_name), f"real-world {_short(class_name)}")
    rows = [
        (f"PlantVillage\n{_short(class_name)}\n(learned)", _sample(a, n, rng)),
        (f"Real-world\n{_short(class_name)}\n(0 correct)", _sample(b, n, rng)),
    ]
    if pred is not None:
        c = need(os.path.join(PV_ROOT, pv_split, pred), f"PlantVillage {pv_split} {_short(pred)}")
        rows.append((f"PlantVillage\n{_short(pred)}\n(top prediction)", _sample(c, n, rng)))
    return rows


def _draw(rows, n, thumb, out_path, suptitle, caption):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nrows = len(rows)
    fig, axes = plt.subplots(nrows, n, figsize=(n * 1.5, nrows * 1.7 + 0.8), squeeze=False)
    for r, (label, paths) in enumerate(rows):
        for c in range(n):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if c < len(paths):
                ax.imshow(_load_thumb(paths[c], thumb))
                ax.set_title(os.path.basename(paths[c]), fontsize=4, pad=1)
        axes[r][0].set_ylabel(label, rotation=0, ha="right", va="center",
                              fontsize=8, labelpad=38)
    fig.suptitle(suptitle, fontsize=11, y=0.995)
    fig.text(0.5, 0.008, caption, ha="center", va="bottom", fontsize=8, wrap=True)
    fig.tight_layout(rect=[0.04, 0.03, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES,
                        help="Class folder names to compare (default: the two zero classes).")
    parser.add_argument("--pv-split", default="test", choices=["train", "val", "test"],
                        help="PlantVillage split to sample (default: test, the controlled set).")
    parser.add_argument("--n", type=int, default=8, help="Samples per row (default 8).")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed (reproducible figure).")
    parser.add_argument("--thumb", type=int, default=180, help="Thumbnail max side px.")
    parser.add_argument("--predicted-from", default="efficientnetb0_on",
                        help="Run whose confusion counts decide the Row C predicted class.")
    args = parser.parse_args()

    os.makedirs(CONFUSION_DIR, exist_ok=True)
    all_rows, written = [], []
    for cls in args.classes:
        rng = random.Random(args.seed)   # same seed per class -> reproducible
        rows = _rows_for_class(cls, args.pv_split, args.n, rng, args.predicted_from)
        cap = _caption(cls, args.predicted_from)
        out = os.path.join(CONFUSION_DIR, f"domain_gap_{_short(cls).replace(' ', '_')}.png")
        _draw(rows, args.n, args.thumb, out,
              f"PlantVillage vs real-world - {_short(cls)}", cap)
        written.append(out)
        print(f"  wrote {out}")
        print(f"    {cap}")
        all_rows.extend(rows)

    if len(args.classes) > 1:
        combined = os.path.join(CONFUSION_DIR, "domain_gap_combined.png")
        caps = " | ".join(_caption(c, args.predicted_from) for c in args.classes)
        _draw(all_rows, args.n, args.thumb, combined,
              "PlantVillage vs real-world - structural-zero classes", caps)
        written.append(combined)
        print(f"  wrote {combined}")

    print(f"\nDone. {len(written)} figure(s) in {CONFUSION_DIR}.")
    print("Read: does Row B (real) look like Row A (PlantVillage same class) or like "
          "Row C (PlantVillage predicted class)? B~C => label/definition mismatch (data); "
          "B unlike A only => domain shift.")


if __name__ == "__main__":
    main()
