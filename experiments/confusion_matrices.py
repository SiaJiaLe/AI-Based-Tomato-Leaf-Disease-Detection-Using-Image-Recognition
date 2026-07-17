"""Recover every evaluated row's real-world confusion matrix as NUMBERS.

    python -m experiments.confusion_matrices                # re-infer + report
    python -m experiments.confusion_matrices --report-only  # no GPU, reuse saved counts
    python -m experiments.confusion_matrices --markdown     # markdown to stdout
    python -m experiments.confusion_matrices --force        # re-infer even if cached

WHY THIS EXISTS
---------------
`common.evaluate._plot_confusion` already draws `cm_real_world_test.png` for every
evaluated run, and it even computes the counts - `return cm.tolist()` (evaluate.py:85).
Both callers throw that return value away. So the matrices exist only as PIXELS.

You can eyeball one PNG. You cannot compare eighteen of them, rank error modes, or
put "what does Target_Spot actually get called" in a thesis table. And the saved
classification_report cannot substitute: per-class precision/recall/F1/support pins
down the DIAGONAL and the column totals, but not the OFF-diagonal - which is the
entire question ("misclassified as WHAT").

Hence one re-inference pass. It is the only way; there is no offline shortcut.

IS THIS A SECOND READ OF THE TEST SET?
--------------------------------------
No. The read-once rule exists so the real-world set cannot influence model or
hyperparameter SELECTION. This pass loads FROZEN checkpoints, runs DETERMINISTIC
inference, and reports a richer summary of the very same forward pass whose
aggregate is already published. Nothing is trained, tuned, or chosen. The
predictions are the same predictions.

That claim is CHECKED, not asserted: every row recomputes real-world macro-F1 and
hard-fails unless it reproduces the published number to 1e-6. A matrix that cannot
reproduce its own published F1 aborts the run. (It doubles as a regression test on
the repaired conda env, and it would catch a preprocessing mismatch - e.g. scoring
the 240px row at 224.)

And the exclusion of the val-only rows is STRUCTURAL, not a list to maintain: this
script only touches rows that ALREADY have a published eval_results_real_world.json.
`droppath03` and `mixstyle_l12` lost on validation and their real-world set was
never read, so they have no such file, so this script cannot read it for them. Doing
less is the rigorous choice - reading the test set now for configs we discarded is
the thing that would actually violate the protocol.

Isolation: additive. `common/evaluate.py`, `run.py`, `compare*.py` and every config
are imported or ignored, never modified.

Output: experiments/results/confusion/
  <run>_cm_real_world.json   raw counts + class names + reproduced macro-F1 (permanent:
                             once written, this pass never needs repeating)
  <run>.png                  per-run row-normalized matrix
  grid_efficientnetb0.png    the EfficientNetB0 story as one multi-panel figure
  confusion_report.{txt,md}  aligned text for the terminal, markdown for the report
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from experiments.common.data import AlbumentationsImageFolder, build_eval_transform
from experiments.common.evaluate import _load_model, _metrics, _predict
from experiments.compile_results import (BACKBONE_ORDER, BASELINE, RESULTS_DIR,
                                         STORY_ROWS, Table, _fmt, _load,
                                         render_markdown, render_text)

OUT_DIR = os.path.join(RESULTS_DIR, "confusion")

# The published macro-F1 must come back bit-for-bit. Float reduction order can move
# the last ulp or two; anything above this is a real difference, not arithmetic.
TOL = 1e-6


def _short(name):
    """Compact a run name enough to fit a table header."""
    return name.replace("efficientnetb0_on", "on").replace("efficientnetb0", "eb0")


def _clean(cls):
    return cls.replace("Tomato___", "")


def _resolution_of(cfg):
    return int(cfg.get("architecture_mod", {}).get("input_resolution", 224))


def _real_world_loader(cfg, image_size):
    """Rebuild the row's real-world loader EXACTLY as its own eval path did.

    run_arch.py:121-125 dispatches on input_resolution, because a model trained at
    240 scored at 224 is a silent train/eval preprocessing mismatch - the precise
    hazard CLAUDE.md warns about. Mirror that dispatch here; the macro-F1 check
    would catch it if this drifted.
    """
    if image_size != 224:
        from experiments.plan2_arch.data_res import (build_eval_transform_res,
                                                     default_resize_to)
        eval_tf = build_eval_transform_res(image_size, default_resize_to(image_size))
    else:
        eval_tf = build_eval_transform(224)
    ds = AlbumentationsImageFolder(cfg["real_world_dir"], transform=eval_tf)
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=False,
                        num_workers=4, pin_memory=True)
    return ds, loader


def cm_path(run):
    return os.path.join(OUT_DIR, f"{run}_cm_real_world.json")


def compute_cm(run, device, force=False):
    """Return the saved/recomputed confusion matrix for one run, or None to skip."""
    published = _load(run, "eval_results_real_world.json")
    if published is None:
        # Never evaluated => the real-world set was never read for this row. This
        # script will not be the thing that reads it. See the module docstring.
        return None

    if not force and os.path.isfile(cm_path(run)):
        with open(cm_path(run)) as f:
            return json.load(f)

    results_dir = os.path.join(RESULTS_DIR, run)
    model, cfg, class_to_idx = _load_model(results_dir, device)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    image_size = _resolution_of(cfg)
    ds, loader = _real_world_loader(cfg, image_size)
    real_idx_to_class = {v: k for k, v in ds.class_to_idx.items()}

    yt_local, yp = _predict(model, loader, device)
    # Remap real-world folder indices into the training label space BY NAME -
    # identical to common.evaluate:117, so a different on-disk class ordering
    # cannot silently scramble the matrix.
    yt = np.array([class_to_idx[real_idx_to_class[i]] for i in yt_local])

    recomputed = _metrics(yt, yp, class_names)["macro_f1"]
    expected = published["macro_f1"]
    if abs(recomputed - expected) > TOL:
        raise RuntimeError(
            f"{run}: recomputed real-world macro-F1 {recomputed:.6f} != published "
            f"{expected:.6f} (tolerance {TOL}). This pass is only legitimate if it "
            f"reproduces the published forward pass exactly, so it stops here rather "
            f"than write a matrix that does not match its own reported number. Likely "
            f"causes: the environment drifted, the checkpoint changed, or this row was "
            f"scored at the wrong input resolution (this one ran at {image_size}px).")

    cm = confusion_matrix(yt, yp, labels=list(range(len(class_names))))
    data = {"run": run, "class_names": class_names, "input_resolution": image_size,
            "counts": cm.tolist(), "macro_f1": recomputed,
            "macro_f1_published": expected, "n_images": int(len(yt))}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(cm_path(run), "w") as f:
        json.dump(data, f, indent=2)
    print(f"  {run}: macro-F1 {recomputed:.4f} reproduces published {expected:.4f} "
          f"at {image_size}px, {len(yt)} images -> counts saved.", flush=True)
    return data


def _norm(counts):
    cm = np.array(counts, dtype=float)
    sums = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, sums, out=np.zeros_like(cm), where=sums != 0)


def _draw(ax, data, title, fontsize=6):
    names = [_clean(c) for c in data["class_names"]]
    cmn = _norm(data["counts"])
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=fontsize)
    ax.set_yticklabels(names, fontsize=fontsize)
    ax.set_xlabel("Predicted", fontsize=fontsize + 1)
    ax.set_ylabel("True", fontsize=fontsize + 1)
    ax.set_title(title, fontsize=fontsize + 2)
    for i in range(len(names)):
        for j in range(len(names)):
            if cmn[i, j] > 0.005:
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=fontsize - 1,
                        color="white" if cmn[i, j] > 0.5 else "black")
    return im


def plot_one(data):
    fig, ax = plt.subplots(figsize=(10, 8))
    _draw(ax, data, f"{data['run']} - real-world test (row-normalized)", fontsize=7)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, f"{data['run']}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_grid(mats, runs, fname, suptitle):
    """All the EfficientNetB0 rows in one figure - the point is the COMPARISON."""
    present = [r for r in runs if r in mats]
    if len(present) < 2:
        return None
    ncols = 3
    nrows = (len(present) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, run in zip(axes, present):
        _draw(ax, mats[run], f"{_short(run)}  (RW F1 {mats[run]['macro_f1']:.4f})", fontsize=5)
    for ax in axes[len(present):]:
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def top_confusions(data, k=5):
    """The k biggest off-diagonal cells: (true, pred, count, fraction of that class)."""
    cm = np.array(data["counts"])
    names = data["class_names"]
    out = []
    for i in range(len(names)):
        support = cm[i].sum()
        for j in range(len(names)):
            if i != j and cm[i, j] > 0:
                out.append((_clean(names[i]), _clean(names[j]), int(cm[i, j]),
                            cm[i, j] / support if support else 0.0))
    out.sort(key=lambda t: t[2], reverse=True)
    return out[:k]


def confusions_table(mats, runs):
    headers = ["row", "true class", "predicted as", "n", "% of true class"]
    body = []
    for run in runs:
        if run not in mats:
            continue
        for t, p, n, frac in top_confusions(mats[run]):
            body.append([_short(run), t, p, str(n), f"{100 * frac:.1f}"])
    return Table("Top-5 real-world confusions per row",
                 headers, body,
                 "Read as: images whose TRUE class is column 2 were predicted as column 3. "
                 "'% of true class' is the share of that class's real-world images that landed "
                 "in that wrong bucket, which is the number that says whether it is a dominant "
                 "failure mode or a rounding error.")


def error_profile_table(mats, runs):
    """For each true class, its dominant WRONG prediction in each row.

    This is the table that answers the actual research question: did an intervention
    change WHICH mistake the model makes, or only how often it makes the same one?
    """
    present = [r for r in runs if r in mats]
    if not present:
        return Table("Dominant error per class", ["class"], [], "No matrices available.")
    names = mats[present[0]]["class_names"]
    headers = ["true class"] + [_short(r) for r in present]
    body = []
    for i, c in enumerate(names):
        cells = []
        for run in present:
            cm = np.array(mats[run]["counts"])
            support = cm[i].sum()
            if support == 0:
                cells.append("no images")
                continue
            correct = cm[i, i] / support
            off = [(j, cm[i, j]) for j in range(len(names)) if j != i and cm[i, j] > 0]
            if not off:
                # No errors at all. Reporting a "dominant error" of 0% here would
                # invent a failure mode that does not exist.
                cells.append(f"none (ok {100 * correct:.0f}%)")
                continue
            j, n = max(off, key=lambda t: t[1])
            cells.append(f"{_clean(names[j])[:14]} {100 * n / support:.0f}% (ok {100 * correct:.0f}%)")
        body.append([_clean(c)] + cells)
    return Table("Dominant error per true class (real-world)", headers, body,
                 "Each cell: the single most common WRONG prediction for that true class, its "
                 "share of the class, and in brackets the share correctly classified. If the "
                 "wrong bucket is identical across rows, the interventions changed the RATE of "
                 "a failure but not its NATURE - which is itself a finding.")


def focus_table(mats, runs, target="Tomato___Target_Spot"):
    """Where does a chosen class actually go? Built for Target_Spot.

    Target_Spot scores near-zero real-world F1 in EVERY row of this study and has
    never been explained. The matrix says where those images land instead - the
    cheapest available shot at that.
    """
    present = [r for r in runs if r in mats]
    if not present or target not in mats[present[0]]["class_names"]:
        return None
    names = mats[present[0]]["class_names"]
    i = names.index(target)
    headers = ["row", "images", "correct", "-> 1st", "-> 2nd", "-> 3rd"]
    body = []
    for run in present:
        cm = np.array(mats[run]["counts"])
        support = cm[i].sum()
        if support == 0:
            body.append([_short(run), "0", "n/a", "no images", "", ""])
            continue
        # Only destinations that actually RECEIVED images. Ranking all nine and
        # taking three would pad the table with "SomeClass 0%", which reads as
        # "some images went there" when none did.
        dest = sorted(((j, cm[i, j]) for j in range(len(names)) if j != i and cm[i, j] > 0),
                      key=lambda t: t[1], reverse=True)[:3]
        cells = [f"{_clean(names[j])[:18]} {100 * n / support:.0f}%" for j, n in dest]
        cells += [""] * (3 - len(cells))
        body.append([_short(run), str(int(support)), f"{100 * cm[i, i] / support:.0f}%"] + cells)
    return Table(f"Focus: where do {_clean(target)} images actually go? (real-world)",
                 headers, body,
                 f"{_clean(target)} has near-zero real-world F1 in every row of this study and "
                 "is so far unexplained. If its images land overwhelmingly in ONE other class "
                 "across every row, that points at a labelling or class-definition problem in "
                 "the real-world set - a data failure, not a modelling one. If they scatter, "
                 "the class is simply not learnable from PlantVillage's version of it.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-only", action="store_true",
                        help="Build the report from saved counts; no model, no GPU.")
    parser.add_argument("--force", action="store_true",
                        help="Re-infer even where counts are already saved.")
    parser.add_argument("--markdown", action="store_true",
                        help="Print markdown to stdout instead of aligned text.")
    parser.add_argument("--no-write", action="store_true", help="Print only; write no files.")
    args = parser.parse_args()

    ablation = [f"{bb}_{t}" for bb in BACKBONE_ORDER for t in ("off", "on")]
    story = [r for r, _ in STORY_ROWS]
    # Baseline first: an error profile is only readable against the row it should beat.
    eb0_story = [BASELINE] + [r for r in story if r != BASELINE]
    all_runs = ablation + [r for r in story if r not in ablation]

    device = torch.device("cpu")
    if not args.report_only:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Re-inferring real-world predictions on {device}. Frozen checkpoints, "
              f"deterministic; each row must reproduce its published macro-F1.", flush=True)

    mats, skipped = {}, []
    for run in all_runs:
        if not os.path.isdir(os.path.join(RESULTS_DIR, run)):
            continue
        if args.report_only:
            if os.path.isfile(cm_path(run)):
                with open(cm_path(run)) as f:
                    mats[run] = json.load(f)
            continue
        data = compute_cm(run, device, force=args.force)
        if data is None:
            skipped.append(run)
        else:
            mats[run] = data

    if not mats:
        print("No confusion matrices available. Run without --report-only first "
              "(needs the run directories and their checkpoints).")
        return

    if not args.no_write:
        os.makedirs(OUT_DIR, exist_ok=True)
        for run in mats:
            plot_one(mats[run])
        plot_grid(mats, eb0_story, "grid_efficientnetb0.png",
                  "EfficientNetB0 real-world confusion: baseline vs every intervention")
        plot_grid(mats, ablation, "grid_ablation.png",
                  "Backbone ablation real-world confusion: stack OFF vs ON")

    preamble = [
        "Every matrix below was recomputed from the row's own FROZEN checkpoint and verified "
        "to reproduce the macro-F1 already published in its eval_results_real_world.json to "
        "1e-6. Nothing was trained, tuned, or selected: these are the same predictions, "
        "summarised in more detail than the aggregate metrics could show.",
        "Rows that were never evaluated (sweep members that lost on validation) have no "
        "published real-world results, and this script does not read the test set for them.",
        "CAVEAT: these describe single-seed (42) runs. A confusion pair that shifts between "
        "two rows may be noise, exactly as the +0.0052 headline may be. Read this as error "
        "STRUCTURE and as hypotheses - not as evidence of significance.",
    ]

    tables = [error_profile_table(mats, eb0_story)]
    focus = focus_table(mats, eb0_story)
    if focus:
        tables.append(focus)
    tables.append(confusions_table(mats, eb0_story))
    tables.append(error_profile_table(mats, ablation))

    text = render_text(tables, preamble)
    md = render_markdown(tables, preamble)
    print(md if args.markdown else text)

    if skipped:
        print("Skipped (never evaluated - real-world set deliberately not read for them): "
              + ", ".join(skipped))

    if args.no_write:
        return
    txt_path = os.path.join(OUT_DIR, "confusion_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    md_path = os.path.join(OUT_DIR, "confusion_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(f"\nWrote {txt_path}  (aligned text)")
    print(f"Wrote {md_path}  (markdown - paste into the report)")
    print(f"Wrote {len(mats)} matrices + figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
