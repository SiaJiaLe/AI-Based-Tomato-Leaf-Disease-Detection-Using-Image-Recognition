"""Plan 3 end-to-end: DINOv2 + ChromaDB retrieval and CNN fusion.

    python -m experiments.plan3_chromadb.run_plan3 \
        --chroma-path /content/drive/MyDrive/tomato_fyp/chroma_store \
        --cache-dir  /content/drive/MyDrive/tomato_fyp/dino_cache

Execution order (mirrors plan3 sec 9):
  1. Load the frozen bgrand_real (seed 42) checkpoint.
  2. Build the Chroma index from the TRAIN split only, with leak assertions.
  3. Retrieval sanity check (self-distance ~0) + class-alignment assertion.
  4. CNN forward on val/test/real; calibration (ECE + temperature T) on validation.
  5. DINO-embed val/test/real; query k_max neighbours once and reuse.
  6. Row A grid (k, temp) on VALIDATION -> freeze -> read real-world once.
  7. Row B grid (k, temp, w) on VALIDATION -> freeze -> read real-world once.
  8. Row B-abs grid (tau, d_max) on VALIDATION -> freeze -> read real-world once.
  9. Novelty-signal AUROC on the real set (sec 7), reported regardless of the verdict.
 10. Compare to the pre-registered +0.03 rule and write all rows + a summary.

HYGIENE (sec 2): every tunable knob is selected on validation macro-F1; the real-world
set influences NO selection - its metrics are computed only after the params are frozen.
Rows are written in the SAME shape as every prior row (via common.evaluate metrics), so
they drop straight into the report tables. ASCII-only output.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

from experiments.common.evaluate import _metrics, _plot_confusion
from experiments.common.seeding import seed_everything
from experiments.compile_results import RESULTS_DIR, _load

from .calibration import report_calibration
from .cnn import cnn_forward_folder, load_cnn
from .embed_dino import embed_paths_cached, load_dino
from .fusion import (abstain_mask, fuse, query_neighbors, retrieval_matrix)
from .index import assert_class_alignment, build_index, sanity_check

# Pre-registered baseline (plan3 sec 1): bgrand_real 3-seed real-world macro-F1.
BASELINE_MEAN = 0.4641
BASELINE_STD = 0.0312
ADOPT_THRESHOLD = 0.03      # fusion is a contribution only if it beats baseline by > this

# Validation grid (plan3 sec 5).
K_GRID = [1, 5, 10, 20]
TEMP_GRID = [0.1, 0.3, 1.0]
W_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
TAU_GRID = [0.35, 0.45, 0.55]
DMAX_PCTILES = [60, 75, 90]
K_MAX = max(K_GRID)

ROW_A_DIR = "plan3_retrieval"
ROW_B_DIR = "plan3_fusion"
ROW_BA_DIR = "plan3_fusion_abstain"


def _macro_f1(yt, yp, n_classes):
    """macro-F1 with a fixed label set; abstained rows (yp == -1) count as errors."""
    return float(f1_score(yt, yp, labels=list(range(n_classes)),
                          average="macro", zero_division=0))


def _fix_abstain_macro(metrics, yt, yp, n_classes):
    """When yp contains abstentions (-1), common.evaluate._metrics averages macro over
    the union of labels - which pulls -1 in as a spurious class and divides by the wrong
    count. Recompute macro precision/recall/F1 over the FIXED real class set (abstained
    rows correctly count as recall misses). accuracy / weighted_f1 / the per-class report
    from _metrics are already correct (they use the fixed label set or count -1 as wrong)."""
    from sklearn.metrics import precision_recall_fscore_support
    p, r, f1, _ = precision_recall_fscore_support(
        yt, yp, labels=list(range(n_classes)), average="macro", zero_division=0)
    m = dict(metrics)
    m["macro_precision"], m["macro_recall"], m["macro_f1"] = float(p), float(r), float(f1)
    return m


def _dino_embed_split(paths, cache_dir, tag, dino, device, model_name):
    cache = os.path.join(cache_dir, f"dino_{model_name}_{tag}.npz") if cache_dir else None
    return embed_paths_cached(paths, cache, model=dino, device=device, model_name=model_name)


def _write_row(run_dir, run_name, controlled, real, class_names, real_yt, real_yp, extra):
    """Write one Plan 3 row in the canonical shape: eval_results.json (controlled),
    eval_results_real_world.json (real, with gap), a confusion PNG, plus plan3 extras."""
    os.makedirs(run_dir, exist_ok=True)
    controlled = dict(controlled); controlled["model"] = run_name
    real = dict(real); real["model"] = run_name
    real["generalization_gap_accuracy"] = controlled["accuracy"] - real["accuracy"]
    real["generalization_gap_macro_f1"] = controlled["macro_f1"] - real["macro_f1"]
    real["scored_on"] = "plan3_dinov2_chromadb"
    real.update(extra)
    with open(os.path.join(run_dir, "eval_results.json"), "w", encoding="utf-8") as f:
        json.dump(controlled, f, indent=2)
    with open(os.path.join(run_dir, "eval_results_real_world.json"), "w", encoding="utf-8") as f:
        json.dump(real, f, indent=2)
    # Confusion PNG from the real predictions (abstained rows, yp == -1, are dropped
    # from the matrix by the fixed label set - i.e. the matrix is over answered images).
    _plot_confusion(real_yt, real_yp, class_names,
                    os.path.join(run_dir, "cm_real_world_test.png"),
                    f"{run_name} - real-world test (row-normalized)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default=os.path.join(RESULTS_DIR, "efficientnetb0_on_bgrand_real"),
                    help="Directory with the seed-42 bgrand_real best_model.pth.")
    ap.add_argument("--data-dir", default=None, help="Overrides the checkpoint's data_dir.")
    ap.add_argument("--real-dir", default=None, help="Overrides the checkpoint's real_world_dir.")
    ap.add_argument("--chroma-path", default="chroma_store",
                    help="Persistent Chroma dir. On Colab point at Drive to survive the runtime.")
    ap.add_argument("--cache-dir", default=None,
                    help="Dir for cached DINO embeddings (npz). On Colab point at Drive.")
    ap.add_argument("--dino-model", default="dinov2_vits14")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--rebuild-index", action="store_true",
                    help="Force a fresh HNSW build even if a full index already exists.")
    ap.add_argument("--out-dir", default=RESULTS_DIR)
    args = ap.parse_args()

    # HNSW index construction is the only stochastic step in plan3 (sec 2.4); seed first.
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("  WARNING: no CUDA device. DINO extraction and CNN inference will be slow.\n",
              flush=True)
    print(f"Plan 3 on {device}. DINO={args.dino_model}, seed={args.seed}.\n", flush=True)

    # --- 1. CNN (frozen) ---
    model, cfg, class_to_idx = load_cnn(args.ckpt_dir, device)
    data_dir = args.data_dir or cfg["data_dir"]
    real_dir = args.real_dir or cfg["real_world_dir"]
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")
    n_classes = len(class_to_idx)
    print(f"CNN: {cfg['run_name']} | data={data_dir} | real={real_dir} | "
          f"{n_classes} classes.\n", flush=True)

    # --- 2/3. Index + sanity + alignment ---
    cache_dir = args.cache_dir
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    train_cache = os.path.join(cache_dir, f"dino_{args.dino_model}_train.npz") if cache_dir else None
    col, classes = build_index(train_dir, args.chroma_path, cache_file=train_cache,
                               device=device, model_name=args.dino_model,
                               rebuild=args.rebuild_index)
    sanity_check(col, train_dir, device=device, model_name=args.dino_model)
    assert_class_alignment(classes, class_to_idx)
    classes_idx = {c: i for i, c in enumerate(classes)}

    # --- 4. CNN forward on each split + calibration (validation) ---
    print("\nCNN forward passes (val / test / real)...", flush=True)
    val_cnn = cnn_forward_folder(model, val_dir, class_to_idx, device, args.batch_size)
    test_cnn = cnn_forward_folder(model, test_dir, class_to_idx, device, args.batch_size)
    real_cnn = cnn_forward_folder(model, real_dir, class_to_idx, device, args.batch_size)
    class_names = val_cnn["class_names"]

    T, ece_raw, ece_cal = report_calibration(val_cnn["logits"], val_cnn["probs"], val_cnn["yt"])
    # Use temperature-calibrated CNN probabilities in fusion (sec 4.5). T==1 -> no change.
    def _cal(logits):
        z = torch.from_numpy(logits.astype("float32")) / T
        return torch.softmax(z, dim=1).numpy().astype("float32")
    val_p, test_p, real_p = _cal(val_cnn["logits"]), _cal(test_cnn["logits"]), _cal(real_cnn["logits"])

    # --- 5. DINO embeddings (aligned to the CNN path order) + neighbour queries ---
    print("\nDINO embeddings + neighbour queries...", flush=True)
    dino = load_dino(args.dino_model, device)
    val_emb = _dino_embed_split(val_cnn["paths"], cache_dir, "val", dino, device, args.dino_model)
    test_emb = _dino_embed_split(test_cnn["paths"], cache_dir, "test", dino, device, args.dino_model)
    real_emb = _dino_embed_split(real_cnn["paths"], cache_dir, "real", dino, device, args.dino_model)
    val_nb = query_neighbors(col, val_emb, k_max=K_MAX)
    test_nb = query_neighbors(col, test_emb, k_max=K_MAX)
    real_nb = query_neighbors(col, real_emb, k_max=K_MAX)

    val_yt, test_yt, real_yt = val_cnn["yt"], test_cnn["yt"], real_cnn["yt"]

    # ========================= Row A: retrieval only =========================
    print("\n=== Row A: retrieval only (DINOv2 + Chroma, no CNN) ===", flush=True)
    bestA, bestA_f1 = None, -1.0
    for k in K_GRID:
        for temp in TEMP_GRID:
            S, _ = retrieval_matrix(val_nb, classes_idx, n_classes, k, temp)
            f1 = _macro_f1(val_yt, S.argmax(1), n_classes)
            if f1 > bestA_f1:
                bestA_f1, bestA = f1, (k, temp)
    kA, tempA = bestA
    print(f"  VAL pick: k={kA}, temp={tempA} (val macro-F1 {bestA_f1:.4f}).", flush=True)
    # freeze -> read controlled test + real-world once
    S_test, _ = retrieval_matrix(test_nb, classes_idx, n_classes, kA, tempA)
    S_real, near_real_A = retrieval_matrix(real_nb, classes_idx, n_classes, kA, tempA)
    A_ctrl = _metrics(test_yt, S_test.argmax(1), class_names)
    A_real_yp = S_real.argmax(1)
    A_real = _metrics(real_yt, A_real_yp, class_names)
    _write_row(os.path.join(args.out_dir, ROW_A_DIR), ROW_A_DIR, A_ctrl, A_real,
               class_names, real_yt, A_real_yp,
               {"val_macro_f1": bestA_f1, "params": {"k": kA, "temp": tempA}})
    print(f"  REAL-WORLD macro-F1 {A_real['macro_f1']:.4f} (acc {A_real['accuracy']:.4f}).",
          flush=True)

    # ========================= Row B: fusion (no abstention) =========================
    print("\n=== Row B: CNN + Chroma fusion (no abstention) ===", flush=True)
    bestB, bestB_f1 = None, -1.0
    for k in K_GRID:
        for temp in TEMP_GRID:
            S, _ = retrieval_matrix(val_nb, classes_idx, n_classes, k, temp)
            for w in W_GRID:
                f1 = _macro_f1(val_yt, fuse(val_p, S, w).argmax(1), n_classes)
                if f1 > bestB_f1:
                    bestB_f1, bestB = f1, (k, temp, w)
    kB, tempB, wB = bestB
    print(f"  VAL pick: k={kB}, temp={tempB}, w={wB} (val macro-F1 {bestB_f1:.4f}).", flush=True)
    if wB in (0.0, 1.0):
        print(f"  NOTE: validation chose an endpoint w={wB} -> fusion adds nothing over "
              f"{'pure retrieval' if wB == 0.0 else 'the pure CNN'}.", flush=True)
    S_test_B, _ = retrieval_matrix(test_nb, classes_idx, n_classes, kB, tempB)
    S_real_B, near_real_B = retrieval_matrix(real_nb, classes_idx, n_classes, kB, tempB)
    B_ctrl = _metrics(test_yt, fuse(test_p, S_test_B, wB).argmax(1), class_names)
    fused_real = fuse(real_p, S_real_B, wB)
    B_real_yp = fused_real.argmax(1)
    B_real = _metrics(real_yt, B_real_yp, class_names)
    _write_row(os.path.join(args.out_dir, ROW_B_DIR), ROW_B_DIR, B_ctrl, B_real,
               class_names, real_yt, B_real_yp,
               {"val_macro_f1": bestB_f1, "params": {"k": kB, "temp": tempB, "w": wB}})
    print(f"  REAL-WORLD macro-F1 {B_real['macro_f1']:.4f} (acc {B_real['accuracy']:.4f}).",
          flush=True)

    # ========================= Row B-abs: fusion + abstention =========================
    print("\n=== Row B-abs: fusion with abstention ===", flush=True)
    # Fixed fusion params from Row B; select only tau, d_max on validation.
    S_val_B, near_val_B = retrieval_matrix(val_nb, classes_idx, n_classes, kB, tempB)
    fused_val = fuse(val_p, S_val_B, wB)
    dmax_cands = [float(np.percentile(near_val_B, p)) for p in DMAX_PCTILES]
    bestBA, bestBA_f1 = None, -1.0
    for tau in TAU_GRID:
        for d_max in dmax_cands:
            ab = abstain_mask(fused_val, near_val_B, tau, d_max)
            yp = np.where(ab, -1, fused_val.argmax(1))
            f1 = _macro_f1(val_yt, yp, n_classes)   # abstain counts as error
            if f1 > bestBA_f1:
                bestBA_f1, bestBA = f1, (tau, d_max)
    tau, d_max = bestBA
    cov_val = float((~abstain_mask(fused_val, near_val_B, tau, d_max)).mean())
    print(f"  VAL pick: tau={tau}, d_max={d_max:.4f} (val macro-F1 {bestBA_f1:.4f}, "
          f"coverage {cov_val:.3f}).", flush=True)

    # freeze -> real-world once (reuse Row B's fused_real / near_real_B)
    ab_real = abstain_mask(fused_real, near_real_B, tau, d_max)
    BA_real_yp = np.where(ab_real, -1, fused_real.argmax(1))
    coverage = float((~ab_real).mean())
    covered = ~ab_real
    acc_covered = (float((BA_real_yp[covered] == real_yt[covered]).mean())
                   if covered.any() else 0.0)
    BA_real = _metrics(real_yt, BA_real_yp, class_names)     # full set, abstain = error
    BA_real = _fix_abstain_macro(BA_real, real_yt, BA_real_yp, n_classes)
    # controlled test with the same abstention rule
    test_fused = fuse(test_p, S_test_B, wB)
    _, near_test_B = retrieval_matrix(test_nb, classes_idx, n_classes, kB, tempB)
    ab_test = abstain_mask(test_fused, near_test_B, tau, d_max)
    BA_test_yp = np.where(ab_test, -1, test_fused.argmax(1))
    BA_ctrl = _metrics(test_yt, BA_test_yp, class_names)
    BA_ctrl = _fix_abstain_macro(BA_ctrl, test_yt, BA_test_yp, n_classes)
    _write_row(os.path.join(args.out_dir, ROW_BA_DIR), ROW_BA_DIR, BA_ctrl, BA_real,
               class_names, real_yt, BA_real_yp,
               {"val_macro_f1": bestBA_f1,
                "params": {"k": kB, "temp": tempB, "w": wB, "tau": tau, "d_max": d_max},
                "coverage": coverage, "accuracy_on_covered": acc_covered})
    print(f"  REAL-WORLD macro-F1 {BA_real['macro_f1']:.4f} (abstain=error) | "
          f"coverage {coverage:.3f} | acc-on-covered {acc_covered:.4f}.", flush=True)

    # ========================= 9. Novelty-signal AUROC (sec 7) =========================
    print("\n=== Secondary finding: retrieval distance as a novelty signal ===", flush=True)
    cnn_real_pred = real_p.argmax(1)
    cnn_wrong = (cnn_real_pred != real_yt).astype(int)
    near_nearest = np.array([nb[0][0] for nb in real_nb], dtype="float32")  # k=1 distance
    if cnn_wrong.sum() in (0, len(cnn_wrong)):
        auroc = float("nan")
        print("  AUROC undefined (CNN was right or wrong on every image).", flush=True)
    else:
        auroc = float(roc_auc_score(cnn_wrong, near_nearest))
    mean_correct = float(near_nearest[cnn_wrong == 0].mean()) if (cnn_wrong == 0).any() else None
    mean_wrong = float(near_nearest[cnn_wrong == 1].mean()) if (cnn_wrong == 1).any() else None
    print(f"  nearest-distance mean: CNN-correct {mean_correct} vs CNN-wrong {mean_wrong}",
          flush=True)
    print(f"  AUROC (nearest distance predicts CNN error): {auroc:.4f}", flush=True)

    # ========================= 10. Verdict + summary =========================
    seed42 = _load("efficientnetb0_on_bgrand_real", "eval_results_real_world.json")
    seed42_f1 = float(seed42["macro_f1"]) if seed42 else None
    best_row_f1 = max(A_real["macro_f1"], B_real["macro_f1"], BA_real["macro_f1"])
    delta_vs_mean = best_row_f1 - BASELINE_MEAN
    adopted = delta_vs_mean > ADOPT_THRESHOLD
    verdict = ("ADOPT: fusion clears the pre-registered +0.03 rule."
               if adopted else
               "NEGATIVE (as pre-registered): best row does not beat baseline by >+0.03. "
               "Report as a tried-and-bounded negative; the headline result does not depend on it.")
    print("\n=== VERDICT ===", flush=True)
    print(f"  baseline (3-seed mean)      : {BASELINE_MEAN:.4f} +/- {BASELINE_STD:.4f}", flush=True)
    if seed42_f1 is not None:
        print(f"  baseline (seed-42, this ckpt): {seed42_f1:.4f}", flush=True)
    print(f"  Row A retrieval  real-world : {A_real['macro_f1']:.4f}", flush=True)
    print(f"  Row B fusion     real-world : {B_real['macro_f1']:.4f}", flush=True)
    print(f"  Row B-abs        real-world : {BA_real['macro_f1']:.4f} "
          f"(coverage {coverage:.3f})", flush=True)
    print(f"  best - baseline_mean        : {delta_vs_mean:+.4f}  (threshold +{ADOPT_THRESHOLD})",
          flush=True)
    print(f"  {verdict}", flush=True)

    summary = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dino_model": args.dino_model, "seed": args.seed, "n_classes": n_classes,
        "baseline": {"three_seed_mean": BASELINE_MEAN, "three_seed_std": BASELINE_STD,
                     "seed42_this_ckpt": seed42_f1},
        "calibration": {"temperature": T, "ece_raw": ece_raw, "ece_calibrated": ece_cal},
        "row_A_retrieval": {"params": {"k": kA, "temp": tempA}, "val_macro_f1": bestA_f1,
                            "real_world_macro_f1": A_real["macro_f1"],
                            "real_world_accuracy": A_real["accuracy"]},
        "row_B_fusion": {"params": {"k": kB, "temp": tempB, "w": wB}, "val_macro_f1": bestB_f1,
                         "real_world_macro_f1": B_real["macro_f1"],
                         "real_world_accuracy": B_real["accuracy"]},
        "row_B_abstain": {"params": {"k": kB, "temp": tempB, "w": wB, "tau": tau, "d_max": d_max},
                          "val_macro_f1": bestBA_f1,
                          "real_world_macro_f1": BA_real["macro_f1"],
                          "coverage": coverage, "accuracy_on_covered": acc_covered},
        "novelty_signal": {"auroc": auroc, "mean_nearest_correct": mean_correct,
                           "mean_nearest_wrong": mean_wrong},
        "verdict": {"best_real_world_macro_f1": best_row_f1,
                    "delta_vs_baseline_mean": delta_vs_mean,
                    "adopt_threshold": ADOPT_THRESHOLD, "adopted": adopted, "text": verdict},
    }
    out_path = os.path.join(args.out_dir, "plan3_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}", flush=True)
    print(f"Wrote rows: {ROW_A_DIR}, {ROW_B_DIR}, {ROW_BA_DIR} under {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
