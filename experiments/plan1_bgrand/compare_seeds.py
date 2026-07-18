"""Aggregate the 3-seed replication (Plan 6) into mean +/- std, isolated from
compare.py / compare_bgrand.py.

The 3 x 3 matrix: recipe {baseline, synthetic, real} x seed {42, 43, 44}. Seed-42
synthetic/real are the existing efficientnetb0_on_bgrand / _bgrand_real runs; every
other cell is a seedrep run. All nine go through run_bgrand -> evaluate_run on the
cleaned real set, so the only variable is the background treatment.

Turns the single-seed headline into:
  * per-recipe real-world macro-F1 and accuracy, mean +/- sample std (n=3);
  * the two contrasts PAIRED BY SEED -- real - baseline and real - synthetic (the
    realism contrast) -- with mean +/- std and whether all seeds agree in direction;
  * controlled macro-F1 per recipe as a sanity column (should stay flat).

    python -m experiments.plan1_bgrand.compare_seeds

Read-once: only reads cells that already have a published eval_results_real_world.json.
Missing cells (not trained yet) are reported, and stats fall back to whatever seeds
are present. ASCII-only output (compute-node C locale). Touches nothing else.
"""
import json
import os
import statistics

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(REPO_ROOT, "experiments", "results")

SEEDS = [42, 43, 44]

# recipe -> {seed: run_name}. Seed 42 synthetic/real reuse the existing runs.
MATRIX = {
    "baseline": {
        42: "efficientnetb0_seedrep_base_s42",
        43: "efficientnetb0_seedrep_base_s43",
        44: "efficientnetb0_seedrep_base_s44",
    },
    "synthetic": {
        42: "efficientnetb0_on_bgrand",
        43: "efficientnetb0_seedrep_bgrand_s43",
        44: "efficientnetb0_seedrep_bgrand_s44",
    },
    "real": {
        42: "efficientnetb0_on_bgrand_real",
        43: "efficientnetb0_seedrep_bgrandreal_s43",
        44: "efficientnetb0_seedrep_bgrandreal_s44",
    },
}


def _load(run_name, real_world):
    fname = "eval_results_real_world.json" if real_world else "eval_results.json"
    path = os.path.join(RESULTS_DIR, run_name, fname)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _mean_std(vals):
    """(mean, sample_std) over present values; std is None if fewer than 2."""
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    mean = statistics.fmean(vals)
    std = statistics.stdev(vals) if len(vals) >= 2 else None
    return mean, std


def _fmt(mean, std):
    if mean is None:
        return "n/a"
    if std is None:
        return f"{mean:.4f} (1 seed)"
    return f"{mean:.4f} +/- {std:.4f}"


def _cell_metric(run_name, real_world, key):
    d = _load(run_name, real_world)
    return d.get(key) if d else None


def main():
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("\n=== Plan 6: 3-seed replication (real backgrounds vs baseline vs synthetic) ===")
    out("All cells on the cleaned real set. Seed-42 synthetic/real reuse the existing runs.\n")

    # --- per-cell table + collect metrics ---
    rw_f1 = {r: {} for r in MATRIX}      # recipe -> seed -> real-world macro_f1
    rw_acc = {r: {} for r in MATRIX}
    ctrl_f1 = {r: {} for r in MATRIX}
    missing = []

    out(f"  {'recipe':10} {'seed':>4}  {'run_name':40} {'rw_f1':>8} {'rw_acc':>8} {'ctrl_f1':>8}")
    out("  " + "-" * 84)
    for recipe in ("baseline", "synthetic", "real"):
        for seed in SEEDS:
            run = MATRIX[recipe][seed]
            f1 = _cell_metric(run, True, "macro_f1")
            acc = _cell_metric(run, True, "accuracy")
            cf1 = _cell_metric(run, False, "macro_f1")
            rw_f1[recipe][seed] = f1
            rw_acc[recipe][seed] = acc
            ctrl_f1[recipe][seed] = cf1
            if f1 is None:
                missing.append(run)
                out(f"  {recipe:10} {seed:>4}  {run:40} {'MISSING':>8} {'':>8} {'':>8}")
            else:
                cf = f"{cf1:.4f}" if cf1 is not None else "n/a"
                out(f"  {recipe:10} {seed:>4}  {run:40} {f1:>8.4f} {acc:>8.4f} {cf:>8}")

    # --- per-recipe mean +/- std ---
    out("\n  Per-recipe real-world macro-F1 / accuracy (mean +/- sample std over seeds):")
    out(f"  {'recipe':10} {'rw_macro_f1':>22} {'rw_accuracy':>22} {'ctrl_macro_f1':>18}")
    out("  " + "-" * 74)
    recipe_summary = {}
    for recipe in ("baseline", "synthetic", "real"):
        f1m, f1s = _mean_std([rw_f1[recipe][s] for s in SEEDS])
        am, as_ = _mean_std([rw_acc[recipe][s] for s in SEEDS])
        cm, _ = _mean_std([ctrl_f1[recipe][s] for s in SEEDS])
        recipe_summary[recipe] = {"rw_macro_f1_mean": f1m, "rw_macro_f1_std": f1s,
                                  "rw_acc_mean": am, "rw_acc_std": as_,
                                  "ctrl_macro_f1_mean": cm}
        cm_str = f"{cm:.4f}" if cm is not None else "n/a"
        out(f"  {recipe:10} {_fmt(f1m, f1s):>22} {_fmt(am, as_):>22} {cm_str:>18}")

    # --- paired contrasts (by seed) ---
    def contrast(a, b, label):
        """a - b per seed present in both; print values + mean/std + agreement."""
        per = []
        for s in SEEDS:
            va, vb = rw_f1[a][s], rw_f1[b][s]
            if va is not None and vb is not None:
                per.append((s, va - vb))
        out(f"\n  {label} (real-world macro-F1, paired by seed):")
        if not per:
            out("    no seed has both cells yet.")
            return None
        for s, d in per:
            out(f"    seed {s}: {d:+.4f}")
        m, sd = _mean_std([d for _, d in per])
        agree = "all agree (+)" if all(d > 0 for _, d in per) else (
                "all agree (-)" if all(d < 0 for _, d in per) else "MIXED sign")
        out(f"    mean {_fmt(m, sd)}   [{len(per)} seed(s), {agree}]")
        return {"per_seed": {s: d for s, d in per}, "mean": m, "std": sd, "agreement": agree}

    real_vs_base = contrast("real", "baseline", "REAL - BASELINE")
    real_vs_syn = contrast("real", "synthetic", "REAL - SYNTHETIC (the realism contrast)")

    if missing:
        out(f"\n  NOTE  {len(missing)} cell(s) not trained yet: {', '.join(missing)}")
        out("        Stats above use only the seeds present. Re-run after the array finishes.")
    else:
        out("\n  All 9 cells present. Report the per-recipe mean +/- std and the paired")
        out("  contrasts above; with n=3 this is descriptive (direction + spread), not a p-value.")

    # --- machine-readable summary ---
    summary = {"matrix": MATRIX, "seeds": SEEDS, "recipe_summary": recipe_summary,
               "real_minus_baseline": real_vs_base, "real_minus_synthetic": real_vs_syn,
               "missing_cells": missing}
    out_path = os.path.join(RESULTS_DIR, "seedrep_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    out(f"\n  Wrote {out_path}")


if __name__ == "__main__":
    main()
