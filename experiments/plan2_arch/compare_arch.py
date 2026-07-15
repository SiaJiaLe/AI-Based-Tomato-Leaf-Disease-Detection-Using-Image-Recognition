"""Compare a Plan 2 architecture row against the fixed baseline.

    python -m experiments.plan2_arch.compare_arch --run efficientnetb0_on_droppath02

Prints controlled / real-world / gap deltas plus the per-class real-world F1
table. The delta is attributable to the single architecture_mod alone, because
the split, seed, budget, and solution stack are identical to the baseline.

Judge on the REAL-WORLD macro-F1 and gap rows — lab accuracy is a sanity check,
not the target (plan2 §5).

Isolated: does NOT touch experiments/compare.py or the master ablation table.
"""
import argparse
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(REPO_ROOT, "experiments", "results")

BASELINE = "efficientnetb0_on"


def _load(run_name, real_world):
    fname = "eval_results_real_world.json" if real_world else "eval_results.json"
    path = os.path.join(RESULTS_DIR, run_name, fname)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _row(label, base, run, key):
    b = base.get(key) if base else None
    g = run.get(key) if run else None
    if b is None or g is None:
        return f"  {label:24} {'n/a':>10} {'n/a':>10} {'n/a':>10}"
    return f"  {label:24} {b:>10.4f} {g:>10.4f} {g-b:>+10.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default=BASELINE, help="Baseline run name.")
    parser.add_argument("--run", required=True, help="Plan 2 row to compare (e.g. efficientnetb0_on_droppath02).")
    args = parser.parse_args()
    baseline, run = args.baseline, args.run

    base_c, run_c = _load(baseline, False), _load(run, False)
    base_r, run_r = _load(baseline, True), _load(run, True)

    if run_c is None:
        raise SystemExit(f"No results for {run}. Train and evaluate it first.")
    if base_c is None:
        print(f"WARNING: baseline {baseline} results not found — showing {run} only.\n")

    print(f"\n=== Plan 2 Tier 1: {run}  vs  {baseline} ===\n")
    print(f"  {'metric':24} {'baseline':>10} {'tier1':>10} {'delta':>10}")
    print("  " + "-" * 56)
    print(_row("controlled_accuracy", base_c, run_c, "accuracy"))
    print(_row("controlled_macro_f1", base_c, run_c, "macro_f1"))
    print(_row("realworld_accuracy", base_r, run_r, "accuracy"))
    print(_row("realworld_macro_f1", base_r, run_r, "macro_f1"))
    print(_row("gap_accuracy", base_r, run_r, "generalization_gap_accuracy"))
    print(_row("gap_macro_f1", base_r, run_r, "generalization_gap_macro_f1"))

    if run_r is not None:
        print("\n  Per-class REAL-WORLD F1:")
        print(f"  {'class':45} {'baseline':>10} {'tier1':>10} {'delta':>10}")
        print("  " + "-" * 77)
        classes = [c for c in run_r["classification_report"]
                   if c not in ("accuracy", "macro avg", "weighted avg")]
        for c in classes:
            g = run_r["classification_report"][c]["f1-score"]
            b = (base_r["classification_report"][c]["f1-score"]
                 if base_r and c in base_r["classification_report"] else None)
            if b is None:
                print(f"  {c:45} {'n/a':>10} {g:>10.4f} {'n/a':>10}")
            else:
                print(f"  {c:45} {b:>10.4f} {g:>10.4f} {g-b:>+10.4f}")

    out = os.path.join(RESULTS_DIR, run, "compare_vs_baseline.json")
    summary = {
        "baseline": baseline, "run": run,
        "controlled": {"baseline": base_c, "run": run_c},
        "real_world": {"baseline": base_r, "run": run_r},
    }
    if os.path.isdir(os.path.dirname(out)):
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
