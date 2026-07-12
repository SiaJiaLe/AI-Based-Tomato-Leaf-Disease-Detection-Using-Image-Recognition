"""Compare the Plan 1 run against its baseline — isolated from compare.py.

Prints the delta between `efficientnetb0_on` (baseline) and
`efficientnetb0_on_bgrand` (background randomization added), on both the
controlled and real-world test sets, plus the per-class real-world F1 table.
The delta is attributable to background randomization alone.

    python -m experiments.plan1_bgrand.compare_bgrand

Does NOT touch experiments/compare.py or the master ablation table.
"""
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(REPO_ROOT, "experiments", "results")

BASELINE = "efficientnetb0_on"
BGRAND = "efficientnetb0_on_bgrand"


def _load(run_name, real_world):
    fname = "eval_results_real_world.json" if real_world else "eval_results.json"
    path = os.path.join(RESULTS_DIR, run_name, fname)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _row(label, base, bg, key):
    b = base.get(key) if base else None
    g = bg.get(key) if bg else None
    if b is None or g is None:
        return f"  {label:24} {'n/a':>10} {'n/a':>10} {'n/a':>10}"
    return f"  {label:24} {b:>10.4f} {g:>10.4f} {g-b:>+10.4f}"


def main():
    base_c, bg_c = _load(BASELINE, False), _load(BGRAND, False)
    base_r, bg_r = _load(BASELINE, True), _load(BGRAND, True)

    if bg_c is None:
        raise SystemExit(f"No results for {BGRAND}. Run the training first.")
    if base_c is None:
        print(f"WARNING: baseline {BASELINE} results not found — showing bgrand only.\n")

    print(f"\n=== Plan 1: {BGRAND}  vs  {BASELINE} ===\n")
    print(f"  {'metric':24} {'baseline':>10} {'bgrand':>10} {'delta':>10}")
    print("  " + "-" * 56)
    print(_row("controlled_accuracy", base_c, bg_c, "accuracy"))
    print(_row("controlled_macro_f1", base_c, bg_c, "macro_f1"))
    print(_row("realworld_accuracy", base_r, bg_r, "accuracy"))
    print(_row("realworld_macro_f1", base_r, bg_r, "macro_f1"))
    print(_row("gap_accuracy", base_r, bg_r, "generalization_gap_accuracy"))
    print(_row("gap_macro_f1", base_r, bg_r, "generalization_gap_macro_f1"))

    # Per-class real-world F1 (the failure-analysis view).
    if bg_r is not None:
        print("\n  Per-class REAL-WORLD F1:")
        print(f"  {'class':45} {'baseline':>10} {'bgrand':>10} {'delta':>10}")
        print("  " + "-" * 77)
        classes = [c for c in bg_r["classification_report"]
                   if c not in ("accuracy", "macro avg", "weighted avg")]
        for c in classes:
            g = bg_r["classification_report"][c]["f1-score"]
            b = (base_r["classification_report"][c]["f1-score"]
                 if base_r and c in base_r["classification_report"] else None)
            if b is None:
                print(f"  {c:45} {'n/a':>10} {g:>10.4f} {'n/a':>10}")
            else:
                print(f"  {c:45} {b:>10.4f} {g:>10.4f} {g-b:>+10.4f}")

    # Persist a small machine-readable summary next to the run.
    out = os.path.join(RESULTS_DIR, BGRAND, "compare_vs_baseline.json")
    summary = {
        "baseline": BASELINE, "bgrand": BGRAND,
        "controlled": {"baseline": base_c, "bgrand": bg_c},
        "real_world": {"baseline": base_r, "bgrand": bg_r},
    }
    if os.path.isdir(os.path.dirname(out)):
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
