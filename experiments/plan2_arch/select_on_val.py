"""Pick a sweep winner using PlantVillage VALIDATION macro-F1 only.

    # Tier 1 — drop-path rate
    python -m experiments.plan2_arch.select_on_val \
        efficientnetb0_on_droppath02 efficientnetb0_on_droppath03

    # Tier 3 — MixStyle insertion depth
    python -m experiments.plan2_arch.select_on_val \
        efficientnetb0_on_mixstyle_l12 efficientnetb0_on_mixstyle_l123

This script reads ONLY metrics.json (which contains val history and no test-set
numbers). It cannot see the real-world set even if it wanted to — that is the
point. Selecting a hyperparameter on the held-out real-world set would turn the
measurement into a contest, which the CP2 spec forbids.

It prints the winner's run name; feed that to run_arch.py --eval-only.
"""
import argparse
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(REPO_ROOT, "experiments", "results")


def _setting_of(mod):
    """Describe whichever architecture_mod this row varies.

    Read from the row's own metrics, never hardcoded — hardcoding one tier's key
    is exactly what made compare_arch.py label every row 'Tier 1'.
    """
    if "drop_path_rate" in mod:
        return str(mod["drop_path_rate"])
    if "input_resolution" in mod:
        return f"{mod['input_resolution']}px"
    if "mixstyle" in mod:
        return f"layers={mod['mixstyle']['layers']}"
    return "-"


def _val_f1(run_name):
    path = os.path.join(RESULTS_DIR, run_name, "metrics.json")
    if not os.path.isfile(path):
        return None, None
    with open(path) as f:
        m = json.load(f)
    return m.get("best_val_macro_f1"), _setting_of(m.get("architecture_mod", {}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="Candidate run names to compare on val.")
    parser.add_argument("--baseline", default="efficientnetb0_on",
                        help="Shown for context only; not eligible to win.")
    parser.add_argument("--print-winner", action="store_true",
                        help="Print only the winning run name (for shell scripts).")
    args = parser.parse_args()

    if args.print_winner:
        scored = [(f1, run) for run in args.runs
                  for f1, _ in [_val_f1(run)] if f1 is not None]
        if not scored:
            raise SystemExit("No candidate metrics.json found — train the sweep first.")
        print(max(scored)[1])
        return

    print("\n=== Sweep selection — PlantVillage VAL macro-F1 (real-world NOT read) ===\n")
    print(f"  {'run':40} {'setting':>16} {'val_macro_f1':>14}")
    print("  " + "-" * 72)

    base_f1, _ = _val_f1(args.baseline)
    if base_f1 is not None:
        print(f"  {args.baseline + ' (baseline)':40} {'-':>16} {base_f1:>14.4f}")

    scored = []
    for run in args.runs:
        f1, setting = _val_f1(run)
        if f1 is None:
            print(f"  {run:40} {'-':>16} {'MISSING':>14}")
            continue
        print(f"  {run:40} {setting:>16} {f1:>14.4f}")
        scored.append((f1, run, setting))

    if not scored:
        raise SystemExit("\nNo candidate metrics.json found — train the sweep first.")

    scored.sort(reverse=True)
    best_f1, best_run, best_setting = scored[0]
    print(f"\n  Winner on val: {best_run} ({best_setting}, val macro-F1 {best_f1:.4f})")
    if base_f1 is not None:
        print(f"  vs baseline val macro-F1 {base_f1:.4f} ({best_f1 - base_f1:+.4f})")
    print(f"\n  Next — read the test sets ONCE for the winner:\n"
          f"    python -m experiments.plan2_arch.run_arch \\\n"
          f"      --config experiments/plan2_arch/configs/{best_run}.yaml --eval-only\n"
          f"    python -m experiments.plan2_arch.compare_arch --run {best_run}\n")


if __name__ == "__main__":
    main()
