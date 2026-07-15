"""Pick the drop_path_rate winner using PlantVillage VALIDATION macro-F1 only.

    python -m experiments.plan2_arch.select_on_val \
        efficientnetb0_on_droppath02 efficientnetb0_on_droppath03

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


def _val_f1(run_name):
    path = os.path.join(RESULTS_DIR, run_name, "metrics.json")
    if not os.path.isfile(path):
        return None, None
    with open(path) as f:
        m = json.load(f)
    rate = m.get("architecture_mod", {}).get("drop_path_rate")
    return m.get("best_val_macro_f1"), rate


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

    print("\n=== Tier 1 selection — PlantVillage VAL macro-F1 (real-world NOT read) ===\n")
    print(f"  {'run':40} {'drop_path':>10} {'val_macro_f1':>14}")
    print("  " + "-" * 66)

    base_f1, _ = _val_f1(args.baseline)
    if base_f1 is not None:
        print(f"  {args.baseline + ' (baseline)':40} {'-':>10} {base_f1:>14.4f}")

    scored = []
    for run in args.runs:
        f1, rate = _val_f1(run)
        if f1 is None:
            print(f"  {run:40} {'-':>10} {'MISSING':>14}")
            continue
        print(f"  {run:40} {str(rate):>10} {f1:>14.4f}")
        scored.append((f1, run, rate))

    if not scored:
        raise SystemExit("\nNo candidate metrics.json found — train the sweep first.")

    scored.sort(reverse=True)
    best_f1, best_run, best_rate = scored[0]
    print(f"\n  Winner on val: {best_run} (drop_path_rate={best_rate}, val macro-F1 {best_f1:.4f})")
    if base_f1 is not None:
        print(f"  vs baseline val macro-F1 {base_f1:.4f} ({best_f1 - base_f1:+.4f})")
    print(f"\n  Next — read the test sets ONCE for the winner:\n"
          f"    python -m experiments.plan2_arch.run_arch \\\n"
          f"      --config experiments/plan2_arch/configs/{best_run}.yaml --eval-only\n"
          f"    python -m experiments.plan2_arch.compare_arch --run {best_run}\n")


if __name__ == "__main__":
    main()
