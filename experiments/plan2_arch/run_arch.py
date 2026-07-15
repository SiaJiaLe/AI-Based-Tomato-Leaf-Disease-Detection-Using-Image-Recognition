"""Entry point for Plan 2 architecture rows (Tier 1: drop-path).

    # train a sweep member WITHOUT touching the real-world test set
    python -m experiments.plan2_arch.run_arch --config <cfg> --train-only

    # after selecting the winner on val, read the test sets ONCE
    python -m experiments.plan2_arch.run_arch --config <cfg> --eval-only

    # train + evaluate in one go (single-row mode)
    python -m experiments.plan2_arch.run_arch --config <cfg>

`--train-only` exists to protect the hygiene rule: while sweeping drop_path_rate
we must select on PlantVillage val macro-F1 only. Evaluating every sweep member
would read the held-out real-world set more than once per row, which is exactly
what the CP2 spec forbids. So sweep -> select_on_val.py -> --eval-only winner.

Evaluation uses the SHARED, unchanged common.evaluate.evaluate_run: drop-path is
parameter-free and identity in eval(), so the checkpoint loads into the plain
builder and the network measured is exactly the network trained.
"""
import argparse
import os

import torch
import yaml

from experiments.common.evaluate import evaluate_run

from .engine_arch import train_run_arch
from .evaluate_res import evaluate_run_res

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(REPO_ROOT, "experiments", "results")


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["data_dir"] = os.path.join(REPO_ROOT, cfg["data_dir"])
    cfg["real_world_dir"] = os.path.join(REPO_ROOT, cfg["real_world_dir"])
    return cfg


def _assert_one_variable(cfg: dict) -> None:
    """Guard the one-variable rule: this row must be the baseline stack plus
    drop-path and nothing else. A stray `background_randomization` block would
    silently bundle Plan 1 into a Plan 2 row and destroy attribution."""
    if "background_randomization" in cfg:
        raise ValueError(
            "This config contains a background_randomization block. Plan 2 tiers are "
            "standalone rows against efficientnetb0_on — they must NOT bundle Plan 1.")
    baseline_stack = {"advanced_augmentation": True, "label_smoothing": 0.1,
                      "strong_head": True, "cbam": True, "stage_b": "two_group"}
    if cfg["stack"] != baseline_stack:
        raise ValueError(
            f"stack differs from the efficientnetb0_on baseline.\n"
            f"  expected: {baseline_stack}\n  got:      {cfg['stack']}\n"
            "Tier rows must change only architecture_mod.")
    mod = cfg.get("architecture_mod", {})
    known = {"drop_path_rate", "input_resolution"}  # Tier 1, Tier 2
    unknown = set(mod) - known
    if unknown:
        raise ValueError(f"Unknown architecture_mod key(s): {sorted(unknown)}. Known: {sorted(known)}.")
    if len(mod) != 1:
        raise ValueError(
            f"A tier row must set EXACTLY ONE architecture_mod key; got {sorted(mod)}. "
            "Tiers are standalone — to test a combination, make it an explicit separate row.")
    (key, value), = mod.items()
    print(f"One-variable check OK: baseline stack + {key}={value}.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the Plan 2 row config YAML.")
    parser.add_argument("--train-only", action="store_true",
                        help="Train only; do NOT read the test sets (use while sweeping on val).")
    parser.add_argument("--eval-only", action="store_true", help="Skip training; only evaluate.")
    args = parser.parse_args()

    if args.train_only and args.eval_only:
        raise SystemExit("--train-only and --eval-only are mutually exclusive.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    cfg = load_config(args.config)
    results_dir = os.path.join(RESULTS_DIR, cfg["run_name"])
    print(f"\n===== {cfg['run_name']} =====", flush=True)

    if not args.eval_only:
        _assert_one_variable(cfg)
        train_run_arch(cfg, results_dir, device)
    if not args.train_only:
        # Resolution rows MUST evaluate at their training resolution —
        # common.evaluate hardcodes 224, so reusing it for a 240 model would be
        # a silent train/eval preprocessing mismatch.
        resolution = int(cfg.get("architecture_mod", {}).get("input_resolution", 224))
        if resolution != 224:
            evaluate_run_res(results_dir, device)
        else:
            evaluate_run(results_dir, device)
    else:
        print("Trained only — test sets untouched. Select on val, then re-run with --eval-only.",
              flush=True)


if __name__ == "__main__":
    main()
