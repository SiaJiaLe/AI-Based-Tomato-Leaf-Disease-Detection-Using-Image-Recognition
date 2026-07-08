"""Entry point for one ablation run (or all of them).

    python -m experiments.run --config experiments/configs/resnet34_on.yaml
    python -m experiments.run --all                 # train+eval every config
    python -m experiments.run --config ... --eval-only   # re-evaluate only

One config = one run = one results directory under experiments/results/. The
config's data_dir / real_world_dir are resolved relative to the repo root so
the same YAML works on any machine.
"""
import argparse
import glob
import os

import torch
import yaml

from experiments.common.engine import train_run
from experiments.common.evaluate import evaluate_run

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(REPO_ROOT, "experiments", "configs")
RESULTS_DIR = os.path.join(REPO_ROOT, "experiments", "results")


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Resolve dataset paths relative to the repo root.
    cfg["data_dir"] = os.path.join(REPO_ROOT, cfg["data_dir"])
    cfg["real_world_dir"] = os.path.join(REPO_ROOT, cfg["real_world_dir"])
    return cfg


def process(config_path: str, device, eval_only: bool):
    cfg = load_config(config_path)
    results_dir = os.path.join(RESULTS_DIR, cfg["run_name"])
    print(f"\n===== {cfg['run_name']} =====", flush=True)
    if not eval_only:
        train_run(cfg, results_dir, device)
    evaluate_run(results_dir, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to a single run config YAML.")
    parser.add_argument("--all", action="store_true", help="Run every config in experiments/configs/.")
    parser.add_argument("--eval-only", action="store_true", help="Skip training; only evaluate.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    if args.all:
        configs = sorted(glob.glob(os.path.join(CONFIG_DIR, "*.yaml")))
        failed = []
        for path in configs:
            try:
                process(path, device, args.eval_only)
            except Exception as exc:  # keep the batch going; report at the end
                print(f"!! {os.path.basename(path)} FAILED: {exc}", flush=True)
                failed.append(os.path.basename(path))
        if failed:
            print(f"\nFAILED runs: {failed}", flush=True)
        else:
            print("\nAll runs completed.", flush=True)
    elif args.config:
        process(args.config, device, args.eval_only)
    else:
        parser.error("Provide --config <path> or --all.")


if __name__ == "__main__":
    main()
