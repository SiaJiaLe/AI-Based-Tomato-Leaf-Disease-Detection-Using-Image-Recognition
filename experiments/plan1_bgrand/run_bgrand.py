"""Entry point for the Plan 1 background-randomization run.

    python -m experiments.plan1_bgrand.run_bgrand \
        --config experiments/plan1_bgrand/configs/efficientnetb0_on_bgrand.yaml
    python -m experiments.plan1_bgrand.run_bgrand --config ... --eval-only

Trains with background randomization, then evaluates on BOTH test sets using
the SHARED, unchanged common.evaluate.evaluate_run (bgrand is train-only, so
evaluation is identical to every ablation run).
"""
import argparse
import os

import torch
import yaml

from experiments.common.evaluate import evaluate_run

from .engine_bgrand import train_run_bgrand

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(REPO_ROOT, "experiments", "results")


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["data_dir"] = os.path.join(REPO_ROOT, cfg["data_dir"])
    cfg["real_world_dir"] = os.path.join(REPO_ROOT, cfg["real_world_dir"])
    bg = cfg["background_randomization"]
    bg["background_dir"] = os.path.join(REPO_ROOT, bg["background_dir"])
    return cfg


def _assert_backgrounds_clean(cfg: dict) -> None:
    """Guard the two hygiene rules that would silently invalidate the result."""
    bg = os.path.realpath(cfg["background_randomization"]["background_dir"])
    rw = os.path.realpath(cfg["real_world_dir"])
    if not os.path.isdir(bg):
        raise FileNotFoundError(
            f"background_dir does not exist: {bg}\n"
            "Generate it first: python -m experiments.plan1_bgrand.make_synthetic_backgrounds")
    # background_dir must be disjoint from the real-world test source (no leakage).
    if bg == rw or bg.startswith(rw + os.sep) or rw.startswith(bg + os.sep):
        raise ValueError(
            f"background_dir ({bg}) overlaps the real-world test dir ({rw}). "
            "Using field test images as backgrounds leaks the test domain into training.")
    n_imgs = len([f for f in os.listdir(bg)
                  if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"))])
    if n_imgs == 0:
        raise FileNotFoundError(f"No background images in {bg}.")
    print(f"Backgrounds OK: {n_imgs} images in {bg} (disjoint from real-world test).", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the bgrand run config YAML.")
    parser.add_argument("--eval-only", action="store_true", help="Skip training; only evaluate.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    cfg = load_config(args.config)
    results_dir = os.path.join(RESULTS_DIR, cfg["run_name"])
    print(f"\n===== {cfg['run_name']} =====", flush=True)

    if not args.eval_only:
        _assert_backgrounds_clean(cfg)
        train_run_bgrand(cfg, results_dir, device)
    evaluate_run(results_dir, device)


if __name__ == "__main__":
    main()
