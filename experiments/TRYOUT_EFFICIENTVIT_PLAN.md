# Plan — EfficientViT-B0 try-out (NOT part of the research)

Kept out of the platform `implementation_plan.md` (that file is the backend/platform v5
plan) so nothing there is clobbered.

## Goal
Train a single Vision Transformer (`efficientvit_b0`, timm MIT-EfficientViT) on the same
8-class tomato split, in **both** augmentation conditions (OFF/ON), and evaluate each on the
controlled PlantVillage test set and the real-world set. Exploratory only — not a thesis
result; no other model retrained.

## Isolation (additive only)
- Zero edits to `experiments/common/*`, `run.py`, `compare.py`, the ablation configs, or any
  study file.
- New package `experiments/tryout_efficientvit/` **imports** the shared modules read-only for
  parity: `common/data.py` (preprocessing + OFF/ON augmentation toggle),
  `common/evaluate.py` (`_metrics`, `_plot_confusion`), `common/seeding.py`.
- No config added under `experiments/configs/`, so `run.py --all` never picks it up.

## Why a bespoke trainer (not the shared engine)
`common/engine.py` fine-tunes via a CNN stage-group interface (`BuiltModel.groups`).
EfficientViT is a transformer with no such decomposition, so the try-out has its own small
two-phase fine-tune (head warm-up → full fine-tune) and leaves the study engine untouched.

## Training protocol (fixed: 224px, batch 32, seed 42, ImageNet norm, label_smoothing 0.1)
- Phase A: freeze all but the classifier head; AdamW lr 1e-3; 5 epochs.
- Phase B: unfreeze all; AdamW lr 1e-4, weight_decay 0.05, cosine schedule; ≤20 epochs;
  early-stop on val macro-F1 (patience 7); checkpoint best epoch (same selection as study).
- OFF/ON = `advanced_augmentation` false/true (the study's defining OFF/ON variable).

## Files added
- `experiments/tryout_efficientvit/__init__.py`
- `experiments/tryout_efficientvit/train_efficientvit.py` — train + eval CLI (`--config`, `--eval-only`)
- `experiments/tryout_efficientvit/config_off.yaml`, `config_on.yaml`
- `experiments/tryout_efficientvit/requirements.txt` — note: no new dependency (timm 1.0.27 has it)
- `experiments/TRYOUT_EFFICIENTVIT_COLAB.md` — Colab cell-by-cell guide
- `experiments/TRYOUT_EFFICIENTVIT_PLAN.md` — this file

## Outputs (per run, under `results/efficientvit_b0_{off,on}/`)
`best_model.pth`, `best_by_loss.pth`, `metrics.json`, `resolved_config.json`,
`eval_results.json`, `eval_results_real_world.json`, `cm_controlled_test.png`,
`cm_real_world_test.png` — same schema as every study run, marked `tryout: true`.

## Run
```
python -m experiments.tryout_efficientvit.train_efficientvit --config experiments/tryout_efficientvit/config_off.yaml
python -m experiments.tryout_efficientvit.train_efficientvit --config experiments/tryout_efficientvit/config_on.yaml
```
Colab: follow `experiments/TRYOUT_EFFICIENTVIT_COLAB.md`.
