# Plan 1 — Background-Randomization Run (EfficientNetB0)

Adds ONE new experiment row, `efficientnetb0_on_bgrand` = the existing
`efficientnetb0_on` Stack-ON recipe **plus** background randomization, to test
whether replacing PlantVillage's uniform backdrops with varied backgrounds
during training shrinks the real-world generalization gap. See
`../../plan1_background_randomization.md` (spec) and the Plan 1 section of
`../../implementation_plan.md` (design).

## Isolation guarantee
This package only **imports** from `experiments/common/*` — it never edits it.
The 12 ablation runs, their configs, `run.py`, and `compare.py` are untouched.
The single new variable vs `efficientnetb0_on` is the `BackgroundRandomize`
transform prepended to the **train** pipeline. Val/test/real-world are the same
augmentation-free eval transform as every other run.

## How it works
- Training source `data/processed/train` is **unchanged**.
- Each train image, on the fly: with probability `prob` (0.5) the leaf is
  segmented and composited onto a random background; with probability `1-prob`
  the original PlantVillage image passes through untouched. So the model learns
  **both** originals and composites — no image copying needed.
- Segmentation keys on the leaf (not on green), so brown/yellow lesions stay
  inside the mask. Default route is **rembg (U^2-Net)** — robust on the
  textured/shadowed backdrops this dataset actually has; `hsv_threshold` is the
  no-download classical fallback.
- The mask is **eroded inward** (`mask_erode_px`) before compositing so no
  original backdrop leaks through as a halo (the artifact that hurt the first
  classical run).

## Files
| File | Purpose |
|---|---|
| `bg_randomize.py` | `BackgroundRandomize` transform + `segment_leaf` / `composite`. |
| `data_bgrand.py` | Train loader with bgrand prepended; val loader from shared eval transform. |
| `engine_bgrand.py` | Two-stage training (reuses `common.engine._run_epoch`); same checkpoints/metrics. |
| `run_bgrand.py` | Entry point; asserts backgrounds exist and are disjoint from real-world; then `common.evaluate.evaluate_run`. |
| `compare_bgrand.py` | `efficientnetb0_on` vs `efficientnetb0_on_bgrand` delta + per-class real-world F1. |
| `sanity_check_masks.py` | Saves original/mask/composite grids to inspect segmentation first. |
| `make_synthetic_backgrounds.py` | Generates ~60 domain-neutral textures into `data/backgrounds_generic/`. |
| `configs/efficientnetb0_on_bgrand.yaml` | The run config. |
| `run_bgrand_slurm.sh` | HPC job: backgrounds → sanity grids → train+eval → compare. |

## Run it (HPC)
```bash
git pull
# ONE-TIME, ON THE LOGIN NODE (needs internet) — installs rembg + downloads U^2-Net:
pip install -r experiments/plan1_bgrand/requirements.txt
python -c "from rembg import new_session; new_session('u2net')"
# then one GPU batch job does everything:
sbatch experiments/plan1_bgrand/run_bgrand_slurm.sh
```
No internet on the compute nodes? Do the two login-node steps above first, or
set `segmentation: hsv_threshold` in the config to skip rembg entirely.

## Run it (manual / debugging)
```bash
python -m experiments.plan1_bgrand.make_synthetic_backgrounds   # once
python -m experiments.plan1_bgrand.sanity_check_masks           # eyeball masks
python -m experiments.plan1_bgrand.run_bgrand \
    --config experiments/plan1_bgrand/configs/efficientnetb0_on_bgrand.yaml
python -m experiments.plan1_bgrand.compare_bgrand
```

Re-evaluate without retraining: add `--eval-only` to `run_bgrand`.

## Reading the result
- **Real-world macro-F1 ↑ and gap ↓** → background shortcut confirmed; keep it.
- **Lab ↓ but real-world ↑** → ideal domain-shift story (gave up a lab shortcut
  for real robustness).
- **Both flat** → check mask/composite quality (sanity grids) before concluding.

## Tuning `prob` (optional, on VALIDATION only)
Edit `prob` in the config (`0.3` / `0.5` / `0.7`), pick the best on PlantVillage
val macro-F1, then read real-world once. Never tune on real-world.

## Backgrounds
Synthetic by default. Swap in real CC0 textures later by dropping image files
into `data/backgrounds_generic/` — see that folder's README.
