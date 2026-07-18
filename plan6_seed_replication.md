# Plan 6 — 3-seed replication of the realism result

## Goal
Turn the single-seed (42) headline into a mean ± spread across 3 seeds, so the
"real backgrounds beat baseline AND beat synthetic backgrounds" claim is defensible
rather than n=1. Decisions locked with the user:
- **Recipes:** all three — baseline, synthetic (`bgrand`), real (`bgrand_real`).
- **Seeds:** 42, 43, 44 (3 total).
- **Baseline:** same code path, background randomization disabled via `prob: 0.0`
  (NOT the run.py `efficientnetb0_on`), so the only variable across the matrix is
  the background treatment.

## The 3 x 3 matrix (9 cells, 7 new)
| recipe \ seed | 42 | 43 | 44 |
|---|---|---|---|
| baseline (prob 0) | **NEW** base_s42 | **NEW** base_s43 | **NEW** base_s44 |
| synthetic (bgrand) | reuse `efficientnetb0_on_bgrand` | **NEW** bgrand_s43 | **NEW** bgrand_s44 |
| real (bgrand_real) | reuse `efficientnetb0_on_bgrand_real` | **NEW** bgrandreal_s43 | **NEW** bgrandreal_s44 |

Seed-42 synthetic/real already exist as `run_bgrand` outputs on the cleaned 333-image
set (refreshed 2026-07-18) — re-running them would be identical, so they are reused,
not retrained. **7 new training runs.**

## Why prob:0 is a correct baseline
`data_bgrand.build_train_transform_bgrand` always prepends `BackgroundRandomize`,
but it is an `A.ImageOnlyTransform` with `p=prob`. At `prob=0.0` albumentations never
calls `apply()`, so no segmentation and no compositing happen — the effective train
transform is exactly `_basic_four + _advanced_block + Normalize + ToTensor`, i.e. the
EfficientNetB0 Stack-ON pipeline. All three recipes then differ only in
`background_randomization` (`prob`, `background_dir`), which is the one variable we
want to isolate. (The constructor still loads `background_dir` into memory even at
prob 0, so the config must point at a valid, disjoint folder — we use
`data/backgrounds_generic_real`; it is never sampled.)

## Files to CREATE (all additive; isolation contract preserved)
New folder `experiments/plan1_bgrand/configs/seedrep/` with 7 YAMLs, each a copy of
`efficientnetb0_on_bgrand_real.yaml` changing only `run_name`, `seed`, and the
`background_randomization` block:

- `base_s42.yaml`, `base_s43.yaml`, `base_s44.yaml` — `prob: 0.0` (baseline)
- `bgrand_s43.yaml`, `bgrand_s44.yaml` — synthetic: `background_dir:
  data/backgrounds_generic_synthetic`, `prob: 0.5` (copy of existing bgrand, new seed)
- `bgrandreal_s43.yaml`, `bgrandreal_s44.yaml` — real: `background_dir:
  data/backgrounds_generic_real`, `prob: 0.5` (copy of existing bgrand_real, new seed)

`run_name`s: `efficientnetb0_seedrep_<recipe>_s<seed>` so results land in distinct
`experiments/results/` dirs and never collide with existing runs. Everything else
(stack, training budget, segmentation, mask_cache_dir, real_world_dir) is byte-for-byte
the existing recipe, so seed is the only within-recipe difference.

Also create:
- `experiments/plan1_bgrand/run_seedrep_slurm.sh` — a SLURM **job array** (7 tasks,
  one config each, `gpu-24c-l4-4g`, `gres gpu:l4:1`, ~1h wall each). An array so a
  single wall clock doesn't have to hold all 7 sequentially. Each task:
  `python -m experiments.plan1_bgrand.run_bgrand --config <the task's yaml>`.
  Preflight (`preflight_env`) + a background-set check run once at task 0.
- `experiments/plan1_bgrand/compare_seeds.py` — reads
  `eval_results_real_world.json` + `eval_results.json` for all 9 cells (existing +
  new), groups by recipe, and prints:
  - real-world macro-F1 and accuracy per cell, then **mean ± sample-std (n=3)** per
    recipe;
  - the two contrasts **per seed** and their mean ± std: `real - baseline` and
    `real - synthetic` (the +0.0365 realism contrast, now replicated);
  - a direction-agreement line (do all 3 seeds put real > baseline? real > synthetic?);
  - controlled (PlantVillage) macro-F1 per recipe as a sanity column (should stay
    flat — if a seed's controlled number moves a lot, that run is suspect).
  Read-once discipline unchanged: it only reads rows that have a published
  `eval_results_real_world.json`. ASCII-only output (compute-node C locale).

## Compute
7 runs. `bgrand`/`bgrand_real` seeds 43/44 reuse the existing `data/mask_cache/`
(masks keyed by leaf image, so U^2-Net is not re-run); baseline does no segmentation.
So each run is ~40 epochs of plain training, ~40-60 min on an L4. As a 7-task array
that is ~1-2h wall if they schedule in parallel, well within limits.

## Reporting / traps to avoid
- Report the matrix as **mean ± std over 3 seeds**, plus "all 3 seeds agree in
  direction" if they do. With n=3 this is descriptive rigor, not a significance test;
  do not claim a p-value.
- **Do NOT mix** the seedrep baseline (prob-0, run_bgrand) with the main-study
  `efficientnetb0_on` (run.py) in one contrast — they are different code paths/RNG.
  The realism contrast is computed entirely WITHIN the seedrep matrix. The main-study
  `efficientnetb0_on` stays the baseline for the ablation/architecture tables only.
- All 9 cells are on the cleaned 333-image real set, so they are directly comparable
  to the confusion analysis just generated.

## Sequence
1. Create the 7 configs + the 2 scripts (this plan).
2. Offline test of `compare_seeds.py` on stub JSON (mean/std/contrast maths, ASCII,
   read-once) before any GPU spend.
3. Commit + push. On HPC: `git pull`, confirm `data/backgrounds_generic_synthetic`
   and `data/backgrounds_generic_real` exist, then `sbatch run_seedrep_slurm.sh`.
4. After the array finishes: `python -m experiments.plan1_bgrand.compare_seeds`, read
   the mean ± std table together.

## Open item to confirm before building
The synthetic seed-42 run's `background_dir`: the existing `efficientnetb0_on_bgrand.yaml`
must be read to copy its exact `background_dir` / `prob` / segmentation into the
s43/s44 synthetic configs (the table above assumes `data/backgrounds_generic_synthetic`
— will verify against the file, not guess).
