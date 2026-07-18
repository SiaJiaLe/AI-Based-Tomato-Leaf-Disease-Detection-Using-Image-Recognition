# Plan 8 — Re-split the new dataset and retrain everything (parallel)

## Context / what changed
User deleted repo-root `data/processed/{train,val,test}` and dropped a NEW dataset
(same 10 class folders) into repo-root `data/raw`. Wants to re-split into
train/val/test and retrain ALL models in parallel, like the seed array.

## Two traps found (must handle)
1. **`resnet34_model/scripts/prepare_dataset.py` targets the WRONG tree.** Its BASE_DIR
   is `resnet34_model/`, so it reads/writes `resnet34_model/data/*`, NOT the repo-root
   `data/*` that the `experiments/` pipeline uses (run.py resolves `data/processed`
   against REPO_ROOT). Running it would leave repo-root `data/processed` empty and every
   training job would fail. -> need a repo-root splitter.
2. **`experiments/results/` is gitignored** — the seed-replication + confusion analysis
   exist only on the HPC. Retraining overwrites run folders in place, so archive first.

## Decisions (locked with user)
- Archive current results before overwriting.
- Retrain EVERYTHING: 12 ablation + 2 bgrand + 7 seedrep + 7 Plan 2 = 28 runs, 3 runners.
- Preserve read-once: Plan 2 sweep losers are trained but NOT evaluated on the real set
  (the existing Plan 2 tier scripts already do sweep -> select_on_val -> eval winner).

## What already exists and is REUSED unchanged
- `run_droppath_slurm.sh`, `run_res240_slurm.sh`, `run_mixstyle_slurm.sh` — each trains
  its tier with the correct train-only/select/eval barrier (read-once safe).
- `run_bgrand_slurm.sh`, `run_bgrand_real_slurm.sh` — the two seed-42 bgrand cells.
- `run_seedrep_slurm.sh` — the 7-task seedrep array (base x3, bgrand x2, bgrandreal x2).

## What is NEW (additive; no isolation-contract file touched)
1. `experiments/split_dataset.py` — repo-root splitter. Mirrors prepare_dataset's logic
   (70/15/15, `random.seed(42)`, SYMLINKS, cleans ONLY train/val/test so
   `data/real_environment_dataset` is never touched) but with BASE_DIR = repo root, so it
   reads repo-root `data/raw` and writes repo-root `data/processed`. Prints per-class
   counts; refuses if `data/raw` is missing/empty. ASCII output.
2. `experiments/run_ablation_array_slurm.sh` — a 12-task JOB ARRAY (0-11), one config per
   task via `python -m experiments.run --config <cfg>` (parallel, unlike the sequential
   run_all_slurm.sh). Same partition/gres as the seed array.
3. `experiments/retrain_all.sh` — the master launcher, run ON THE LOGIN NODE. It:
   a. runs `split_dataset.py` (synchronous, fast — just symlinks);
   b. archives `experiments/results` -> `experiments/results_archive_<timestamp>/` and
      recreates an empty `experiments/results`;
   c. `sbatch`-submits, so they queue and run concurrently:
      - run_ablation_array_slurm.sh  (12-task array)
      - run_bgrand_slurm.sh, run_bgrand_real_slurm.sh
      - run_seedrep_slurm.sh         (7-task array)
      - run_droppath_slurm.sh, run_res240_slurm.sh, run_mixstyle_slurm.sh
   d. prints the submitted job IDs and the `squeue` command to watch them.

## Notes / caveats baked into retrain_all.sh
- **Mask cache**: `data/mask_cache/` masks are keyed by leaf-image content; the NEW images
  get NEW keys, so bgrand/seedrep re-segment with U^2-Net (slower first epoch) and never
  reuse a stale mask. The old cache is harmless but the script offers an optional
  `--clear-mask-cache` flag to wipe it for a clean start.
- **rembg/u2net + backgrounds**: still required for bgrand/seedrep/combination — the
  launcher checks `data/backgrounds_generic_real` and `_synthetic` are non-empty and that
  rembg+u2net import, and STOPS before submitting if not (fail fast, don't queue 20 jobs
  that will all die).
- **Baseline dependency**: the Plan 2 / bgrand compare steps compare against
  `efficientnetb0_on`. If they run before the ablation finishes, only their final compare
  line is affected (re-runnable), not the training/eval. Acceptable; noted in the script.
- **real_environment_dataset unchanged** — only PlantVillage changed, so the real-world
  test set is the same cleaned 333-image set; new models are simply measured against it.

## Everything downstream must be re-run afterward (NOT in this plan, flagged for later)
compile_results, confusion_matrices --report-only, compare_seeds, domain_gap_samples —
all read the results that are being regenerated. Re-run them once the jobs finish.

## Test
Offline `tmp/test_split_dataset.py`: build a fake repo-root `data/raw` with a few classes
(incl. the spider-mites folder with a space), run the splitter, assert 70/15/15 counts per
class, that symlinks (not copies) were made, that `real_environment_dataset` is untouched,
and that an empty raw dir refuses. Skip nothing — pure stdlib.

## Sequence
1. Build the 3 new files + the test; run the test.
2. Show diffs; commit + push.
3. On HPC: `git pull`, then `bash experiments/retrain_all.sh` from the repo root.
