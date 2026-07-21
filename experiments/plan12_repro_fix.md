# Plan 12 - Single-environment reproducibility fix + baseline documentation

**Trigger:** a reviewer flagged two things before the write-up:
1. `confusion_matrices` refused to write `resnet50_on`'s matrix - its re-inferred
   real-world macro-F1 (0.232755) differs from the published 0.232497 (> 1e-6). Cause:
   the study ran across multiple Colab sessions on different GPUs (early runs on T4,
   final post-processing on A100); the A100's cuDNN picked a different conv algorithm
   (`CUDNN_STATUS_NOT_SUPPORTED` in the log) so the forward pass isn't bit-identical.
   NOT a data problem - `resnet50_off` reproduced exactly on the same 261-image set, and
   the split is deterministic (seed 42).
2. `efficientnetb0_on` (RW F1 0.3457) vs `efficientnetb0_seedrep_base_s42` (0.3342):
   same recipe, ~0.0115 apart. Confirmed by diffing the configs: identical stack/seed/
   budget, but two DIFFERENT code paths (`run.py` vs `run_bgrand.py` with `prob: 0.0`).
   By design - not a bug.

**Goal:** every number in the results tables is produced in ONE environment (the current
A100) so all 12 ablation rows + the story rows + the 7 seedrep rows reproduce by
construction, and document the two-baseline design so an examiner isn't surprised.

## Decision (user, this session)
Scope = **re-score ALL evaluated rows on the A100** (not just `resnet50_on`). Forward
pass only, frozen checkpoints, no retraining, no re-selection.

## Approach

### A. New additive script - `experiments/reeval_single_env.py`
Re-scores each frozen checkpoint on the current device and writes its
`eval_results_real_world.json` AND its confusion counts from the **same** forward pass,
so published-number == matrix by construction.

- **Reuses existing internals only (imports, no edits):** `_load_model, _predict,
  _metrics, _plot_confusion` (common/evaluate), `_real_world_loader, _resolution_of,
  cm_path, OUT_DIR` (confusion_matrices), `_load, RESULTS_DIR, BACKBONE_ORDER,
  STORY_ROWS` (compile_results), and the 7 seedrep run names (compare_seeds).
- **Rows touched:** the 12 ablation rows, the evaluated STORY_ROWS
  (bgrand, bgrand_real, droppath02, res240, mixstyle_l12), and the 7 seedrep rows.
- **Read-once rule preserved:** ONLY rows that already have a published
  `eval_results_real_world.json` are re-scored. `droppath03` / `mixstyle_l123` lost on
  validation and are skipped (no published file -> nothing to re-measure).
- **8-class safe:** scores against `data/real_environment_dataset` as-is (8 classes),
  using evaluate's own by-name label remap. It does NOT use
  `reevaluate_real_world.py`, whose `--check` gate hardcodes 10 classes and would
  reject the 8-class set.
- **Nothing destroyed:** before overwriting a row's `eval_results_real_world.json` +
  `cm_real_world_test.png`, copy them into `experiments/results/<run>/backup_pre_a100_<ts>/`.
  Inference happens before backup/write, so a mid-run failure leaves everything as-is.
- **Provenance:** each rewritten JSON records `scored_on = "a100_single_env"`, the
  device name, `real_world_n_images`, and `superseded_macro_f1` (the pre-A100 value),
  so the change is fully traceable.
- **Controlled (PlantVillage) results untouched;** the gap is recomputed from the
  untouched `eval_results.json`, so only the real-world term moves.
- `--dry-run` prints old->new per row and writes nothing. `--run <name>` does one row.

### B. Rebuild tables + matrices (no further inference)
1. `python -m experiments.compile_results` -> regenerates all_results.md/.txt/.csv.
2. `python -m experiments.confusion_matrices --report-only` -> builds all 12 matrices
   from the saved counts (no GPU, no re-inference), so the guard can't trip.
3. `python -m experiments.plan1_bgrand.compare_seeds` -> refreshes seedrep_summary.json
   from the re-scored seedrep JSONs (the realism contrast now single-environment too).

### C. Documentation (no code) - the two-baseline note for the write-up
One paragraph to add to the report (drafted, not committed):
> `efficientnetb0_on` is the canonical baseline for the backbone-ablation and
> architecture tables. The seed-replication study uses a separate baseline,
> `efficientnetb0_seedrep_base_s42/43/44`: the identical recipe run through the
> background-randomization code path with the effect disabled (`prob 0.0`), so that
> baseline, synthetic and real differ in exactly one variable. The two baselines are
> never compared directly; the realism contrast (real - baseline) is computed entirely
> within the seed-replication matrix.

## Where it runs
The checkpoints live on Colab/Drive, not the local PC (no GPU, no weights here). So:
- **Local (me):** write + syntax/ASCII/dry-run-logic test the new script, commit+push.
- **Colab (you):** `git pull`, run the new script + the 3 rebuild commands as cells,
  paste back the output.

## Files
- **NEW:** `experiments/reeval_single_env.py` (additive).
- **NEW:** `experiments/plan12_repro_fix.md` (this file).
- **No edits** to any protected file: `experiments/common/*`, `run.py`, `compare.py`,
  `confusion_matrices.py`, `compile_results.py`, `reevaluate_real_world.py`, the 12
  ablation configs, the plan1/plan2 sources, `retrain_all.sh`, or any `*_slurm.sh`.
  The script only imports from them.

## After the re-run
The published numbers will shift microscopically (~4th decimal) - same checkpoints, one
device. I'll update memory (`project_seed_replication` v3 -> the final A100-consistent
numbers) once the rebuilt `seedrep_summary.json` is back.

## Test plan (local, before you run anything on Colab)
- `python -m py_compile experiments/reeval_single_env.py` (syntax).
- Non-ASCII scan (C-locale safety).
- `--dry-run --run efficientnetb0_on` logic read-through (can't execute without weights).
- Confirm `confusion_matrices --report-only` reads counts and does NOT re-infer (so it
  cannot trip the tolerance guard).

## Steps after approval
1. Write `experiments/reeval_single_env.py`; syntax + ASCII + dry-run-logic check.
2. Show you the script; commit + push to `main`.
3. Give you the Colab cells: `git pull` -> re-score -> compile_results ->
   confusion_matrices --report-only -> compare_seeds.
4. You run them on the A100, paste output; I verify all 12 rows reproduce and update
   memory + hand you the two-baseline paragraph for the report.
