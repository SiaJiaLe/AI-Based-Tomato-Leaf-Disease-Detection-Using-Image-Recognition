# Plan 5 — `--refresh`: re-measure on the cleaned real-world images

## Why
The real-world set `data/real_environment_dataset` has been re-processed (images
cleaned) in place. The published numbers were measured on the earlier, messier
version and need to be re-computed on the cleaned images.

`reevaluate_real_world.py` today only does a **set switch**: it archives each run's
retired-set originals into `archive_<old_set>/` and refuses if that archive already
exists. It exists (`archive_real_environment_test/` holds the true retired
originals), so a blanket re-run aborts on the first row — the guard working as
designed. We must NOT overwrite that archive.

We need a different operation: re-measure on the **same** set that is already
published, overwrite the current result in place, and leave the retired archive
alone.

## What changes
One file only: `experiments/reevaluate_real_world.py` (a script created this
session; not under the isolation contract). No configs, no `common/`, no retired
archive touched.

### New `--refresh` mode
- **Precondition (refuses otherwise):** for each row, the existing
  `eval_results_real_world.json` must already have
  `real_world_set == basename(--dir)`. If a row is still on the retired set,
  `--refresh` refuses and tells you to run the normal (non-refresh) switch first.
  This makes it impossible to use `--refresh` to skip the one-time archiving of
  retired originals.
- **Does NOT touch** `archive_real_environment_test/` at all.
- **Backs up the current (about-to-be-replaced) new-set files** — so the first
  pass is never silently lost — into a single timestamped folder per run:
  `archive_real_environment_dataset_<YYYYmmdd_HHMMSS>/` (one timestamp for the
  whole batch). Backs up `eval_results_real_world.json`, `cm_real_world_test.png`,
  and the confusion-counts JSON.
- **Re-runs inference** on `--dir` (frozen checkpoint, same override mechanism),
  then overwrites in place: `eval_results_real_world.json`,
  `cm_real_world_test.png`, and `results/confusion/<run>_cm_real_world.json`.
- **Provenance is coherent:**
  - `real_world_set`, `real_world_dir`, `real_world_n_images` = the cleaned set.
  - `previous_real_world_set` / `previous_macro_f1` are **carried forward
    unchanged** from the existing file (they still describe the RETIRED comparison,
    so the retired-vs-new story is preserved).
  - new fields `superseded_macro_f1` (the first-pass new-set F1 being replaced) and
    `refreshed_at` (timestamp), so the effect of the image cleaning is recorded.
- **Console line** reports first-pass -> cleaned delta:
  `<run>: macro-F1 0.3130 -> 0.33xx (+0.0xxx) on N images. Previous pass backed up.`
- `--dry-run` works with `--refresh`: reports first-pass -> cleaned, writes nothing.
- `macro_f1_published` in the confusion JSON is set to the new cleaned F1, so
  `confusion_matrices --report-only`'s 1e-6 reproduction guard still passes.

### Untouched
The existing default path (set switch, retired archiving, `--check`, single `--run`)
is unchanged; `--refresh` is purely additive and off by default.

## How you'll run it (on the HPC, after `git pull`)
```bash
python -m experiments.reevaluate_real_world --check          # confirm the cleaned set: 10 classes, counts
python -m experiments.reevaluate_real_world --refresh --dry-run   # preview deltas, writes nothing
sbatch experiments/run_reeval_slurm.sh --refresh             # or add --refresh to the batch script
# then, no GPU:
python -m experiments.confusion_matrices --report-only
```
(The SLURM wrapper will be updated to pass `--refresh` through, or you run the
module directly on a GPU node.)

## Tests (offline, in the scratch dir, before any GPU spend)
Extend the existing fixture tests to cover:
- `--refresh` refuses a row still on the retired set.
- `--refresh` does NOT create/modify `archive_real_environment_test/`.
- the current new-set files are copied into `archive_..._<ts>/` before overwrite.
- carried-forward `previous_*` fields survive; `superseded_macro_f1`/`refreshed_at`
  are written; confusion JSON `macro_f1_published` matches the new F1.
