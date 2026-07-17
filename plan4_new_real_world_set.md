# Plan 4 — Replace the real-world test set

Goal: retire `data/processed/real_environment_test` in favour of
`data/real_environment_dataset`, and re-measure every already-evaluated row against it.

Status: awaiting approval. Folders are created; no config or result has been touched.

---

## 1. What this actually costs

Every published real-world number in the study was measured on
`data/processed/real_environment_test`: the 12-row ablation, Plan 1's synthetic-vs-real
contrast (+0.0365), all three Plan 2 tiers, the combination row. **Changing the set changes
the yardstick**, so:

* until re-measured, the new and old real-world numbers **cannot appear in the same table**;
* the whole results chapter is recomputed — including the +0.0365 headline;
* controlled (PlantVillage) numbers are **unaffected**, so the *gap* changes only because its
  real-world term does.

This is legitimate — a better field set is a better instrument, and the current one gives
real-world accuracy of only ~0.41 with `Target_Spot` near-zero in every row. It is worth being
deliberate about, not incidental.

## 2. The trap that makes the naive approach fail

**Editing `real_world_dir` in the YAML configs does nothing for evaluation.**

`run.py:36-41` loads the config, then calls `evaluate_run(results_dir, device)` — passing only
the directory. `evaluate.py:30-31` then does `cfg = ckpt["config"]`: **evaluation reads
`real_world_dir` out of the checkpoint saved at training time.** The YAML's value is used for
training only.

So repointing the YAML and running `--eval-only` would re-evaluate every model on the OLD
directory and print entirely plausible numbers. A silent no-op producing believable results is
the worst failure mode available here, so evaluation needs an explicit override instead.

The configs are still updated (§4) so that any FUTURE training run bakes in the new path — but
that is for provenance, not for this re-evaluation.

## 3. Three hazards this plan closes

1. **Overwriting.** `evaluate_run` writes `eval_results_real_world.json` in place. The old
   numbers would be destroyed. They are archived first.
2. **No provenance.** No result file currently records WHICH real-world set produced it. With
   one set that is merely untidy; with two it is unrecoverable ambiguity. Every file this plan
   writes records its source directory and image count.
3. **Partial downloads.** Fully empty folders make `ImageFolder` raise — fine. But 3 of 10
   classes populated evaluates happily, with seven classes at zero support, producing a
   disastrous macro-F1 caused purely by missing files. The script counts images per class and
   REFUSES to evaluate an incomplete set.

## 4. Files

| File | Change | Why |
|---|---|---|
| `data/real_environment_dataset/<10 classes>/.gitkeep` | NEW (done) | Folder structure reaches the HPC via `git pull`. Names match the training classes exactly — `evaluate.py:117` remaps by NAME, so a mismatch is a `KeyError`. |
| `.gitignore` | MODIFIED | Ignore the images, keep the folders tracked. Without this, a stray `git add -A` on the HPC commits the whole dataset. |
| `experiments/reevaluate_real_world.py` | NEW | The override: score existing frozen checkpoints against a named real-world dir. Archives old results, records provenance, refuses incomplete sets. |
| `experiments/run_reeval_slurm.sh` | NEW | Short GPU job (30 min). No training. |
| `experiments/configs/*.yaml` (12) | MODIFIED | `real_world_dir` -> the new path, so future training records it. **Breaks the isolation contract on the ablation configs — needs explicit approval.** Does not affect existing checkpoints. |
| `experiments/plan2_arch/configs/*.yaml`, `plan1_bgrand/configs/*.yaml` | MODIFIED | Same reason. |

## 5. `reevaluate_real_world.py` behaviour

    python -m experiments.reevaluate_real_world --check    # counts only, no model, no GPU
    python -m experiments.reevaluate_real_world            # archive + re-evaluate every row
    python -m experiments.reevaluate_real_world --run X    # one row

* **`--check` first, always.** Prints per-class image counts for the new set and exits
  non-zero if any of the 10 classes is empty or the set is tiny. Run it after downloading.
* **Only re-evaluates rows that ALREADY have a published
  `eval_results_real_world.json`** — the same structural rule as `confusion_matrices.py`.
  `droppath03` and `mixstyle_l12` lost on validation; their real-world set was never read, and
  it will not be read now on the new set either. The read-once discipline carries over intact.
* **Archives before writing:** `eval_results_real_world.json` ->
  `archive_real_environment_test/eval_results_real_world.json` inside the same run dir, with
  `cm_real_world_test.png`. Refuses to start if an archive already exists (so a second run
  cannot destroy the original by overwriting the archive).
* **Resolution-aware**: dispatches `res240` to the 240px eval path, as `run_arch.py:121-125`
  does. Scoring it at 224 is the silent preprocessing mismatch CLAUDE.md warns about.
* **Writes**: new `eval_results_real_world.json` (same schema, so `compile_results.py` needs no
  change) + `real_world_dir`, `real_world_n_images`, `real_world_set` provenance fields; a new
  `cm_real_world_test.png`; and the confusion counts, so
  `confusion_matrices.py --report-only` works on the new set with no extra inference.
* **Gaps recomputed** from the untouched controlled `eval_results.json`.

## 6. Interaction with Plan 3 (confusion matrices)

`confusion_matrices.py` asserts each row reproduces its published macro-F1. After the switch,
the published file is the NEW set while the checkpoint's `cfg["real_world_dir"]` still names
the OLD one — so it would read the old set, mismatch, and abort. **That is the guard working
correctly**, but it must be taught about the override: it takes the same `--dir` and, once
`reevaluate_real_world` has written the counts, `--report-only` rebuilds everything for free.

**Recommendation: do not run the Plan 3 SLURM job on the old set first.** Its whole output
would be superseded within days. Re-evaluate on the new set, then generate the confusion
matrices from that — one pass, current numbers.

## 7. Order of operations

1. Approve. I write the script + SLURM job, update the configs, commit, push.
2. You: `git pull` on the HPC — the 10 empty folders appear.
3. You: download the images into them.
4. You: `python -m experiments.reevaluate_real_world --check` (login node, no GPU) and paste
   me the counts. **Do not skip this.**
5. You: `sbatch experiments/run_reeval_slurm.sh` — archives, re-evaluates all 18 rows.
6. `compile_results` + `confusion_matrices --report-only` -> the new results chapter.

## 8. What I would ask you to consider

The old set is not worthless: `bgrand_real` beat `bgrand` by +0.0365 **on it**. If that
contrast survives on a new, independent field set, it stops being a single-seed curiosity and
becomes a replicated finding — the strongest claim available. So archive the old numbers
properly (§3.1) and keep the old directory on disk. Two sets agreeing is a result; two sets
disagreeing is also a result. Deleting one is neither.
