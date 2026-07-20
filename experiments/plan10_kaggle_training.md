# Plan 10 - Run the 8-class study on Kaggle Notebooks (HPC credits exhausted)

**Goal:** finish the full 8-class study (all ~24 runs) on Kaggle's free GPU instead
of the Sunway HPC, WITHOUT changing any training logic, any result, or any file in
the no-touch set (`experiments/common/*`, `run.py`, `compare.py`, the 12 ablation
configs, `plan1_bgrand`/`plan2_arch` source that already ran).

The training commands are already portable - `python -m experiments.run --config ...`
resolves every path against the repo root, so it runs unchanged on any single GPU.
Only the SLURM layer (`sbatch`, job arrays, `afterok`) doesn't carry over. This plan
replaces that layer with a plain, resume-safe sequential runner, plus a Kaggle setup
guide. **Nothing in this plan trains locally on your Windows machine; it is all run
on Kaggle.**

---

## Kaggle constraints that shape the design

| Constraint | Consequence for the design |
|---|---|
| Free GPU ~30 hrs/week, **12 hr max per session** | ~24 runs won't finish in one sitting -> the runner must **resume** and skip finished runs. |
| `/kaggle/working` is wiped when a session ends unless you **Save Version (commit)** | Multi-session plan built around commit-runs + carrying the previous output forward. |
| `/kaggle/input` (your uploaded data) is **read-only** | `retrain_all.sh` *moves* the 2 excluded classes out of the real-world set - can't `mv` read-only data. Fix below. |
| No `conda`, no `tomato-ml` env | The runner calls `python -m ...` directly (no `conda activate`); deps installed via `pip` in a notebook cell. |
| No SLURM | Sequential runner replaces the 7 `*_slurm.sh` + `retrain_all.sh` orchestration. |
| Internet needs phone verification | Required for `git clone`, `pip install`, and the one-time `u2net` model download. |

### The read-only-input trap and its fix (important)
An 8-class model evaluated on a 10-class real-world folder **KeyErrors** on the 2
excluded classes (`common/evaluate.py` remaps real-world folders into the training
label space *by name*). On HPC, `retrain_all.sh` solves this by *moving* those 2
folders out of `data/real_environment_dataset`. On Kaggle that folder lives in
read-only `/kaggle/input`, and I also **cannot** edit the 12 ablation configs that
hardcode `real_world_dir: data/real_environment_dataset`.

**Fix (no moves, no config edits):** the setup notebook builds
`data/real_environment_dataset/` in the writable workspace as a directory of
**per-class symlinks pointing at only the 8 kept classes** in the read-only input.
The hardcoded path then already resolves to an 8-class set. `data/raw` can be a
whole-folder symlink (all 10 classes present; `split_dataset --exclude` drops the 2
at split time, exactly as on HPC).

---

## Deliverables (2 new files, both additive - no existing file modified)

### 1. `experiments/run_all_local.sh` - SLURM-free, resume-safe runner
The single command the Kaggle notebook calls. It is `retrain_all.sh` with the
cluster-only parts removed and resume added:

- **No `conda activate`** (Kaggle base env).
- **No `sbatch` / no `afterok`** - runs every block **sequentially** in one process.
- **No archive-and-wipe of `experiments/results/`.** The opposite of `retrain_all.sh`:
  on Kaggle we must *keep* prior results so a resumed session skips finished runs.
- **No `mv` of the read-only real-world set** (the notebook already built the 8-class
  view). It still passes `--exclude ... --skip-if-exists` to the splitter.
- **Resume guard:** a helper `done_marker=<results>/<run>/eval_results_real_world.json`
  is the "this run finished" signal (it is the last file `evaluate_run` writes). Any
  run whose marker exists is **skipped**; so re-running the script after a 12 hr cutoff
  continues where it stopped. Plan 2 sweep members (`--train-only`, no eval) are guarded
  on `best_model.pth` instead.
- Runs the **exact command sequences** already in the SLURM scripts, block by block:
  1. **Split** (once): `python -m experiments.split_dataset --exclude Tomato___Target_Spot Tomato___Tomato_mosaic_virus --skip-if-exists`
  2. **Ablation (12):** `python -m experiments.run --config experiments/configs/<name>.yaml` for each of the 12 (guarded).
  3. **bgrand (1):** `run_bgrand --config .../efficientnetb0_on_bgrand.yaml` then `compare_bgrand` (mask sanity grids kept).
  4. **bgrand_real (1):** `run_bgrand --config .../efficientnetb0_on_bgrand_real.yaml` then `compare_bgrand --bgrand efficientnetb0_on_bgrand_real`.
  5. **seedrep (7):** `run_bgrand --config .../seedrep/<name>.yaml` for the 7 configs (guarded).
  6. **Plan 2 Tier 1 (droppath):** train `droppath02` + `droppath03` `--train-only` (guarded on `best_model.pth`) -> `select_on_val` -> `run_arch --eval-only` on the val winner -> `compare_arch`. **Read-once barrier preserved verbatim.**
  7. **Plan 2 Tier 2 (res240):** `run_arch --config .../efficientnetb0_on_res240.yaml` -> `compare_arch`.
  8. **Plan 2 Tier 3 (mixstyle):** train `l12` + `l123` `--train-only` -> `select_on_val` -> eval winner -> `compare_arch` -> combination row `<winner>_bgrand` train+eval -> `compare_arch`. **The exact 2x2-factorial / read-once logic from `run_mixstyle_slurm.sh`, unchanged.**
  9. **Post-process:** `compile_results`, `confusion_matrices`, `plan1_bgrand.compare_seeds`, each fault-tolerant (one failing step never blocks the others), same as `run_postprocess_slurm.sh`.
- **Fault-tolerant:** like `run.py --all`, a failing run is logged and the batch keeps
  going, so one bad run doesn't waste the rest of a 12 hr session.
- **ASCII-only** output (compute-node C-locale lesson: em-dash in a `print`/file-write
  crashes; the runner uses `-`).
- `rembg`/`u2net` and background-folder preflight checks kept (bgrand/seedrep need them),
  but phrased as **fail-fast with the pip line**, since on Kaggle you *do* install in a
  notebook cell first (unlike HPC, where installing mid-job corrupted the shared env).

### 2. `experiments/KAGGLE_SETUP.md` - copy-paste notebook guide
Written for someone new to Kaggle. Contents:
- **A. One-time:** create Kaggle account, verify phone (unlocks GPU + Internet), the
  exact **folder layout** to zip and upload as a *private Kaggle Dataset*:
  ```
  raw/<10 class folders>/*.jpg
  real_environment_dataset/<10 class folders>/*.jpg
  backgrounds_generic_real/*.jpg
  backgrounds_generic_synthetic/*.jpg      (optional - can be generated)
  ```
- **B. Notebook settings:** Accelerator = GPU (T4/P100), Internet = ON, Persistence.
- **C. The notebook cells** (each explained):
  1. `git clone` the repo into `/kaggle/working` (public) **or** add the repo as a
     dataset if it's private - both documented.
  2. `pip install -r experiments/requirements.txt -r experiments/plan1_bgrand/requirements.txt`
     (the pinned versions - reproduces the env and dodges the numpy<2 / albucore traps).
  3. Pre-download the segmentation model once: `from rembg import new_session; new_session('u2net')`.
  4. **Wire the data** (a short Python cell): symlink `data/raw`, `data/backgrounds_*`
     from the input mount, and build the **8-class** `data/real_environment_dataset`
     view (per-class symlinks, skipping the 2 excluded classes). Set `DATA_ROOT` to the
     input path once.
  5. **(resume)** copy the previous session's `experiments/results/` back into place -
     from the notebook's own prior version output (added as an input) - so the runner
     skips finished runs.
  6. `bash experiments/run_all_local.sh`.
  7. Read results (`experiments/results/all_results.md`, `confusion/`, per-run PNGs).
- **D. The multi-session recipe** (the part that makes ~24 runs fit in 12 hr slices):
  - Use **Save Version -> Save & Run All (Commit)** so the notebook runs headless up to
    12 hr and its `/kaggle/working` is saved as the version's **Output**.
  - Next session: **Add Input -> your previous version's output**, so Cell 5 restores
    results and the runner resumes. Repeat until `all_results.md` lists every row.
  - Note on output size: checkpoints (`best_model.pth`) dominate - VGG16 is the biggest;
    total across ~24 runs is a few GB, well within Kaggle limits.

---

## What I will NOT do
- Not modify any file in the no-touch set (`common/*`, `run.py`, `compare.py`, the 12
  ablation configs) or any `plan1_bgrand`/`plan2_arch` source that already ran.
- Not change any training hyperparameter, seed, split ratio, or the exclusion list -
  the study is identical to the HPC one; only the machine changes.
- Not edit `retrain_all.sh` or the `*_slurm.sh` scripts (HPC still works if credits
  return). `run_all_local.sh` is a *sibling*, not a replacement.
- Not run training on your local Windows machine.

## Caveats to state up front
- **Numbers will differ slightly** from any HPC run: Kaggle's T4/P100 is a different GPU
  from the L4, and cross-GPU floating-point is not bit-identical. The pinned library
  versions keep it as close as possible, but treat these as a fresh, self-consistent set
  (same lesson as [[project_real_world_eval]]: everything is re-read from the new run).
- **Total GPU time** for ~24 runs (VGG16/ResNet50 are the heavy ones) will very likely
  exceed one 12 hr session and may span more than one weekly 30 hr quota. The resume
  design is what makes that survivable.

## Test plan (offline, before you touch Kaggle)
- `bash -n experiments/run_all_local.sh` (syntax) + a dry-run mode that prints the plan
  and the skip/keep decision for each run against a fake `experiments/results/` tree,
  so we verify the resume logic and the exact command list without a GPU.
- Grep the new files for non-ASCII (the em-dash crash guard).

## Steps after you approve
1. Write `experiments/run_all_local.sh` + `experiments/KAGGLE_SETUP.md`.
2. Run the offline dry-run/syntax test; show you the output.
3. Commit + push to `main` (per your workflow).
4. You: upload the data to Kaggle, then follow `KAGGLE_SETUP.md`. I help debug as issues
   come up, and interpret results when they land.
