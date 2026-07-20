# Plan 11 - Run the 8-class study on Google Colab (Kaggle phone-gate blocked)

**Goal:** finish the full 8-class study on Google Colab's free GPU. Colab needs no
phone verification (the Kaggle blocker), and `experiments/run_all_local.sh` already
runs on any single GPU, so **no code changes** - only a new setup guide.
`Tomato___Target_Spot` and `Tomato___Tomato_mosaic_virus` stay excluded from training
and evaluation, same as everywhere else.

## Colab vs Kaggle - what actually changes

| Concern | Kaggle | Colab |
|---|---|---|
| GPU unlock | phone verification (blocked for you) | none - just a Google account |
| Persistence | commit "Save Version" output | **Google Drive** (`/content/drive`) - the runtime's own disk is wiped on disconnect |
| Data delivery | private Kaggle Dataset (read-only mount) | upload zip to Drive, **unzip to local `/content`** for fast training reads |
| Session limits | ~30 h/week, 12 h/session | ~12 h max, **~90 min idle disconnect**, dynamic GPU limits (can be throttled for a day after heavy use) |
| Runner | `run_all_local.sh` (unchanged) | `run_all_local.sh` (unchanged) |

### Two Colab-specific rules that drive the design
1. **Never train off Drive.** Reading tens of thousands of small images over Drive's
   FUSE mount makes each epoch crawl. So: **unzip the data into `/content`** (fast local
   SSD) each session and train from there.
2. **`experiments/results/` must live on Drive.** The runtime disk is wiped on every
   disconnect. If results are symlinked to a Drive folder, every checkpoint + eval JSON
   is written straight to Drive and survives - so reconnecting and re-running
   `run_all_local.sh` **resumes** (it skips finished runs). Checkpoints are written once
   per run (not per step), so Drive write latency is fine for this volume.

This split - **data local, results on Drive** - is the whole trick that makes a
multi-session study survivable on Colab.

## Deliverable (1 new file; no code change)

### `experiments/COLAB_SETUP.md` - copy-paste Colab notebook guide
- **A. One-time:** upload `data_for_kaggle.zip` (the same zip from the HPC) to Google
  Drive, e.g. `MyDrive/tomato_fyp/data.zip`. Make a persistent results folder on Drive,
  e.g. `MyDrive/tomato_fyp/results/`.
- **B. Runtime:** Runtime -> Change runtime type -> **T4 GPU**.
- **C. Cells:**
  1. `nvidia-smi` sanity + `git clone` the repo into `/content/repo`.
  2. `pip install` the pinned requirements (both files) - same pins as Kaggle to dodge
     the numpy<2 / albucore traps.
  3. `from rembg import new_session; new_session('u2net')` (Colab has internet).
  4. **Mount Drive** (`google.colab.drive.mount`).
  5. **Wire data:** unzip `MyDrive/tomato_fyp/data.zip` -> `/content/repo/data`
     (auto-detect the folder holding `raw/` + `real_environment_dataset/`, same helper
     as the Kaggle guide), then build the **8-class real-world view**. Because `/content`
     is writable (unlike Kaggle's read-only input), the two excluded classes are simply
     moved aside into `real_environment_dataset_excluded/` - same effect as
     `retrain_all.sh`, reversible.
  6. **Persist results:** symlink `/content/repo/experiments/results` ->
     `MyDrive/tomato_fyp/results` so everything the runner writes lands on Drive.
  7. `!bash experiments/run_all_local.sh --dry-run` then `!bash experiments/run_all_local.sh`.
  8. Read `experiments/results/all_results.md`.
- **D. Multi-session recipe:** because results are on Drive, resuming is just: reconnect,
  re-run cells 1-6 (clone, install, unzip, re-link Drive results), then run the runner
  again - finished runs are skipped. Keep the tab active / interact periodically so the
  ~90 min idle timer doesn't disconnect you mid-run.
- **E. Honest caveats:**
  - Free Colab GPU is **not guaranteed** and can be throttled for ~24 h after heavy use;
    the full ~24-run study will span several sessions. Colab Pro removes most of this if
    you want to pay.
  - Numbers differ slightly from the L4 HPC (cross-GPU FP) - treat as a fresh
    self-consistent set, same as the Kaggle note.
  - The `pip install` (~torch reinstall) runs every fresh session; ~2-3 min each time.

## What I will NOT do
- No change to `run_all_local.sh` or any training code - Colab reuses it as-is.
- No change to the Kaggle guide (it stays valid if the phone gate ever clears).
- No training on your local Windows machine.

## Test plan
- The runner is already tested (syntax + ASCII + dry-run, commit 82a5942). The only new
  artifact is a markdown guide, so the check is: non-ASCII scan of `COLAB_SETUP.md`, and
  a read-through that the cell sequence matches the runner's assumptions (data at
  `repo/data`, results at `repo/experiments/results`).

## Steps after approval
1. Write `experiments/COLAB_SETUP.md`.
2. Non-ASCII scan; show you the cells.
3. Commit + push to `main`.
4. You: upload the zip to Drive, open a Colab notebook, follow the cells. I debug + help
   interpret results as they land.
