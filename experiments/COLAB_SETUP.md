# Running the 8-class study on Google Colab

Use this if Kaggle's phone verification is blocking you. Colab needs no phone
verification - just a Google account. The runner (`experiments/run_all_local.sh`) is
the same one Kaggle uses; nothing about training changes. `Tomato___Target_Spot` and
`Tomato___Tomato_mosaic_virus` stay **excluded from training and evaluation**.

## The one thing to understand first

Colab **erases the runtime's disk every time it disconnects** (and it disconnects after
~90 min idle, or ~12 h max). So the whole setup is built around Google Drive:

- **Data lives on Drive, but is UNZIPPED to local `/content` before training.** Training
  reads tens of thousands of small images every epoch - reading those over Drive would
  be painfully slow, so the images must sit on the fast local disk.
- **Results are written straight to Drive** (via a symlink). Every checkpoint and metric
  file survives a disconnect, so when you reconnect and re-run, the runner **skips the
  runs that already finished** and continues. That resume ability is what lets ~24 runs
  span several short Colab sessions.

> Numbers will differ slightly from the L4 HPC run (different GPU, floating-point isn't
> bit-identical). Treat this as one fresh, self-consistent set of results.

---

## A. One-time setup on Google Drive

1. In Google Drive, make a folder: **`MyDrive/tomato_fyp/`**.
2. Upload your data zip into it as **`MyDrive/tomato_fyp/data.zip`** (the same
   `data_for_kaggle.zip` you made on the HPC - just rename it `data.zip`, or change the
   `ZIP` path in Cell 5).
3. That's it - the results folder is created automatically.

---

## B. New Colab notebook

1. Go to **https://colab.research.google.com** -> **New notebook**.
2. **Runtime -> Change runtime type -> Hardware accelerator = T4 GPU -> Save.**

---

## C. The cells

Paste each as its own cell and run in order.

### Cell 1 - GPU check + get the code (clean checkout)
```python
!nvidia-smi -L
# Fresh checkout every session. Safe to re-run: your data comes from Cell 5 and your
# results live on Drive (Cell 6), so nothing important lives inside /content/repo.
# `%cd /content` first so we never delete the folder we're standing in.
%cd /content
!rm -rf /content/repo
!git clone https://github.com/SiaJiaLe/AI-Based-Tomato-Leaf-Disease-Detection-Using-Image-Recognition.git /content/repo
%cd /content/repo
!git log --oneline -1
```
`nvidia-smi -L` must list a GPU (e.g. `Tesla T4` or `A100`). If it errors, the runtime
type isn't set to GPU (Step B). The last line must print a commit (e.g.
`92c1170 COLAB_SETUP: ...`); if it says "not a git repository", the clone didn't finish -
just run the cell again.

### Cell 2 - install the pinned dependencies
```python
!pip install -q torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
!pip install -q -r experiments/requirements.txt
!pip install -q -r experiments/plan1_bgrand/requirements.txt
import torch; print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available())
```
`CUDA: True` must print.

> **IMPORTANT - after Cell 2, restart the session once.** Cell 2 downgrades numpy to
> 1.26.4, but the kernel still has the old numpy binary loaded, so the next import often
> fails with `numpy.dtype size changed ... Expected 96, got 88` (or a similar
> binary-incompatibility / numpy error). This is expected. Do **Runtime -> Restart
> session**, then continue from **Cell 3** - the pip installs persist across a restart,
> so you do NOT re-run Cell 1 or Cell 2. (If you prefer, just restart proactively right
> after Cell 2 finishes and skip the error entirely.)

### Cell 3 - download the segmentation model once
```python
# Colab preinstalls cupy built for numpy 2; it breaks rembg's import under our numpy
# 1.26.4 pin. We never use cupy, so remove it first (harmless if it isn't installed).
!pip uninstall -y cupy-cuda12x cupy
from rembg import new_session
new_session("u2net")
print("u2net ready")
```
> If this cell still errors with a numpy binary-incompatibility message, you skipped the
> restart above - do **Runtime -> Restart session** and run Cell 3 again.

### Cell 4 - mount Google Drive
```python
from google.colab import drive
drive.mount("/content/drive")
```
Follow the popup to authorize. Drive then appears at `/content/drive/MyDrive`.

### Cell 5 - unzip data to local disk + build the 8-class eval view
```python
import os, glob, zipfile, shutil

REPO    = "/content/repo"
DRIVE   = "/content/drive/MyDrive/tomato_fyp"     # <-- your Drive folder
ZIP     = os.path.join(DRIVE, "data.zip")         # <-- your uploaded data zip
EXCLUDE = {"Tomato___Target_Spot", "Tomato___Tomato_mosaic_virus"}
IMG = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

data = os.path.join(REPO, "data")
real = os.path.join(data, "real_environment_dataset")
raw  = os.path.join(data, "raw")
bgr  = os.path.join(data, "backgrounds_generic_real")

def class_images(d):
    """Total image files across the immediate <class>/ subfolders of d."""
    if not os.path.isdir(d):
        return 0
    return sum(sum(1 for f in os.listdir(os.path.join(d, c)) if f.lower().endswith(IMG))
               for c in os.listdir(d) if os.path.isdir(os.path.join(d, c)))

def flat_images(d):
    """Image files sitting directly inside d (backgrounds are a flat folder, not per-class)."""
    return len([f for f in os.listdir(d) if f.lower().endswith(IMG)]) if os.path.isdir(d) else 0

def resolve_root(base):
    """Return the dir whose <class>/ subfolders actually hold images. Handles a
    same-name nested folder (real_environment_dataset/real_environment_dataset) and
    .gitkeep-only placeholder shells - so re-unzipping on a resumed session self-heals."""
    if class_images(base) > 0:
        return base
    for sub in (sorted(os.listdir(base)) if os.path.isdir(base) else []):
        p = os.path.join(base, sub)
        if os.path.isdir(p) and class_images(p) > 0:
            return p
    return base

# Unzip from Drive -> local /content if ANY of the three sources is empty. Note that
# backgrounds_generic_real AND real_environment_dataset ship in the repo as EMPTY
# placeholder folders (README + .gitignore only), so the git clone pre-creates them -
# we must MERGE the zip's images into them, not skip because the folder already exists.
need = (class_images(raw) == 0 or class_images(resolve_root(real)) == 0 or flat_images(bgr) == 0)
if need:
    assert os.path.isfile(ZIP), f"Zip not found on Drive: {ZIP}"
    print("Unzipping data from Drive to local disk (one-time per session, a few minutes)...")
    shutil.rmtree("/content/_data_tmp", ignore_errors=True)
    with zipfile.ZipFile(ZIP) as z:
        z.extractall("/content/_data_tmp")
    def find_data_root(b0):
        for b in [b0] + sorted(glob.glob(os.path.join(b0, "*"))):
            if (os.path.isdir(os.path.join(b, "raw")) and
                    os.path.isdir(os.path.join(b, "real_environment_dataset"))):
                return b
        raise RuntimeError("raw/ + real_environment_dataset/ not found inside the zip")
    src = find_data_root("/content/_data_tmp")
    os.makedirs(data, exist_ok=True)
    # RECURSIVELY merge each top-level item into data/. If a destination is missing we
    # move it wholesale (fast); if it already exists we descend and move in only the files
    # it's missing. The recursion matters: the git clone pre-creates EMPTY per-class
    # folders under real_environment_dataset (and README/.gitignore under
    # backgrounds_generic_real), so a shallow one-level merge would see those folders
    # exist and skip the images inside them. Recursing copies the images into them.
    def merge_into(s, d):
        if not os.path.exists(d):
            shutil.move(s, d); return
        if os.path.isdir(s) and os.path.isdir(d):
            for item in os.listdir(s):
                merge_into(os.path.join(s, item), os.path.join(d, item))
    for name in os.listdir(src):
        merge_into(os.path.join(src, name), os.path.join(data, name))
    shutil.rmtree("/content/_data_tmp", ignore_errors=True)
print("data/ contents:", sorted(os.listdir(data)))

# Flatten a nested / placeholder-only real_environment_dataset so it holds the images.
resolved = resolve_root(real)
if resolved != real:
    print(f"Real images are nested at {resolved} - flattening into {real}")
    tmp = real + "__flatten_tmp"
    shutil.move(resolved, tmp)
    shutil.rmtree(real, ignore_errors=True)     # drop .gitkeep-only shells / empty nesting
    shutil.move(tmp, real)

# 8-class real-world view: move the 2 excluded classes aside (writable local disk, so
# unlike Kaggle we can just move them - reversible; the Drive zip is untouched).
excl = os.path.join(data, "real_environment_dataset_excluded"); os.makedirs(excl, exist_ok=True)
for c in EXCLUDE:
    p = os.path.join(real, c)
    if os.path.isdir(p):
        shutil.move(p, os.path.join(excl, c)); print("Moved out of eval set:", c)

# Verify ALL THREE sources before training - fail loudly here, not hours into a run.
def report(root, label, per_class=True):
    if per_class:
        cls = sorted(c for c in os.listdir(root) if os.path.isdir(os.path.join(root, c)))
        print(f"\n{label}: {len(cls)} classes")
        for c in cls:
            n = sum(1 for f in os.listdir(os.path.join(root, c)) if f.lower().endswith(IMG))
            print(f"  {n:5d}  {c}")
    else:
        print(f"\n{label}: {flat_images(root)} images")
report(real, "REAL eval (want 8, all nonzero)")
report(raw,  "RAW training (want 10, all nonzero)")
report(bgr,  "REAL backgrounds (want ~30-100)", per_class=False)
assert class_images(real) > 0, "REAL eval set has no images - check the zip layout!"
assert class_images(raw)  > 0, "RAW training set has no images - check the zip layout!"
assert flat_images(bgr)   > 0, "backgrounds_generic_real is EMPTY - bgrand_real / seedrep will fail! Check the zip has these photos."
os.makedirs(os.path.join(data, "mask_cache"), exist_ok=True)
print("\nData ready.")
```

### Cell 6 - point results at Drive so they survive disconnects
```python
import os
DRIVE_RESULTS = "/content/drive/MyDrive/tomato_fyp/results"
os.makedirs(DRIVE_RESULTS, exist_ok=True)
link = "/content/repo/experiments/results"
if os.path.islink(link):
    os.remove(link)
elif os.path.isdir(link) and not os.listdir(link):
    os.rmdir(link)
os.symlink(DRIVE_RESULTS, link)
print("results ->", os.path.realpath(link))
```

### Cell 7 - run the study
```python
# Preview the plan + what would be skipped, without training:
!cd /content/repo && bash experiments/run_all_local.sh --dry-run
```
```python
# The real run (long - trains, evaluates, post-processes; writes to Drive):
!cd /content/repo && bash experiments/run_all_local.sh
```

### Cell 8 - read the results
```python
!cat /content/repo/experiments/results/all_results.md
!ls /content/repo/experiments/results/confusion/
```

---

## D. Multi-session recipe (how ~24 runs fit into short sessions)

Because results live on Drive, resuming is simple:

1. If Colab disconnects (or you close it), just **reconnect** / open the notebook again.
2. Re-run **Cells 1-6** (clone, install, mount Drive, unzip data, re-link results to
   Drive). Cell 5 skips the unzip if data is already there; Cell 6 re-attaches the same
   Drive results folder.
3. Run **Cell 7** again. The runner sees the finished runs on Drive and **skips** them,
   continuing with what's left.
4. Repeat until Cell 8 shows every row and `results/seedrep_summary.json` exists.

**Keep the tab active.** Colab disconnects after ~90 min of no interaction. Leave the
browser tab open and click into it occasionally, or the session drops mid-run (the
runner will still resume next time - you just lose the in-progress run).

---

## E. Caveats (so nothing surprises you)

- **Free Colab GPU isn't guaranteed** and can be throttled for ~24 h after heavy use.
  The full study spans several sessions; if you get GPU-limited, wait and resume later.
  Colab Pro removes most of this if you choose to pay.
- **Drive space:** the checkpoints (`best_model.pth`, VGG16 is biggest) total a few GB
  on Drive - make sure your Drive has room.
- **The pip install (~torch reinstall) runs every fresh session** (~2-3 min). That's
  normal - Colab's disk is fresh each time.
- If `rembg` ever fails to import, the runner **skips only** the background blocks
  (bgrand / bgrand_real / seedrep / mixstyle combo) and still finishes the ablation +
  Plan 2 Tiers 1-2. Fix rembg and re-run to fill the rest in.

## What "excluded from training" means here (verification)
- **Split:** the runner calls `split_dataset --exclude Tomato___Target_Spot
  Tomato___Tomato_mosaic_virus`, so neither enters train/val/test.
- **Evaluation:** Cell 5 moves those two out of `real_environment_dataset`, so real-world
  metrics are over 8 classes and never KeyError.
- Both are reversible - their images stay on disk (raw/ and the `_excluded/` folder).

---

## F. Retrain everything from scratch (keeps the old results)

Use this when you want a **fresh full run of all 28 models** but must not lose the
completed study. Because the runner in Cell 7 **skips** any run that already has results,
a fresh retrain needs an **empty** `results/`. The trick: rename the current Drive
`results` folder to `results_archive_<timestamp>` - nothing is deleted, every old
checkpoint/metric/confusion file is preserved - then start with a fresh empty `results/`.

> Archive the results you actually want to keep FIRST. If you re-scored on the A100
> (`reeval_single_env.py`), that final single-environment set is what gets archived now -
> archive it before retraining or it is overwritten.

### Archive cell - run ONCE, in place of Cell 6
```python
# === RETRAIN FROM SCRATCH: archive the current results, then start fresh ===
# Run this ONCE, only when you deliberately want to retrain every model. It RENAMES the
# current Drive results folder to results_archive_<timestamp> (nothing is deleted - every
# old checkpoint, metric and confusion file is preserved, exactly like your existing
# results_archive_* folders), then creates a fresh empty results/ so Cell 7 retrains all
# runs instead of skipping them.
#
# DO NOT run this on a normal resume session - it would archive your half-finished fresh
# run and start over. Run it once, then only Cells 1-5 + 7 on later sessions.
import os, shutil, time

DRIVE   = "/content/drive/MyDrive/tomato_fyp"
results = os.path.join(DRIVE, "results")
assert os.path.isdir("/content/drive/MyDrive"), "Drive not mounted - run Cell 4 first."

if os.path.isdir(results) and os.listdir(results):
    stamp   = time.strftime("%Y%m%d_%H%M%S")
    archive = os.path.join(DRIVE, f"results_archive_{stamp}")
    shutil.move(results, archive)
    print("Archived old results ->", archive, "|", len(os.listdir(archive)), "items preserved")
else:
    print("No existing results to archive (missing or already empty).")

# Fresh empty results/ + re-point the repo symlink at it.
os.makedirs(results, exist_ok=True)
link = "/content/repo/experiments/results"
if os.path.islink(link):
    os.remove(link)
elif os.path.isdir(link) and not os.listdir(link):
    os.rmdir(link)
os.symlink(results, link)
print("results ->", os.path.realpath(link), "(fresh, empty)")
print("Now run Cell 7 to retrain everything from scratch.")
```

**Fresh-retrain flow:** Cells 1-5 (setup + data) -> **this archive cell (once, instead of
Cell 6)** -> Cell 7 (train). On later resume sessions run **Cells 1-5 + 7 only** - do NOT
re-run the archive cell, or it will archive the half-finished fresh run and start over.

**To read an archived study later:** point the symlink back at it for one session -
`os.remove("/content/repo/experiments/results")` then
`os.symlink("/content/drive/MyDrive/tomato_fyp/results_archive_<stamp>", "/content/repo/experiments/results")`
- then Cell 8 reads that archived set. Re-run Cell 6 to return to the live results.
