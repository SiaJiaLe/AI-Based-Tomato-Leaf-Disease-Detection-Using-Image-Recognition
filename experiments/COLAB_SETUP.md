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

### Cell 1 - GPU check + get the code
```python
!nvidia-smi -L
!git clone https://github.com/SiaJiaLe/AI-Based-Tomato-Leaf-Disease-Detection-Using-Image-Recognition.git /content/repo
%cd /content/repo
!git log --oneline -1
```
`nvidia-smi -L` must list a GPU (e.g. `Tesla T4`). If it errors, the runtime type isn't
set to GPU (Step B).

### Cell 2 - install the pinned dependencies
```python
!pip install -q torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
!pip install -q -r experiments/requirements.txt
!pip install -q -r experiments/plan1_bgrand/requirements.txt
import torch; print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available())
```
`CUDA: True` must print.

> **If a later cell throws a numpy/torch import error:** Colab preinstalls a newer numpy,
> and downgrading it to 1.26.4 sometimes needs a kernel restart to take effect. Do
> **Runtime -> Restart session**, then re-run from **Cell 3** (the pip installs persist
> across a restart, so you can skip Cell 2).

### Cell 3 - download the segmentation model once
```python
from rembg import new_session
new_session("u2net")
print("u2net ready")
```

### Cell 4 - mount Google Drive
```python
from google.colab import drive
drive.mount("/content/drive")
```
Follow the popup to authorize. Drive then appears at `/content/drive/MyDrive`.

### Cell 5 - unzip data to local disk + build the 8-class eval view
```python
import os, glob, zipfile, shutil

REPO   = "/content/repo"
DRIVE  = "/content/drive/MyDrive/tomato_fyp"      # <-- your Drive folder
ZIP    = os.path.join(DRIVE, "data.zip")          # <-- your uploaded data zip
EXCLUDE = {"Tomato___Target_Spot", "Tomato___Tomato_mosaic_virus"}

data = os.path.join(REPO, "data")

# Unzip from Drive -> local /content (fast training reads). Skips if already done.
need = not (os.path.isdir(os.path.join(data, "raw")) and
            os.path.isdir(os.path.join(data, "real_environment_dataset")))
if need:
    assert os.path.isfile(ZIP), f"Zip not found on Drive: {ZIP}"
    print("Unzipping data to local disk (one-time per session, a few minutes)...")
    shutil.rmtree("/content/_data_tmp", ignore_errors=True)
    with zipfile.ZipFile(ZIP) as z:
        z.extractall("/content/_data_tmp")
    # find the folder holding raw/ + real_environment_dataset/ (zip may nest under data/)
    def find_root(base):
        for b in [base] + sorted(glob.glob(os.path.join(base, "*"))):
            if (os.path.isdir(os.path.join(b, "raw")) and
                    os.path.isdir(os.path.join(b, "real_environment_dataset"))):
                return b
        raise RuntimeError("raw/ + real_environment_dataset/ not found inside the zip")
    src = find_root("/content/_data_tmp")
    os.makedirs(data, exist_ok=True)
    for name in os.listdir(src):
        shutil.move(os.path.join(src, name), os.path.join(data, name))
    shutil.rmtree("/content/_data_tmp", ignore_errors=True)
print("data/ contents:", sorted(os.listdir(data)))

# 8-class real-world view: move the 2 excluded classes aside (local disk is writable,
# so unlike Kaggle we can just move them - reversible; the Drive zip is untouched).
real = os.path.join(data, "real_environment_dataset")
excl = os.path.join(data, "real_environment_dataset_excluded"); os.makedirs(excl, exist_ok=True)
for c in EXCLUDE:
    p = os.path.join(real, c)
    if os.path.isdir(p):
        shutil.move(p, os.path.join(excl, c)); print("Moved out of eval set:", c)
kept = [c for c in sorted(os.listdir(real)) if os.path.isdir(os.path.join(real, c))]
print(f"Real-world eval classes (should be 8): {len(kept)}")

os.makedirs(os.path.join(data, "mask_cache"), exist_ok=True)
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
