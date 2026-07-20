# Running the 8-class study on Kaggle Notebooks

Your Sunway HPC credits ran out mid-training. This guide finishes the **full
8-class study** (all ~24 runs) on Kaggle's **free** GPU instead. Nothing in your
training changes - only the machine. `Tomato___Target_Spot` and
`Tomato___Tomato_mosaic_virus` stay **excluded from training and evaluation**.

The one script that does the work is `experiments/run_all_local.sh` (the SLURM-free,
**resume-safe** version of `retrain_all.sh`). Kaggle stops every session at 12 hours;
the runner **skips whatever already finished**, so you just re-run it across a few
sessions until the whole table is filled in.

---

## What you need to know about Kaggle (read once)

| Thing | Value |
|---|---|
| Free GPU | Tesla T4 or P100, **~30 hours/week**, **12 hours max per session** |
| GPU + Internet | Require a **phone-verified** account (Settings -> Phone Verification) |
| `/kaggle/working` | Your writable workspace. **Wiped when a session ends** unless you *Save Version (commit)* |
| `/kaggle/input` | Your uploaded data - **read-only** |

Because ~24 runs (VGG16 and ResNet50 are the slow ones) will **not** finish in one
12-hour session, the plan is: run, let Kaggle save the output, feed that output back
into the next session, run again. The runner picks up where it left off each time.

> **Expect the numbers to differ slightly from any earlier HPC run.** Kaggle's T4/P100
> is a different GPU from the L4, and floating-point isn't bit-identical across GPUs.
> The pinned library versions keep it as close as possible. Treat this as one fresh,
> self-consistent set of results (re-read everything from this run).

---

## A. One-time: upload your data as a private Kaggle Dataset

1. On your PC, gather your `data/` into this exact layout and zip it:
   ```
   raw/                              <- 10 class folders (the splitter drops the 2 excluded ones)
     Tomato___Bacterial_spot/*.jpg
     Tomato___Early_blight/*.jpg
     ... (all 10)
   real_environment_dataset/         <- your real-world test photos, 10 class folders
     Tomato___Bacterial_spot/*.jpg
     ... (all 10; the 2 excluded ones are simply not linked in - see the wiring cell)
   backgrounds_generic_real/*.jpg    <- REQUIRED: the real CC0 background photos (bgrand_real + seedrep-real need these)
   backgrounds_generic_synthetic/*.jpg   <- OPTIONAL: can be generated on Kaggle if omitted
   ```
2. Kaggle -> **Datasets -> New Dataset** -> upload the zip -> make it **Private** ->
   give it a title (e.g. `tomato-fyp-data`). Note the path it mounts at, which looks
   like `/kaggle/input/tomato-fyp-data`.

> The 2 excluded classes may stay inside `raw/` and `real_environment_dataset/` in the
> upload - they are omitted at split time and not linked into the eval view, so you
> don't need to hand-filter anything. To turn them back **on** later, empty `EXCLUDE`
> in `run_all_local.sh` and drop the skip in the wiring cell.

---

## B. New Notebook settings

1. Kaggle -> **Code -> New Notebook**.
2. Right sidebar: **Accelerator = GPU T4 x2** (or P100), **Internet = On**.
3. **Add Input** -> your `tomato-fyp-data` dataset.

---

## C. The notebook cells

Paste these as separate cells. Replace the two paths marked `<-- CHANGE`.

### Cell 1 - get the code
```python
# If your GitHub repo is PUBLIC:
!git clone https://github.com/SiaJiaLe/AI-Based-Tomato-Leaf-Disease-Detection-Using-Image-Recognition.git /kaggle/working/repo
%cd /kaggle/working/repo
!git log --oneline -1
```
If the repo is **private**, either use a token
(`https://<TOKEN>@github.com/...`) or upload the repo itself as a second private
Kaggle Dataset and `!cp -r /kaggle/input/<repo-dataset>/* /kaggle/working/repo/`.

### Cell 2 - install the pinned dependencies
```python
# torch first, from the CUDA 12.1 index, so we get the GPU build (not CPU):
!pip install -q torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
# then the rest (pins force numpy<2 + albucore, avoiding the two env traps):
!pip install -q -r experiments/requirements.txt
!pip install -q -r experiments/plan1_bgrand/requirements.txt
import torch; print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
```
`CUDA available: True` must print. If it says `False`, the Accelerator isn't set to GPU.

### Cell 3 - download the segmentation model once (needs Internet)
```python
from rembg import new_session
new_session("u2net")           # caches U^2-Net so bgrand/seedrep can segment offline
print("u2net ready")
```

### Cell 4 - wire the data (symlinks + the 8-class real-world view)
```python
import os, glob

REPO = "/kaggle/working/repo"
EXCLUDE = {"Tomato___Target_Spot", "Tomato___Tomato_mosaic_virus"}

# Auto-find the folder that actually holds raw/ + real_environment_dataset/.
# Kaggle mounts a dataset at /kaggle/input/<slug> (NOT the website URL). If you
# zipped WITH a data/ prefix, the folders sit under .../<slug>/data - this checks
# one level deeper too, so either layout works.
def find_data_root():
    for base in sorted(glob.glob("/kaggle/input/*")) + sorted(glob.glob("/kaggle/input/*/*")):
        if (os.path.isdir(os.path.join(base, "raw"))
                and os.path.isdir(os.path.join(base, "real_environment_dataset"))):
            return base
    raise RuntimeError(
        "Could not find raw/ + real_environment_dataset/ under /kaggle/input.\n"
        "Run  !find /kaggle/input -maxdepth 3 -type d  and set DATA_ROOT by hand below.")

DATA_ROOT = find_data_root()          # or hard-code it, e.g. "/kaggle/input/tomato-fyp-data"
print("DATA_ROOT =", DATA_ROOT)

data = os.path.join(REPO, "data"); os.makedirs(data, exist_ok=True)
def link(src, dst):
    if not (os.path.islink(dst) or os.path.exists(dst)):
        os.symlink(src, dst)

# whole-folder symlinks into the read-only input (fine to read through a link)
for name in ["raw", "backgrounds_generic_real"]:
    src = os.path.join(DATA_ROOT, name)
    assert os.path.isdir(src), f"MISSING in your dataset: {src}"
    link(src, os.path.join(data, name))

# synthetic backgrounds: link if you uploaded them, else make a writable empty dir
# (run_all_local.sh will generate them there)
syn_src = os.path.join(DATA_ROOT, "backgrounds_generic_synthetic")
syn_dst = os.path.join(data, "backgrounds_generic_synthetic")
if os.path.isdir(syn_src):
    link(syn_src, syn_dst)
else:
    os.makedirs(syn_dst, exist_ok=True); print("No synthetic bg uploaded - will generate on Kaggle.")

# 8-class real-world eval view: per-class symlinks, skipping the 2 excluded classes.
# (evaluate.py maps the real-world folder into the training label space BY NAME, so an
#  8-class model must see an 8-class eval set or it KeyErrors.)
real_src = os.path.join(DATA_ROOT, "real_environment_dataset")
real_dst = os.path.join(data, "real_environment_dataset"); os.makedirs(real_dst, exist_ok=True)
kept = [c for c in sorted(os.listdir(real_src))
        if c not in EXCLUDE and os.path.isdir(os.path.join(real_src, c))]
for c in kept:
    link(os.path.join(real_src, c), os.path.join(real_dst, c))
print(f"Real-world eval classes (should be 8): {len(kept)}")

# writable working dirs the pipeline expects
os.makedirs(os.path.join(data, "processed"), exist_ok=True)
os.makedirs(os.path.join(data, "mask_cache"), exist_ok=True)
print("Data wired.")
```

### Cell 5 - (RESUME) restore results from your previous session
Skip this the **first** time. On later sessions, first **Add Input -> Your Work ->
the previous version's Output** of this notebook, then set the path below:
```python
import os, shutil
PREV_RESULTS = "/kaggle/input/<previous-version-output>/experiments/results"   # <-- CHANGE each session
dst = os.path.join("/kaggle/working/repo", "experiments", "results")
if os.path.isdir(PREV_RESULTS):
    shutil.copytree(PREV_RESULTS, dst, dirs_exist_ok=True)
    print("Restored previous results - finished runs will be skipped.")
else:
    print("No previous results found (first session) - starting fresh.")
```

### Cell 6 - run the study
```python
# Optional first: see the plan and what would be skipped, without training:
!bash experiments/run_all_local.sh --dry-run
```
```python
# The real run (this is the long one - trains, evaluates, post-processes):
!bash experiments/run_all_local.sh
```

### Cell 7 - look at the results
```python
!cat experiments/results/all_results.md
!ls experiments/results/confusion/
```

---

## D. The multi-session recipe (how ~24 runs fit into 12-hour slices)

1. When a session is close to the 12-hour limit (or you want it to run unattended),
   use **Save Version -> Save & Run All (Commit)**. Kaggle re-runs the notebook
   top-to-bottom headless (up to 12 h) and saves the final `/kaggle/working` as that
   version's **Output**.
2. Start the **next** session on the same notebook. In **Add Input**, add **your own
   previous version's output**, update `PREV_RESULTS` in Cell 5, and run all cells
   again. The runner sees the finished runs and **skips** them, continuing with what's
   left.
3. Repeat until Cell 7's `all_results.md` lists every row and
   `experiments/results/seedrep_summary.json` exists.

Notes:
- **Watch the weekly 30-hour GPU quota** (Kaggle shows it under your avatar). If you
  hit it, training pauses until the weekly reset; your saved results carry over.
- **Output size**: the `best_model.pth` checkpoints dominate (VGG16 is biggest). The
  full set is a few GB - within Kaggle's limits. They must persist because Plan 2's
  winner-selection and the confusion-matrix step re-read the checkpoints.
- If `rembg` ever fails to import, the runner **skips only** the background blocks
  (bgrand / bgrand_real / seedrep / mixstyle-combo) and still completes the ablation
  and Plan 2 Tiers 1-2. Re-running after fixing rembg fills the rest in.

---

## What "excluded from training" means here (verification)

- **Split:** `run_all_local.sh` calls
  `split_dataset --exclude Tomato___Target_Spot Tomato___Tomato_mosaic_virus`, so those
  two never enter `data/processed/{train,val,test}` - the models never train or
  validate on them.
- **Evaluation:** Cell 4 builds `data/real_environment_dataset` from only the other 8
  classes, so the real-world metrics are computed over 8 classes and never KeyError.
- Their raw images stay on disk (in your uploaded dataset), untouched - the exclusion
  is fully reversible.
