# Running ONLY Plan 3 (DINOv2 + ChromaDB fusion) on Google Colab

This runs the **one** Plan 3 experiment - it does **not** retrain any model. Plan 3
trains nothing: it loads your frozen `efficientnetb0_on_bgrand_real` (seed 42)
checkpoint, builds a ChromaDB index over the PlantVillage **train** embeddings from
DINOv2, and evaluates three new rows (retrieval / fusion / fusion+abstention) against
the fixed baseline. See `plan3_chromadb_retrieval_fusion.md` for the design and the
pre-registered decision rule.

## What Plan 3 needs to find

| Needs | Where it comes from |
|---|---|
| `data/processed/{train,val,test}` | rebuilt by the split cell (same 8-class split as the study) |
| `data/real_environment_dataset` (8-class) | unzipped from your Drive `data.zip`, 2 classes moved aside |
| the `efficientnetb0_on_bgrand_real` checkpoint | **auto-located** in `results/` OR any `results_archive_*/` |

> **You can archive your results first - Plan 3 still works.** Cell 5 searches every
> `results_archive_*` folder for the `bgrand_real` checkpoint and uses the newest one.
> Plan 3 only ADDS three new row folders (`plan3_retrieval`, `plan3_fusion`,
> `plan3_fusion_abstain`) + `plan3_summary.json`; it overwrites nothing.

---

## Runtime type

**Runtime -> Change runtime type -> T4 GPU** (A100 if you have Pro - faster, but T4 is
fine; Plan 3 is inference-only and takes minutes, not hours).

---

## The cells

Paste each as its own cell and run in order.

### Cell 1 - GPU check + get the code (clean checkout)
```python
!nvidia-smi -L
%cd /content
!rm -rf /content/repo
!git clone https://github.com/SiaJiaLe/AI-Based-Tomato-Leaf-Disease-Detection-Using-Image-Recognition.git /content/repo
%cd /content/repo
!git log --oneline -1
```
`nvidia-smi -L` must list a GPU. The last line must print a commit.

### Cell 2 - install dependencies (+ chromadb)
```python
!pip install -q torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
!pip install -q -r experiments/requirements.txt
!pip install -q -r experiments/plan3_chromadb/requirements.txt
import torch; print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available())
```
`CUDA: True` must print.

> **After Cell 2, restart the session once.** `experiments/requirements.txt` pins
> numpy 1.26.4, but the live kernel still has the old numpy loaded, so the next import
> can fail with `numpy.dtype size changed ... Expected 96, got 88`. Do **Runtime ->
> Restart session**, then continue from **Cell 3** - pip installs persist across a
> restart, so you do NOT re-run Cell 1 or Cell 2. (The Drive mount does NOT persist,
> which is why Cell 3 re-mounts it.)

Plan 3 does **not** use `rembg` / u2net (it trains nothing), so there is no segmentation
cell here.

### Cell 3 - mount Google Drive
```python
from google.colab import drive
drive.mount("/content/drive")
```
Authorize in the popup. This is also where the DINOv2 weights cache and the DINO
embedding cache live, so they survive between sessions.

### Cell 4 - unzip data + build the 8-class split
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

def class_images(d):
    if not os.path.isdir(d):
        return 0
    return sum(sum(1 for f in os.listdir(os.path.join(d, c)) if f.lower().endswith(IMG))
               for c in os.listdir(d) if os.path.isdir(os.path.join(d, c)))

def resolve_root(base):
    if class_images(base) > 0:
        return base
    for sub in (sorted(os.listdir(base)) if os.path.isdir(base) else []):
        p = os.path.join(base, sub)
        if os.path.isdir(p) and class_images(p) > 0:
            return p
    return base

# Unzip raw + real from Drive -> local disk if either is empty. (Plan 3 needs raw to
# rebuild the split, real for evaluation, and the background photos for generating
# composites on the fly during indexing.)
need = (class_images(raw) == 0 or class_images(resolve_root(real)) == 0)
if need:
    assert os.path.isfile(ZIP), f"Zip not found on Drive: {ZIP}"
    print("Unzipping data from Drive to local disk (a few minutes)...")
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
    shutil.rmtree(real, ignore_errors=True)
    shutil.move(tmp, real)

# 8-class real-world view: move the 2 excluded classes aside (reversible; zip untouched).
excl = os.path.join(data, "real_environment_dataset_excluded"); os.makedirs(excl, exist_ok=True)
for c in EXCLUDE:
    p = os.path.join(real, c)
    if os.path.isdir(p):
        shutil.move(p, os.path.join(excl, c)); print("Moved out of eval set:", c)

assert class_images(raw)  > 0, "RAW training set has no images - check the zip layout!"
assert class_images(real) > 0, "REAL eval set has no images - check the zip layout!"
print("raw classes:", len([c for c in os.listdir(raw) if os.path.isdir(os.path.join(raw, c))]),
      "| real classes:", len([c for c in os.listdir(real) if os.path.isdir(os.path.join(real, c))]))
```
```python
# Build data/processed/{train,val,test} - the SAME 8-class, seed-42 split the study used.
# --skip-if-exists makes this instant if the split already exists on this session.
!cd /content/repo && python -m experiments.split_dataset \
    --exclude Tomato___Target_Spot Tomato___Tomato_mosaic_virus --skip-if-exists
!ls /content/repo/data/processed
```
The last line must list `train`, `val`, `test`.

### Cell 5 - locate the bgrand_real checkpoint + point results at Drive
```python
import os, glob, shutil

DRIVE = "/content/drive/MyDrive/tomato_fyp"

# Find the seed-42 bgrand_real checkpoint in the live results OR any archive folder.
cands = glob.glob(os.path.join(DRIVE, "results", "efficientnetb0_on_bgrand_real", "best_model.pth")) \
      + glob.glob(os.path.join(DRIVE, "results_archive_*", "efficientnetb0_on_bgrand_real", "best_model.pth"))
assert cands, ("No efficientnetb0_on_bgrand_real/best_model.pth found under results/ or "
               "results_archive_*/. Plan 3 needs that trained checkpoint.")
cands.sort(key=os.path.getmtime)
CKPT_DIR = os.path.dirname(cands[-1])         # newest wins
print("Using checkpoint:", CKPT_DIR)

# Live results/ (where Plan 3 writes its 3 new rows). Created fresh if you archived.
DRIVE_RESULTS = os.path.join(DRIVE, "results")
os.makedirs(DRIVE_RESULTS, exist_ok=True)
link = "/content/repo/experiments/results"
if os.path.islink(link):
    os.remove(link)
elif os.path.isdir(link) and not os.listdir(link):
    os.rmdir(link)
os.symlink(DRIVE_RESULTS, link)
print("results ->", os.path.realpath(link))

# Hand CKPT_DIR to the next cell (subprocess) via an env var.
os.environ["PLAN3_CKPT_DIR"] = CKPT_DIR
```

### Cell 6 - run Plan 3
```python
# DINOv2 weights (~85 MB) cache to Drive so they aren't re-downloaded each session.
%env TORCH_HOME=/content/drive/MyDrive/tomato_fyp/torch_cache
```
```python
# The whole Plan 3 pipeline: build index -> sanity check -> calibrate -> validation
# grid search -> read real-world ONCE -> novelty AUROC -> verdict. Takes a few minutes.
#   - chroma index is built locally (fast, rebuilt from cached embeddings each session)
#   - DINO embeddings are cached to Drive, so a resumed session skips re-embedding
!cd /content/repo && python -m experiments.plan3_chromadb.run_plan3 \
    --ckpt-dir "$PLAN3_CKPT_DIR" \
    --chroma-path /content/chroma_store \
    --cache-dir  /content/drive/MyDrive/tomato_fyp/dino_cache
```
Watch for, in order: `sanity check PASSED`, `class alignment OK`, the calibration line
(`ECE raw ... -> T=... -> ECE calibrated ...`), the three `REAL-WORLD macro-F1` lines,
the novelty `AUROC`, and the final `=== VERDICT ===` block.

### Cell 7 - read the results
```python
import json
s = json.load(open("/content/repo/experiments/results/plan3_summary.json"))
print(json.dumps(s, indent=2))
print("\nRows written:")
!ls /content/repo/experiments/results/plan3_retrieval \
     /content/repo/experiments/results/plan3_fusion \
     /content/repo/experiments/results/plan3_fusion_abstain
```
`plan3_summary.json` also lands on Drive at `MyDrive/tomato_fyp/results/plan3_summary.json`,
so it survives the runtime.

---

## How to read the outcome

The pre-registered rule (plan §1): **fusion is adopted only if real-world macro-F1 beats
the `bgrand_real` 3-seed mean (0.4641) by more than +0.03.** The `VERDICT` block prints
`best - baseline_mean` and says ADOPT or NEGATIVE.

- **A NEGATIVE is a valid, expected result** - the plan says so (§8). It does not touch
  your headline finding (real vs synthetic backgrounds, +0.1146). Report it as a
  tried-and-bounded negative.
- **The novelty AUROC (§7) is reported regardless.** If it's meaningfully above 0.5,
  "retrieval distance predicts CNN failure on out-of-distribution inputs" is a genuine
  finding on its own, even if fusion fails.
- **Calibration (T, ECE)** is a legitimate figure for the report - it explains *why*
  fusion does or doesn't help.

## Resuming after a disconnect

Plan 3 is short, but if it drops: re-run **Cells 1-6**. The DINO embedding cache on Drive
makes the second run fast (no re-embedding), and `--skip-if-exists` skips the re-split.
If Plan 3 already finished, its rows and `plan3_summary.json` are on Drive - just re-run
Cell 7 after re-mounting (Cell 3) and re-linking (Cell 5).
```
