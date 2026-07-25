# EfficientViT-B0 TRY-OUT on Google Colab (NOT part of the research)

This trains **one** Vision Transformer (`efficientvit_b0`) on your 8-class tomato split,
in **both** augmentation conditions (OFF and ON), and evaluates each on the controlled
PlantVillage test set **and** the real-world set. It retrains **nothing else** and touches
no existing study result — it only adds `results/efficientvit_b0_off/` and
`results/efficientvit_b0_on/`.

> This is exploratory. It is **not** a thesis result and every artifact is labelled
> `tryout`. The two runs differ only in the advanced-augmentation toggle (the same OFF/ON
> variable your CNN ablation isolates); CBAM and stage-group unfreezing are CNN-only and
> don't apply to a transformer.

---

## Runtime type
**Runtime → Change runtime type → T4 GPU** (A100 if you have Pro — faster). Two short runs.

---

## The cells

### Cell 1 — GPU check + get the code
```python
!nvidia-smi -L
%cd /content
!rm -rf /content/repo
!git clone https://github.com/SiaJiaLe/AI-Based-Tomato-Leaf-Disease-Detection-Using-Image-Recognition.git /content/repo
%cd /content/repo
!git log --oneline -1
```

### Cell 2 — install dependencies (same pins as the study; no new packages)
```python
!pip install -q torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
!pip install -q -r experiments/requirements.txt
import torch; print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available())
```
`CUDA: True` must print.

> **After Cell 2, restart the session once** (Runtime → Restart session), then continue
> from **Cell 3**. `experiments/requirements.txt` pins numpy 1.26.4 but the live kernel
> still has the old numpy loaded, so the next import can fail with
> `numpy.dtype size changed ...`. Pip installs persist across a restart, so do NOT re-run
> Cell 1 or Cell 2. (The Drive mount does not persist — Cell 3 re-mounts it.)

### Cell 3 — mount Google Drive
```python
from google.colab import drive
drive.mount("/content/drive")
```

### Cell 4 — point `data/` at your Drive data + build the 8-class split
```python
import os, shutil

REPO  = "/content/repo"
DRIVE = "/content/drive/MyDrive/tomato_fyp"          # <-- your Drive folder
SRC   = os.path.join(DRIVE, "data")                  # extracted data/ on Drive
EXCLUDE = {"Tomato___Target_Spot", "Tomato___Tomato_mosaic_virus"}
IMG = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

data = os.path.join(REPO, "data")
raw  = os.path.join(data, "raw")
real = os.path.join(data, "real_environment_dataset")

def class_images(d):
    if not os.path.isdir(d): return 0
    return sum(sum(1 for f in os.listdir(os.path.join(d, c)) if f.lower().endswith(IMG))
               for c in os.listdir(d) if os.path.isdir(os.path.join(d, c)))

# Copy raw + real from Drive -> local disk if missing (raw rebuilds the split;
# real is the held-out eval set).
os.makedirs(data, exist_ok=True)
for name in ("raw", "real_environment_dataset"):
    dst = os.path.join(data, name)
    if class_images(dst) == 0:
        s = os.path.join(SRC, name)
        assert os.path.isdir(s), f"Not found on Drive: {s}"
        print("Copying", name, "from Drive...")
        shutil.copytree(s, dst, dirs_exist_ok=True)

# 8-class real-world view: move the 2 excluded classes aside (reversible).
excl = os.path.join(data, "real_environment_dataset_excluded"); os.makedirs(excl, exist_ok=True)
for c in EXCLUDE:
    p = os.path.join(real, c)
    if os.path.isdir(p):
        shutil.move(p, os.path.join(excl, c)); print("Moved out of eval set:", c)

assert class_images(raw)  > 0, "RAW training set has no images!"
assert class_images(real) > 0, "REAL eval set has no images!"
print("raw classes:", len([c for c in os.listdir(raw) if os.path.isdir(os.path.join(raw, c))]),
      "| real classes:", len([c for c in os.listdir(real) if os.path.isdir(os.path.join(real, c))]))
```
```python
# Build data/processed/{train,val,test} — the SAME 8-class, seed-42 split the study uses.
!cd /content/repo && python -m experiments.split_dataset \
    --exclude Tomato___Target_Spot Tomato___Tomato_mosaic_virus --skip-if-exists
!ls /content/repo/data/processed
```
Last line must list `train`, `val`, `test`.

### Cell 5 — point `results` at Drive (so the try-out survives the runtime)
```python
import os
DRIVE_RESULTS = "/content/drive/MyDrive/tomato_fyp/results"
os.makedirs(DRIVE_RESULTS, exist_ok=True)
link = "/content/repo/experiments/results"
if os.path.islink(link): os.remove(link)
elif os.path.isdir(link) and not os.listdir(link): os.rmdir(link)
os.symlink(DRIVE_RESULTS, link)
print("results ->", os.path.realpath(link))
```
> If you archived your study results, this creates a fresh live `results/`. The try-out
> only ADDS `efficientvit_b0_off/` and `efficientvit_b0_on/` — it overwrites nothing.

### Cell 6 — train + evaluate BOTH conditions
```python
# ~a few minutes each on a T4. Watch for the [A ...]/[B ...] epoch lines, then the
# "controlled acc ... f1 ... | real-world acc ... f1 ... | gap ..." summary per run.
!cd /content/repo && python -m experiments.tryout_efficientvit.train_efficientvit \
    --config experiments/tryout_efficientvit/config_off.yaml
!cd /content/repo && python -m experiments.tryout_efficientvit.train_efficientvit \
    --config experiments/tryout_efficientvit/config_on.yaml
```

### Cell 7 — read the results
```python
import json
for run in ("efficientvit_b0_off", "efficientvit_b0_on"):
    base = f"/content/repo/experiments/results/{run}"
    c = json.load(open(f"{base}/eval_results.json"))
    r = json.load(open(f"{base}/eval_results_real_world.json"))
    print(f"\n=== {run} ===")
    print(f"  controlled : acc {c['accuracy']:.4f}  macro-F1 {c['macro_f1']:.4f}")
    print(f"  real-world : acc {r['accuracy']:.4f}  macro-F1 {r['macro_f1']:.4f}")
    print(f"  gen. gap   : acc {r['generalization_gap_accuracy']:+.4f}  "
          f"macro-F1 {r['generalization_gap_macro_f1']:+.4f}")
```
Confusion matrices are at `results/efficientvit_b0_{off,on}/cm_{controlled,real_world}_test.png`.

---

## How to read it
- These are **try-out** numbers — a quick look at whether a ViT backbone behaves
  differently on your data. Compare the real-world macro-F1 and the generalization gap
  against your EfficientNet-B0 runs, but do **not** fold them into the pre-registered
  ablation tables.
- The OFF vs ON contrast here tells you whether the advanced field-condition augmentation
  helps a transformer the way it did (or didn't) for the CNNs.

## Resuming after a disconnect
Re-run Cells 1–5, then re-run Cell 6. `--skip-if-exists` makes the re-split instant. To only
re-score an existing checkpoint, add `--eval-only` to the Cell 6 commands.
