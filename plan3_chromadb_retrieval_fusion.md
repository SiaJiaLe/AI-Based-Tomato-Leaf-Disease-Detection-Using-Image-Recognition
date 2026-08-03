# Plan 3 — DINOv2 + ChromaDB Retrieval and CNN Fusion

**Owner:** Sia Jia Le (22062566)
**Baseline to beat:** `efficientnetb0_on_bgrand_real` — real-world macro-F1 **0.4641 ± 0.0312** (3-seed), 8-class test set (333 images before class removal; use current cleaned set).
**Status:** exploratory. Adopt only if it clears the pre-registered threshold in §1.

> **Why a different encoder.** If Chroma stores embeddings from your own fine-tuned EfficientNetB0, retrieval is k-NN on the exact representation the classifier head already sits on. When the CNN misclassifies a field image, the embedding is already in the wrong region, so retrieval fetches the same wrong neighbours — correlated errors, no new information. Using **DINOv2** (self-supervised, trained on a large diverse image corpus, no text objective) gives an *independent* representation whose failures are uncorrelated with your CNN's. That independence is the entire premise of fusion. DINOv2 is preferred over CLIP because leaf disease is a fine-grained texture problem, not a caption-level semantic one.

---

## 1. Pre-registered decision rule (write this down BEFORE running anything)

> Fusion (CNN + Chroma) is adopted as a contribution **only if** real-world macro-F1 exceeds `bgrand_real` by **more than +0.03** on the same test set. Below +0.03 it is reported as a tried-and-bounded negative.

Rationale: your seed spread is ±0.02–0.03. Anything smaller is noise. `droppath02` flipped sign (−0.0092 → +0.0025) between test-set versions — that is what sub-threshold "gains" are worth.

**Secondary outcome, reported regardless of the above:** does raw nearest-neighbour distance flag the field images the CNN gets wrong? (See §7.) This can be a genuine finding even if fusion fails.

---

## 2. Non-negotiable hygiene

1. **Database contains PlantVillage TRAIN split embeddings only.** Never val, never test, never real-world. Storing test embeddings makes controlled numbers meaningless (the answer is literally in the database); storing field embeddings breaks condition 1 and invalidates every generalization claim in the project.
2. **All parameters (`k`, `temp`, `w`, `τ`) selected on the PlantVillage VALIDATION split.** Real-world set read **once** per final row.
3. **One variable per row.** Row A (retrieval alone) and Row B (fusion) are separate rows against the fixed `bgrand_real` baseline.
4. Same 8-class label space, same frozen split (`data/processed`), seed 42 — the same seed used by `common/seeding.py` across all prior ablation and replication runs, kept for consistency rather than for any property of the number itself.

   **What the seed actually controls here.** This plan trains nothing, so the seed does less work than in Plans 1–2. DINOv2 extraction is deterministic (model in `.eval()`, no dropout, no augmentation), and the CNN checkpoints already exist from the `bgrand_real` runs (seeds 42/43/44). The one genuine source of run-to-run variation is **ChromaDB's HNSW index construction**, which is stochastic — so the same query can return slightly different approximate neighbours between rebuilds. Set the seed before building the index, and if a result looks marginal, rebuild the index and re-check rather than assuming the number is exact.

   **Which CNN checkpoint to fuse with.** Use the **seed-42** `bgrand_real` checkpoint for the main fusion rows so the CNN side is fixed and the comparison is clean. If fusion clears the §1 threshold, repeat Row B across the seed-43 and seed-44 checkpoints to confirm the gain is not specific to one trained model — the same replication discipline that validated the headline result.
5. Assert in code that no file path under the Chroma ingest list resolves to `val/`, `test/`, or the real-environment directory.

---

## 3. Environment and setup

### 3.1 Install

```bash
pip install chromadb
# torch / torchvision assumed present from the existing pipeline
```

DINOv2 model: `dinov2_vits14` (ViT-S/14, 384-dim embeddings). Start small — if the effect exists it will show at ViT-S. Only escalate to `dinov2_vitb14` if ViT-S shows promise, and report it as a separate row.

**Weights download.** `torch.hub.load` fetches DINOv2 weights (~85 MB) on first call and needs internet access. On Colab this re-downloads every session unless the hub cache is redirected to persistent storage:

```python
import os, torch
os.environ["TORCH_HOME"] = "/content/drive/MyDrive/torch_cache"   # Colab
```

**Preprocessing note:** DINOv2 uses patch size 14, so input dimensions must be multiples of 14. Use **224×224** (16 patches) — conveniently the same size as your CNN pipeline. Normalize with ImageNet mean/std (same constants you already use). No augmentation at embed time.

### 3.2 Storage

The Chroma store persists to `chroma_store/`. With ~4,750 training images at 384 dims it is small (tens of MB), but **on Colab it is destroyed when the runtime ends**. Either rebuild the index each session (a few minutes) or write it to Drive. If rebuilding, note the HNSW determinism caveat in §2.4.

### 3.3 Paths to set

Point these at your actual directories before running anything:

| Variable | Value |
|---|---|
| `TRAIN_DIR` | `data/processed/train` |
| `VAL_DIR` | `data/processed/val` |
| `TEST_DIR` | `data/processed/test` |
| `REAL_DIR` | your real-environment directory |
| `CKPT` | the seed-42 `bgrand_real` checkpoint |

### 3.4 CRITICAL — class index alignment

**This is the failure mode most likely to produce plausible-looking garbage.**

`CLASSES = sorted(os.listdir(TRAIN_DIR))` builds the Chroma label→index map. Your CNN's softmax vector has its own ordering, from `ImageFolder` (which also sorts alphabetically, so they *should* match). But if they differ by even one position, `fuse(p_cnn, s_chroma)` adds one class's CNN probability to a different class's Chroma score — and **no error is raised**. You get numbers that look reasonable and are meaningless.

Assert it explicitly before any fusion runs:

```python
assert CLASSES == train_dataset.classes, (CLASSES, train_dataset.classes)
```

### 3.5 Retrieval sanity check (run BEFORE writing any fusion code)

Embed 10 images that are already **in** the database, query them back, and confirm each returns itself as nearest neighbour at cosine distance ≈ 0:

```python
probe = paths[:10]
e = embed_batch(probe)
for p, v in zip(probe, e):
    r = col.query(query_embeddings=[v.tolist()], n_results=1)
    print(round(r["distances"][0][0], 5), os.path.basename(p))
    # expect ~0.00000
```

If distances are not near zero, something is wrong with L2-normalisation, the distance metric, or the index — find out now, not after building the fusion layer on top.

---

## 4. Technical implementation

> **Scope of the code below.** These are building blocks, not a runnable pipeline. The agent must still write: (a) checkpoint loading and CNN inference to produce softmax vectors on val/test/real sets; (b) the evaluation loop that, per image, embeds with DINOv2 → queries Chroma → fuses → records the prediction; (c) metric computation — reuse the **existing evaluator** so numbers are directly comparable to prior rows, do not write a new one; (d) the validation grid search over `k`, `temp`, `w`, `τ`, `d_max`; (e) the calibration check in §4.5; (f) the novelty-signal AUROC analysis in §7.

### 4.1 Embedding extraction

```python
import torch, numpy as np
from PIL import Image
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(DEVICE).eval()

tf = transforms.Compose([
    transforms.Resize((224, 224)),          # multiple of patch size 14
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

@torch.no_grad()
def embed_batch(paths, batch_size=32):
    """Return L2-normalised (N, 384) float32 embeddings."""
    out = []
    for i in range(0, len(paths), batch_size):
        imgs = [tf(Image.open(p).convert("RGB")) for p in paths[i:i+batch_size]]
        x = torch.stack(imgs).to(DEVICE)
        f = dino(x)                                  # (B, 384) CLS feature
        f = torch.nn.functional.normalize(f, dim=1)  # unit norm -> cosine space
        out.append(f.cpu().numpy().astype("float32"))
    return np.concatenate(out, 0)
```

**L2-normalisation is required.** With unit-norm vectors, Chroma's cosine distance is well-behaved and distances are comparable across queries — which the abstention logic depends on.

### 4.2 Building the Chroma database

```python
import chromadb, os, glob

TRAIN_DIR = "data/processed/train"     # TRAIN ONLY
FORBIDDEN = ("/val/", "/test/", "real_environment")

client = chromadb.PersistentClient(path="chroma_store")
col = client.get_or_create_collection(
    name="plantvillage_train_dinov2",
    metadata={"hnsw:space": "cosine"},   # cosine distance
)

paths, labels = [], []
for cls in sorted(os.listdir(TRAIN_DIR)):
    for p in glob.glob(os.path.join(TRAIN_DIR, cls, "*")):
        assert not any(f in p.replace("\\", "/") for f in FORBIDDEN), f"LEAK: {p}"
        paths.append(p); labels.append(cls)

emb = embed_batch(paths)
B = 512
for i in range(0, len(paths), B):
    col.add(
        ids=[f"tr_{j}" for j in range(i, min(i+B, len(paths)))],
        embeddings=emb[i:i+B].tolist(),
        metadatas=[{"label": l} for l in labels[i:i+B]],
    )
print("indexed", col.count(), "train embeddings")
```

### 4.3 Query → class scores (the non-obvious part)

Chroma returns *distances*. To fuse with softmax you need a probability-like vector over classes. Similarity-weighted voting:

```python
CLASSES = sorted(os.listdir(TRAIN_DIR))
IDX = {c: i for i, c in enumerate(CLASSES)}

def chroma_scores(query_emb, k=10, temp=0.3):
    """Return (scores over classes, raw nearest distance)."""
    r = col.query(query_embeddings=[query_emb.tolist()], n_results=k)
    dists = np.asarray(r["distances"][0])            # cosine distance
    labs  = [m["label"] for m in r["metadatas"][0]]

    sims = 1.0 - dists                               # cosine similarity
    w = np.exp(sims / temp); w /= w.sum()

    s = np.zeros(len(CLASSES))
    for wi, l in zip(w, labs):
        s[IDX[l]] += wi
    return s, float(dists.min())                     # nearest distance = novelty signal
```

> **Verified design point — do not skip.** The normalised score vector **saturates**: with `temp=0.1` it returns max≈1.000 whether the nearest neighbour is at distance 0.05 (confidently known) or the votes are split among far neighbours. Tested across temp ∈ {0.02, 0.1, 0.3, 1.0}, max score stayed ≥0.89 in all cases. **Therefore the normalised score cannot serve as a novelty/abstention signal.** Abstention must use the **raw nearest distance**, returned separately above. Use `temp≈0.3` as a starting point (less saturated than 0.1) and tune on validation.

### 4.4 Fusion

```python
def fuse(p_cnn, s_chroma, w=0.6):
    """p_cnn: softmax vector. s_chroma: from chroma_scores(). Both sum to 1."""
    return w * p_cnn + (1.0 - w) * s_chroma

def predict(p_cnn, s_chroma, near_dist, w=0.6, tau=0.45, d_max=0.75):
    f = fuse(p_cnn, s_chroma, w)
    if f.max() < tau or near_dist > d_max:
        return None, f            # abstain
    return int(f.argmax()), f
```

Two independent abstention triggers — low fused confidence (`τ`) and far nearest neighbour (`d_max`). Both selected on validation.

### 4.5 CNN calibration check (do this before fusing)

Your confusion matrices show confident errors — healthy → Late_blight at 33%, Bacterial_spot → Septoria at 50%. Fusing on miscalibrated softmax will systematically over-trust the CNN exactly when it is confidently wrong.

- [ ] On the PlantVillage **validation** split, bin predictions by confidence (10 bins) and plot mean confidence vs actual accuracy. Report Expected Calibration Error.
- [ ] If badly miscalibrated, fit **temperature scaling** on validation (single scalar `T`, divide logits by `T` before softmax) and use calibrated probabilities in fusion.
- [ ] Report the calibration figure — it is a legitimate contribution on its own and explains *why* fusion does or doesn't work.

---

## 5. Parameter selection (VALIDATION ONLY)

Grid, all evaluated on PlantVillage validation macro-F1:

| Param | Candidates | Used by |
|---|---|---|
| `k` | 1, 5, 10, 20 | Row A, Row B |
| `temp` | 0.1, 0.3, 1.0 | Row A, Row B |
| `w` | 0.0, 0.25, 0.5, 0.75, 1.0 | Row B |
| `τ` | 0.35, 0.45, 0.55 | Row B abstention |
| `d_max` | 60th/75th/90th percentile of validation nearest-distance | Row B abstention |

`w=0.0` (pure Chroma) and `w=1.0` (pure CNN) are included deliberately as sanity anchors — if validation picks an endpoint, fusion adds nothing and you have your answer early.

**Discipline:** freeze every parameter from validation, then read real-world once. Three tunable knobs plus a visible per-class failure pattern makes this the most tuning-prone experiment in the project. If you catch yourself adjusting `w` while looking at field numbers, stop — that is the moment a clean zero-shot result becomes a fitted one.

---

## 6. Rows to produce

| Row | Description | Real-world reads |
|---|---|---|
| Baseline | `efficientnetb0_on_bgrand_real` (existing) | already done |
| **A** | Chroma retrieval alone (DINOv2, no CNN) | once |
| **B** | CNN + Chroma fusion, no abstention | once |
| **B-abs** | Fusion with abstention (report coverage + accuracy-on-covered) | once |

For each: controlled macro-F1, real-world macro-F1, real-world accuracy, gap, per-class F1, confusion matrix.

**Abstention reporting:** never report accuracy on the covered subset alone — that inflates trivially. Always report **coverage** (% of images answered) alongside it, and macro-F1 over the full set treating abstentions as errors.

---

## 7. The secondary finding (report regardless of §1 outcome)

Test whether retrieval distance is a usable novelty signal under domain shift:

- [ ] For every real-world test image, record raw nearest distance and whether the CNN's prediction was correct.
- [ ] Compare mean nearest distance for CNN-correct vs CNN-incorrect images.
- [ ] Compute AUROC using nearest distance as a score for predicting CNN error.

If AUROC is meaningfully above 0.5, you have a real result: *"retrieval distance in a domain-robust embedding space predicts classifier failure on out-of-distribution inputs."* That is the thing retrieval is structurally best at, and it is worth writing up even if fusion never clears +0.03.

It also gives a deployment argument: showing a farmer the three most similar known cases is more trustworthy than a bare softmax number.

---

## 8. Honest expectations

- Chroma is populated with PlantVillage embeddings, so field queries are far from everything stored. Retrieval alone may score **poorly** on real-world data — possibly worse than the CNN. That is expected, not a bug.
- The realistic upside is fusion adding a few points via error decorrelation, plus a usable abstention mechanism.
- The likeliest single outcome is **below +0.03**, i.e. a negative. That is fine and consistent with every architecture tier you tested. Your headline finding (real vs synthetic backgrounds, +0.1146 ± 0.0249, all seeds agreeing) does not depend on this working.

---

## 9. Execution order

1. **Setup (§3):** install, set paths, redirect torch hub cache if on Colab.
2. **Build Chroma DB** from the train split with leak assertions (§4.2).
3. **Retrieval sanity check (§3.5)** — query 10 indexed images back, confirm distance ≈ 0. Do not proceed until this passes.
4. **Class alignment assertion (§3.4)** — confirm `CLASSES == train_dataset.classes`.
5. **Calibration check** on the CNN (§4.5) — cheap, and it predicts whether fusion can help at all.
6. **Row A** on validation → select `k`, `temp`. Then real-world once.
7. **Row B** grid on validation → select `w`, `τ`, `d_max`. Then real-world once.
8. **Novelty-signal analysis** (§7).
9. Compare against the §1 threshold. Adopt or report as a negative.
