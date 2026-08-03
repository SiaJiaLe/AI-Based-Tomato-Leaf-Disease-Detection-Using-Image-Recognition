# Model Training & Evaluation — Full Report (Beginning → Now)

**FYP: AI-Based Tomato Leaf Disease Detection — Sia Jia Le (22062566), Sunway University**
**Scope:** the machine-learning study only — dataset, training, and evaluation.
Excludes the web-application/platform development. (The exploratory EfficientViT / Vision
Transformer try-out is deliberately **not** covered here.)

---

## 0. How to read this report

The study evolved through several dataset revisions. **Numbers are only comparable within
the same dataset revision** — never mix them across the boundaries flagged below. Three
revision boundaries matter:

1. **Real-world test set switch** (2026-07-17): `data/processed/real_environment_test`
   (retired) → `data/real_environment_dataset` (new field set).
2. **Full retrain on a new PlantVillage source** (`data/raw`), with an **8-class scope**
   (two classes turned off — see §7).
3. **The final clean end-to-end run on Google Colab (A100)** — the **reportable** numbers.

So early ablation numbers (§5, §6) are historical (retired 10-class set); the **current
reportable results are the 3-seed replication in §8**.

---

## 1. The problem the whole study is about

PlantVillage-style models reach **~96–99 % controlled (lab) accuracy** but **collapse on
real-world field photos** (~40 % accuracy, ~0.31 macro-F1 at the study's midpoint). The
entire project is a controlled investigation of **one question**:

> *What actually reduces the lab→field generalization gap — the model, or the data?*

Every experiment is framed as a **measurement, not a contest**. Two rules were held
throughout:
- **Judge on real-world macro-F1 (and the gap), never lab accuracy.** Lab accuracy is only
  a sanity check.
- **Touch the held-out real-world test set once per model** (read-once). No tuning on it.

---

## 2. Dataset & splits

- **Training / controlled source:** PlantVillage tomato images. Split **70/15/15**
  train/val/test, **seed 42**, frozen on disk as `data/processed/{train,val,test}`
  (`experiments/split_dataset.py`, symlink-based, deterministic).
- **Held-out real-world set:** independent field photographs
  (`data/real_environment_dataset`) — used **only** for the final generalization
  measurement, read once per model.
- **Class scope:**
  - Originally **10 tomato classes** (Bacterial spot, Early blight, Late blight, Leaf Mold,
    Septoria leaf spot, Spider mites, Target Spot, Yellow Leaf Curl Virus, Mosaic virus,
    Healthy).
  - Later reduced to an **8-class scope** (§7): `Target_Spot` and `Tomato_mosaic_virus`
    turned **off** (not deleted) because they were pathological on the new data. The 8-class
    models are a **separate study**, not cell-comparable to the 10-class runs.
- **New-data counts (Colab):** raw ~6,791 imgs / 10 cls (Target_Spot 1,123, mosaic 439 →
  excluded ⇒ ~5,229 train); real-world set **261 imgs / 8 cls**.

---

## 3. Preprocessing & augmentation contract (the control that makes rows comparable)

Implemented in `experiments/common/data.py` (Albumentations). Input size **224×224**;
ImageNet stats `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.

**Inference/eval pipeline (val, test, real-world — identical everywhere)** —
`build_eval_transform(image_size)`:
`A.Resize(256, 256) → A.CenterCrop(224, 224) → A.Normalize(ImageNet) → ToTensorV2`.
A **hard assertion** (`_assert_no_augmentation`) fails loudly if any augmentation ever leaks
into a val/test transform.

**Training augmentation is a two-level contract** — `build_train_transform(image_size, advanced_augmentation)`:

- **Basic four** — `_basic_four()`, applied in *every* run (train only), identical strengths:

  | Op | Exact parameters |
  |---|---|
  | `A.RandomResizedCrop` | `size=(224, 224), scale=(0.8, 1.0)` (default `ratio=(0.75, 1.333)`) |
  | `A.HorizontalFlip` | `p=0.5` |
  | `A.Rotate` | `limit=20, p=0.5` |
  | `A.ColorJitter` | `brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0, p=0.5` |

- **Advanced field-condition block** — `_advanced_block()`, added *on top* for **Stack ON** only:

  | Op | Exact parameters |
  |---|---|
  | `A.Affine` | `scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-15, 15), p=0.4` |
  | `A.Perspective` | `scale=(0.05, 0.1), p=0.2` |
  | `A.ElasticTransform` | `alpha=80, sigma=4.0, p=0.15` |
  | `A.RandomShadow` | `p=0.15` |
  | `A.GaussianBlur` | `blur_limit=(3, 5), p=0.2` |
  | `A.GaussNoise` | `p=0.2` |
  | `A.ImageCompression` | `quality_range=(70, 100), p=0.2` |
  | `A.CoarseDropout` | `num_holes_range=(1, 5), hole_height_range=(1, 15), hole_width_range=(1, 15), p=0.15` |

  Moderate strengths chosen to preserve lesion features. After both levels:
  `A.Normalize(ImageNet) → ToTensorV2`.

Because OFF and ON share one code path, an OFF/ON pair differs **only** by the advanced
block (plus the head/CBAM/unfreeze flags in §6) — exactly the variables being isolated.

---

## 4. Training protocol & evaluation methodology

**Two-stage transfer learning** (`experiments/common/engine.py` → `train_run`), shared by all runs:
- **Optimizer:** `torch.optim.Adam` (plain Adam, **not** AdamW); **no LR scheduler** (fixed
  per-group learning rates). Loss: `nn.CrossEntropyLoss(label_smoothing=…)` (ON 0.1 / OFF 0.0).
  Batch size **32**. Determinism via `seed_everything(seed, deterministic=True)` + `seed_worker`.
- **Stage A** (`freeze_to_head`): backbone frozen, train the new classifier head only.
  `Adam(head, lr=1e-3)`, **15 epochs**.
- **Stage B** (`configure_stage_b`): unfreeze the deepest **1 stage (OFF, `one_group`)** /
  **2 stages (ON, `two_group`)** and fine-tune with **per-group learning rates**:
  head **1e-3**, deepest stage **1e-4**, second-deepest (ON only) **1e-5**. **≤25 epochs**,
  **early-stop patience 7** on val macro-F1.
- **Model selection = highest validation macro-F1** (canonical `best_model.pth`). Also keeps
  `best_by_loss.pth` (lowest val loss) so no signal is discarded. Checkpoint payload:
  `{state_dict, class_to_idx, config, val_macro_f1, val_loss}`.

**Evaluation** (`experiments/common/evaluate.py`, the single shared evaluator — same
measuring code for every row, which is what makes rows comparable):
- Scores the val-selected checkpoint on **both** the controlled test set and the real-world
  set.
- Reports **accuracy, macro precision/recall/F1, weighted-F1, per-class report**, and
  **confusion matrices** (raw counts + row-normalized PNG).
- Computes the **generalization gap** = controlled − real-world (accuracy and macro-F1).
- Real-world folder labels are **remapped into the training label space by class name**, so a
  different on-disk ordering can never silently scramble metrics.

**Hygiene mechanisms baked into code (not trusted to the operator):**
- Read-once: only rows with a published `eval_results_real_world.json` are ever re-touched.
- Hyperparameter sweeps train **blind** (`--train-only`); a separate `select_on_val.py`
  picks the winner from val macro-F1 (which contains no test numbers), and **only the
  winner** is evaluated on the real-world set.
- **Isolation contract:** every experiment plan lives in its own additive package and
  *imports* shared code read-only — the core ablation, its configs, `run.py`, and
  `compare.py` are never modified by a plan.

---

## 5. Phase 1 — Baseline model comparison (historical, controlled test set)

**Goal:** show the proposed model beats standard/literature approaches under fair, matched
conditions (same epoch budget; each baseline uses its own literature-typical config — i.e. the
**Stack OFF** setting: `advanced_augmentation=false`, `strong_head=false` → plain head
`Dropout(0.2) → Linear`, `cbam=false`, `label_smoothing=0.0`, `stage_b=one_group`). The CNN
backbones are built by `build_backbone(...)` from torchvision `IMAGENET1K_V1` weights
(EfficientNetB0 via `timm.create_model("efficientnet_b0", pretrained=True)`).

**Controlled test-set accuracy (Sunway HPC, 2026-07):**

| Model | Controlled acc |
|---|---|
| **ResNet34 (proposed)** | **96.17 %** (best, macro-F1 0.962) |
| VGG16 | 95.94 % |
| ResNet50 | 95.78 % |
| MobileNetV2 | 94.66 % |
| AlexNet | 90.49 % |
| RandomForest | 87.17 % |
| SVM | 87.11 % |
| KNN | 79.15 % |

Two implementation bugs were found and fixed during first training (worth remembering as a
pattern — *verify layer/block indices against the instantiated model, not the literature
plan*):
- AlexNet head double-pooled an already-pooled tensor.
- EfficientNetB0 assumed 9 block groups; timm's `efficientnet_b0` has 7 — index now derived
  dynamically from `len(model.blocks)`.

> These are **controlled-accuracy** results — a fair-comparison table, not a
> generalization result. The classical models (KNN/SVM/RandomForest) were later **dropped**
> from the CP2 restructure (cited from literature instead).

---

## 6. Phase 2 — The CP2 OFF/ON ablation (12 runs, historical, retired real set)

The study was restructured into a clean **12-run OFF/ON ablation**: **6 CNN backbones**
(resnet34, resnet50, vgg16, alexnet, mobilenetv2, efficientnetb0) × **Stack OFF / Stack ON**.
"Stack" = the proposed solution bundle, toggled by five flags in the config `stack:` block:

| Flag | Stack OFF | Stack ON | What it does |
|---|---|---|---|
| `advanced_augmentation` | `false` | `true` | adds the §3 advanced block |
| `strong_head` | `false` | `true` | OFF: `Dropout(0.2)→Linear`; ON: `BatchNorm1d→Dropout(0.4)→Linear(·,256)→ReLU→Dropout(0.3)→Linear(256,·)` |
| `cbam` | `false` | `true` | ON appends `Sequential_CBAM` to the **deepest two** backbone stages (EfficientNetB0: `blocks[-1]`, `blocks[-2]`) |
| `stage_b` | `one_group` | `two_group` | unfreeze deepest 1 vs 2 stages in Stage B |
| `label_smoothing` | `0.0` | `0.1` | CrossEntropy label smoothing |

The existing per-model folders were
**re-run from scratch** under one seed/engine because the old runs were unseeded and
val-loss-checkpointed (not valid ablation arms).

**Real-world macro-F1, OFF → ON (retired set):**

| Backbone | OFF | ON |
|---|---|---|
| alexnet | 0.1988 | 0.2151 |
| vgg16 | 0.2536 | 0.2460 |
| resnet34 | 0.2659 | 0.2931 |
| resnet50 | 0.2842 | 0.2759 |
| mobilenetv2 | 0.2710 | 0.2571 |
| **efficientnetb0** | 0.2725 | **0.3130** |

**`efficientnetb0_on` is the strongest row in the entire ablation** and became the fixed
baseline for the follow-up intervention studies. Its full profile (retired set): ctrl acc
0.9886, ctrl macro-F1 0.9863, RW acc 0.4111, RW macro-F1 0.3130, gap_F1 0.6733.

---

## 7. Intervention studies on EfficientNetB0-ON

All interventions hold `efficientnetb0_on` fixed and change **one variable**, to answer:
does this specific lever reduce the field gap? Two families were built as isolated additive
packages.

### 7A. Plan 1 — Background randomization (the study's key result)

**Mechanism:** during training, with prob 0.5 the leaf is **segmented and composited onto a
random background**; with prob 0.5 the original passes through — so both are learned, no image
copying. Val/test/real-world stay augmentation-free. A hard rule asserts the background pool
is **disjoint** from the real-world test set (using field images as backgrounds = test-domain
leakage). Implemented additively in `experiments/plan1_bgrand/` (`run_bgrand.py`,
`engine_bgrand.py`, `data_bgrand.py`), importing the shared engine read-only.

**Exact config (`background_randomization:` block):**

| Key | Value | Notes |
|---|---|---|
| `prob` | `0.5` | fraction of train images composited |
| `segmentation` | `pretrained` | rembg / U²-Net leaf matting |
| `mask_erode_px` | `3` | inward erosion to kill backdrop halo |
| `boundary_blur` | `true` | soften composite seam |
| `mask_cache_dir` | `data/mask_cache` | masks keyed by leaf-image content (reused across runs) |
| `background_dir` | `data/backgrounds_generic` (synthetic) **/** `data/backgrounds_generic_real` (**73 real CC0 photos**) | the **one** isolated variable in the real-vs-synthetic contrast |

Everything else (the whole §6 Stack-ON `stack:` block + 15/25-epoch budget) is held identical.

Iterations:
- **v1 (classical HSV masks):** real-world macro-F1 **−0.0193** — but mask-sanity grids showed
  this was an **artifact** (halo of original backdrop bleeding through soft mask edges; edge
  lesions dropped by mask holes). Not a clean test.
- **v2 (rembg / U²-Net masks + inward erosion, disk mask cache):** masks **verified clean**,
  still real-world **−0.0312**. Because masks were good, the artifact hypothesis is ruled out
  ⇒ a **confident bounded negative**: *synthetic-texture* background randomization does not
  reduce the gap (PlantVillage backdrops are already textured; the ON stack is
  augmentation-saturated; procedural textures ≠ real field clutter).
- **Real CC0 backgrounds (73 photos) — the ONLY positive row in the whole study (retired
  set):** RW macro-F1 **0.3182 (+0.0052 vs baseline)** *and* gap narrowed **−0.0164** (the good
  shape — field rose while the gap shrank).

**The comparison that matters — `bgrand` vs `bgrand_real` is itself a clean one-variable
test** (identical technique, prob, segmentation, seed, budget; **only the background source
differs**):

| Background source | RW macro-F1 | vs baseline |
|---|---|---|
| Synthetic textures | 0.2817 | −0.0313 |
| **Real CC0 photos** | **0.3182** | **+0.0052** |
| **Realism effect (real − synthetic)** | | **+0.0365** |

**+0.0365 from background realism alone is larger than any architecture change moved
anything** → *the realism of the data mattered more than any architectural change* — the
thesis conclusion, now backed by a positive result rather than only converging negatives.
Per-class, real backgrounds did something large and specific: Leaf_Mold 0.170→0.351 (+0.181),
Healthy 0.191→0.325 (+0.134). **Honest caveat:** +0.0052 aggregate is single-seed and
indistinguishable from noise at n=1 — which is exactly why the 3-seed replication (§8) was
run.

### 7B. Plan 2 — Architecture tiers (all negative, with two instructive traps)

Each tier = fixed `efficientnetb0_on` + its own one variable (`architecture_mod:` block);
tiers are standalone (never bundled). Built in `experiments/plan2_arch/` (`run_arch.py`,
`engine_arch.py`, `data_res.py`). Exact levers:

| Tier | Config key | Exact value | Sweep sibling (val-only) |
|---|---|---|---|
| 1 drop-path | `drop_path_rate` | `0.2` | `droppath03` = `0.3` |
| 2 resolution | `input_resolution` | `240` (eval `Resize(274) → CenterCrop(240)`, holds the 0.875 crop ratio) | — |
| 3 MixStyle | `mixstyle: {layers, p, alpha}` | `layers=[1,2,3], p=0.5, alpha=0.1` | `mixstyle_l12` = `layers=[1,2]` |

Sweep siblings train blind and the winner is picked on **val** macro-F1 (`select_on_val.py`);
only the winner is read on the real-world set. Results (retired set):

| Row | Ctrl F1 | RW F1 | ΔRW F1 | Δgap F1 |
|---|---|---|---|---|
| `efficientnetb0_on` (baseline) | 0.9863 | 0.3130 | — | — |
| Tier 1 `droppath02` | 0.9848 | 0.2823 | −0.0307 | +0.0293 |
| Tier 2 `res240` (224→240px) | 0.9882 | 0.2717 | −0.0413 | +0.0432 |
| Tier 3 `mixstyle_l123` (MixStyle) | 0.9587 | 0.2847 | −0.0283 | +0.0007 |
| Combo `mixstyle_l123_bgrand` (2×2) | 0.9010 | 0.2653 | −0.0477 | −0.0376 |

**Every architecture tier is negative.** Two traps this dataset demonstrates (both worth
reporting):

1. **Lab-up / field-down (Tier 2, res240):** *raised* controlled accuracy (+0.0019, best-ever
   val 0.9928) while *hurting field the most* (−0.0413). A modification that improves
   PlantVillage but not the field **has not helped** — more pixels just let it lean harder on
   lab-only high-frequency lesion texture.
2. **The gap trap (Combo):** shows the study's **only narrowing gap** (−0.0376) yet the
   **worst** real-world F1 of the five rows (0.2653). The gap shrank only because the lab side
   **collapsed faster** than the field side fell. *Never report a gap improvement without the
   real-world number beside it.*

Additional findings: MixStyle acted as a pure capacity cost (lab and field fell by nearly the
same amount ⇒ gap ~0), consistent with the pre-registered prediction that single-source
MixStyle can only interpolate within PlantVillage's small style variance. A **2×2 factorial**
(baseline / bgrand / mixstyle / both) showed **sub-additivity** — the two harms don't stack
(interaction +0.0118), evidence the two style-perturbation methods are mechanistically
redundant. Known honest limitation stated for the examiner: the MBConv stages hosting MixStyle
are **frozen** under the two-stage transfer protocol, so the claim is "MixStyle *under frozen
early layers* did not help", not "MixStyle does not work" — natural future-work row is
MixStyle + deeper unfreeze.

---

## 8. Data revision + the reportable 3-seed replication

### 8A. Real-world set switch & full retrain

- **Set switch (2026-07-17):** all configs repointed from the retired
  `real_environment_test` to the new `data/real_environment_dataset`. **Every number in §6–§7
  was measured on the retired set** and is not comparable to post-switch numbers. Old numbers
  archived per-run.
  - **The eval trap that bites:** evaluation reads `real_world_dir` **out of the checkpoint**
    (saved at training time), not the YAML — so editing a config and re-running `--eval-only`
    is a silent no-op. A purpose-built `reevaluate_real_world.py` overrides it explicitly, with
    a `--check` counts-only preflight (a partially-downloaded set does *not* fail loudly).
- **Full retrain on a new PlantVillage source** (`data/raw`) via a parallel orchestration
  (`retrain_all.sh` / resume-safe `run_all_local.sh`). **8-class scope:** `Target_Spot` (a
  new-data prediction sink) and `Tomato_mosaic_virus` (a persistent dead class) were turned
  **off**. After the retrain, **everything published is on the new data.**

### 8B. Plan 6 — 3-seed replication (THE reportable set, Colab A100, 2026-07-21)

A **3×3 matrix** — recipe {baseline, synthetic, real} × seed **{42, 43, 44}** — all through one
code path so the **only** variable is the background treatment (baseline = same pipeline with
compositing `prob 0.0`). This answers the examiner's "n=1?" on the realism contrast.
Seed-42 cells **reuse** the existing §7A runs; seeds 43/44 add the new configs in
`experiments/plan1_bgrand/configs/seedrep/` (`bgrand_s43/44.yaml`, `bgrandreal_s43/44.yaml`),
byte-identical to the §7A recipes except `run_name` + `seed`. Paired **by seed** (each contrast
subtracts within the same seed, then averages), and `compare_seeds.py` aggregates.

**Real-world macro-F1 (v3, final, 8-class, real set N=261):**

| Recipe | RW macro-F1 (mean ± std) | Per-seed cells |
|---|---|---|
| Baseline (bg off) | 0.3221 ± 0.0155 | 0.3046 / 0.3274 / 0.3342 |
| Synthetic backgrounds | 0.3495 ± 0.0116 | 0.3404 / 0.3455 / 0.3626 |
| **Real backgrounds** | **0.4641 ± 0.0312** | 0.4282 / 0.4824 / 0.4818 |

**Paired-by-seed contrasts:**
- **Real − baseline: +0.1420 ± 0.0431** (all 3 seeds positive: +.094 / +.155 / +.177)
- **Real − synthetic: +0.1146 ± 0.0249** (all 3 seeds positive: +.088 / +.137 / +.119)
- Synthetic − baseline: +0.0274 (small gain; real ≈ 4× that)

**Why this is strong:**
1. **Fully non-overlapping** distributions — real's worst seed (0.4282) beats synthetic's best
   (0.3626) beats baseline's best (0.3342).
2. Effect ≈ 4× the standard deviation; **unanimous direction** across seeds.
3. Controlled macro-F1 held ~0.93 for every recipe → **the movement is in generalization**,
   not lab fit.
4. **Synthetic being barely above baseline is the key message:** it is the **realism** of the
   backgrounds that transfers, not background variety per se.

This **replicates and supersedes** the earlier single-seed +0.0365. **Report the v3 3-seed
numbers.** (v1 was on the retired PlantVillage source; v2 was a partial/interrupted HPC
retrain — both superseded.)

**Mechanism behind the gain (from confusion analysis):** on the new data the baseline
**collapses to a single default class** on field inputs — `efficientnetb0_on` predicts
`Target_Spot` for ~81 % of *all* real-world images (17.4 % RW accuracy despite 97.8 %
controlled), an extreme domain-shift collapse driven by class imbalance in the new source.
**Real backgrounds break the collapse** (Target_Spot column 81 %→53 %, recovering correct
predictions across Early/Late/Septoria/Mold/Mite/Healthy) — which is *why* real backgrounds
more than double the baseline. `Tomato_mosaic_virus` stays a robust dead class (hence its
exclusion).

---

## 9. Cross-cutting: reproducibility & tooling

- **`compile_results.py`** — regenerates every thesis table from saved JSON only (no GPU, no
  test re-read). Sweep losers appear as `val-only` (evidence hyperparameters were chosen on
  validation, not omitted); reading notes are derived from the data, not hardcoded.
- **`confusion_matrices.py`** — recovers real-world confusion *counts* (the shared evaluator
  draws the PNG but discards the numbers). Not a second test read: frozen checkpoints,
  deterministic inference, and a **hard 1e-6 reproduction check** against the published
  macro-F1. Emits the thesis grid figure (every intervention's error structure side by side)
  and the Target_Spot focus table.
- **Infrastructure evolution:** Sunway HPC (SLURM, L4 GPU, conda `tomato-ml`) → **Kaggle**
  fallback (HPC credits ran out mid-training; phone-gate blocked it) → **Google Colab** (T4/A100)
  running the same resume-safe `run_all_local.sh`. The final clean end-to-end study run was on
  Colab A100.

---

## 10. Headline findings & honest caveats

**Headline:** across the entire study — the 12-run backbone ablation, three architecture tiers,
and a 2×2 factorial — **the only lever that reliably improved real-world generalization was
training on real-background composites.** The 3-seed replication makes it defensible: **real
backgrounds beat the baseline by +0.142 ± 0.043 and synthetic backgrounds by +0.115 ± 0.025 in
real-world macro-F1, unanimously across seeds and with non-overlapping distributions.** *The
realism of the data mattered more than any architectural change.*

**Every architecture/inference-side intervention was a bounded negative:** drop-path, higher
resolution, MixStyle, and the MixStyle+bgrand combo. Two are especially instructive — res240
(lab up / field down) and the combo (gap narrows for the wrong reason).

**Caveats stated honestly (an examiner will look for these):**
- Numbers are **not comparable across dataset revisions** — the retired 10-class set, the new
  8-class set, and the Colab A100 run are distinct. Only §8 is current.
- Two classes (`Target_Spot`, `Tomato_mosaic_virus`) were **excluded** on the new data because
  they were pathological (a prediction sink and a dead class); this is a scope decision that
  raises the reported macro-F1 and must be disclosed.
- The single-seed intervention rows (§7) cannot be distinguished from noise at n=1 — which is
  precisely why the realism contrast was elevated to 3 seeds.

---

## Appendix — Timeline

| Date (2026) | Milestone |
|---|---|
| 07-07/08 | CP2 restructure: unified 12-run OFF/ON ablation runner; re-run all arms under seed 42; classical models dropped. |
| 07-08 | Dataset moved to repo-root `data/processed`; legacy per-model folders cleaned up. |
| 07-12 | Plan 1 v1/v2 (synthetic backgrounds) — bounded negative after mask verification. |
| 07-15/16 | Plan 2 tiers (drop-path, res240, MixStyle) + 2×2 combo — all negative; the two traps documented. |
| 07-16 | **Real-background run — the study's only positive; realism contrast +0.0365 (single seed).** |
| 07-17 | Real-world test set switched to `data/real_environment_dataset`; `reevaluate_real_world.py`. |
| 07-18 | Full retrain on new `data/raw`; **8-class scope** (Target_Spot + mosaic off); confusion analysis (Target_Spot collapse). |
| 07-20 | Infra pivot HPC → Kaggle → Colab. |
| 07-21 | **3-seed replication complete on Colab A100 — the reportable set (real +0.142 / +0.115).** |

---

## Appendix B — Values, functions & methods quick reference

Every value below is taken from the code, not from memory. Files are under `experiments/`.

**Fixed across every run**

| Item | Value | Source |
|---|---|---|
| Input size | `224 × 224` | `engine.py`, `data.py` |
| ImageNet norm | `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` | `data.py` |
| Split | 70/15/15, **seed 42**, symlinks | `split_dataset.py` |
| Optimizer | `torch.optim.Adam` (no scheduler) | `engine.py` |
| Batch size | `32` | configs |
| Stage-A | head-only, `lr=1e-3`, **15 epochs** | `engine.py` |
| Stage-B | per-group LR — head `1e-3`, deepest `1e-4`, 2nd-deepest `1e-5`; **≤25 epochs**, **patience 7** | `engine.py`, `backbones.py` |
| Selection | max **val macro-F1** → `best_model.pth` (+ `best_by_loss.pth`) | `engine.py` |
| Determinism | `seed_everything(seed, deterministic=True)`, `seed_worker` | `seeding.py` |

**Key functions / methods**

| Function | Role |
|---|---|
| `build_loaders(data_dir, image_size, batch_size, advanced_augmentation, seed)` | train/val/test loaders + `class_to_idx` |
| `build_train_transform` / `build_eval_transform` | the §3 augmentation contract |
| `_basic_four()` / `_advanced_block()` | the two augmentation levels |
| `_assert_no_augmentation()` | guard: no aug leaks into val/test |
| `build_backbone(name, num_classes, strong_head, cbam)` → `BuiltModel` | constructs any of the 6 CNNs |
| `BuiltModel.freeze_to_head()` / `.configure_stage_b(two_group)` / `.warm_up()` | the two-stage unfreeze interface |
| `build_head(in, n, strong)` — `plain_head` / `strong_head` | OFF vs ON classifier head |
| `Sequential_CBAM(stage)` | CBAM attention wrapper (ON, deepest 2 stages) |
| `train_run(cfg, results_dir, device)` | the full two-stage loop |
| `evaluate_run(...)` → `_metrics`, `_plot_confusion` | scores both sets, gap, by-name label remap |

**Solution-specific values**

| Solution | Exact values |
|---|---|
| Baseline (Stack OFF) | `advanced_augmentation=false, strong_head=false (Dropout 0.2→Linear), cbam=false, label_smoothing=0.0, stage_b=one_group` |
| Advanced (Stack ON) | `advanced_augmentation=true, strong_head=true (BN→Dropout0.4→Linear256→ReLU→Dropout0.3→Linear), cbam=true, label_smoothing=0.1, stage_b=two_group` |
| Background randomization | `prob=0.5, segmentation=pretrained (U²-Net), mask_erode_px=3, boundary_blur=true`; `background_dir` = synthetic **vs** `backgrounds_generic_real` (73 CC0) — the isolated variable |
| Plan 2 Tier 1 | `drop_path_rate=0.2` (sibling 0.3, val-only) |
| Plan 2 Tier 2 | `input_resolution=240`, eval `Resize(274)→CenterCrop(240)` |
| Plan 2 Tier 3 | `mixstyle: layers=[1,2,3], p=0.5, alpha=0.1` (sibling `[1,2]`, val-only) |
| Seed replication | seeds `{42, 43, 44}`; seed 42 reuses §7A runs; configs in `plan1_bgrand/configs/seedrep/` |
