# CP2 Implementation Plan — Solution-Stack Generalization Ablation

**Owner:** Sia Jia Le (22062566)
**Project:** AI-Based Tomato Leaf Disease Detection (ResNet34, transfer learning)
**Purpose of this document:** A precise, agent-executable plan to restructure the current experiment so that it produces *clean, attributable evidence* about whether the proposed solution stack improves generalization to real-world tomato leaf images.

> **Read this first — the point of the whole exercise.**
> This is an *experiment*, not a mission to make ResNet34 the top scorer. The deliverable is a controlled measurement of how much the proposed solutions help real-world generalization, and where they stop helping. Both "the solutions help" and "the solutions help but a large field gap persists" are valid, publishable outcomes. **Do not tune, select, or reframe anything to chase a better real-world number.** The data decides the story.

---

## 1. The research question being tested

**Primary claim:** *Do the proposed solutions (advanced augmentation, CBAM attention, label smoothing, stronger head, deeper Stage-B fine-tuning) improve a CNN's generalization from PlantVillage (controlled) to a source-independent real-world tomato leaf test set?*

**Secondary claim (breadth):** *Is that effect architecture-agnostic — i.e. does it hold on more than one CNN backbone?*

**Explanatory goal:** *Where the gap persists, explain why* (per-class failure analysis), connecting back to the proposal's secondary motivation about visually-similar diseases (bacterial spot / early blight / leaf mold).

Everything below exists to answer these three things without confounds.

---

## 2. Non-negotiable hygiene rules (do not violate)

These are the rules that keep the generalization claim valid. A violation of any one silently invalidates the result.

1. **Condition 1 only — no real-world images in training.** Training and validation use **PlantVillage only**. The real-world dataset is a **held-out test set**, used for final measurement only.
2. **Source-independent test set.** The real-world test images must share no source, capture session, or duplicates with any training image. (Confirmed already — keep it that way; do not add PlantDoc or any field images into `train/` or `val/`.)
3. **Augmentation is training-set only.** Validation and test sets receive **only** resize + ImageNet normalization. Never augment val/test.
4. **Select on PlantVillage validation; touch real-world test ONCE.** All model selection, checkpoint choice, and hyperparameter decisions use the **PlantVillage validation macro-F1**. The real-world test set is evaluated exactly once per final model, at the very end. No peeking, no iterating against it.
5. **Change one thing at a time.** Between a "control" run and its "treatment" run, the *only* difference is the solution stack. Same data, same splits, same seed, same epoch/patience budget.
6. **Frozen, identical splits across all runs.** Every condition uses the identical train/val/test split (same files in same sets). Persist the split to disk (e.g. a CSV/JSON of file→split) and load it everywhere.
7. **No data leakage in splits.** Split at image level with de-duplication so no near-duplicate appears in two subsets.

---

## 3. Pre-work — codebase cleanup (do this before any training)

### 3.1 Remove the dead weighted sampler
The `WeightedRandomSampler` for "real-world image oversampling" is **incoherent in condition 1** (there are no real-world images in training to oversample) and is currently **not even wired into the DataLoader** (`shuffle=True` is in use). Leaving it in the code contradicts the project's central claim.

- [ ] Delete the sampler construction code and any "domain weight" logic from the training pipeline.
- [ ] Confirm the `DataLoader` uses `shuffle=True` for training (and `shuffle=False` for val/test).
- [ ] Grep the codebase for `WeightedRandomSampler`, `domain_weight`, `oversample`, `PlantDoc` and remove/neutralize dead references.

### 3.2 Reconcile the report with what actually ran
- [ ] In the CP2 methodology, **remove or explicitly mark as "considered, not applied"** these two items from the extensions list: *"Weighted sampler for real-world image oversampling"* and *"Real-world training supplement (PlantDoc)."* They must not appear as part of the training method, because they were not used. Code and report must tell the same story.

### 3.3 Establish reproducibility scaffolding
- [ ] Set and log a global seed (Python, NumPy, PyTorch, CUDA) for every run.
- [ ] Persist the train/val/test split file once; load it in every experiment.
- [ ] Ensure every run writes a config snapshot + metrics to a uniquely named results directory.

**Acceptance criteria for Section 3:** No sampler/PlantDoc code paths remain; splits are frozen to disk; a single seed governs all runs; report no longer describes unused techniques.

---

## 4. The experiment matrix

Legend: **Stack OFF** = two-stage transfer learning + basic augmentation only. **Stack ON** = OFF plus the full solution stack (Section 5). **Every one of the six CNN architectures is run in both conditions**, so the stack's effect is tested on each backbone under identical treatment.

| # | Model | Treatment | Status | Role |
|---|-------|-----------|--------|------|
| 1 | ResNet34 | Stack OFF | **NEW — build** | Control (primary pair) |
| 2 | ResNet34 | Stack ON | Exists (proposed model) | Treatment (primary pair) |
| 3 | VGG16 | Stack OFF | Exists | Control |
| 4 | VGG16 | Stack ON | **NEW — build** | Treatment |
| 5 | ResNet50 | Stack OFF | Exists | Control |
| 6 | ResNet50 | Stack ON | **NEW — build** | Treatment |
| 7 | AlexNet | Stack OFF | Exists | Control |
| 8 | AlexNet | Stack ON | **NEW — build** | Treatment |
| 9 | MobileNetV2 | Stack OFF | Exists | Control |
| 10 | MobileNetV2 | Stack ON | **NEW — build** | Treatment |
| 11 | EfficientNetB0 | Stack OFF | Exists | Control |
| 12 | EfficientNetB0 | Stack ON | **NEW — build** | Treatment |

**Ablation = all six OFF/ON pairs.** The comparisons that carry the thesis:
- **1 vs 2** → does the stack help ResNet34? (primary claim)
- **3-vs-4, 5-vs-6, 7-vs-8, 9-vs-10, 11-vs-12** → does the effect hold across architecture families? (architecture-agnostic breadth claim)

If the stack lifts real-world macro-F1 on all six backbones, the breadth claim is strong: the effect is a property of the solutions, not an artifact of one architecture. If it helps on some and not others, that split is itself an informative finding — report it, don't hide it.

**This also makes the ResNet50 comparison fair.** Previously "plain ResNet50 beat loaded ResNet34" was confounded (architecture *and* stack differed). Now you can compare **ResNet50 Stack ON vs ResNet34 Stack ON** (same treatment → clean architecture comparison) *and* read each model's own OFF→ON delta (clean solution comparison). Report both; don't conflate them.

**Two per-architecture cautions for the write-up (not blockers):**
- **EfficientNetB0** already contains squeeze-and-excitation (channel attention) inside its MBConv blocks. Adding CBAM stacks a second attention mechanism on top. This is defensible but *must be justified* — and its OFF→ON delta is exactly the evidence of whether the extra attention helped or merely added parameters. Be ready for the examiner question.
- **MobileNetV2** uses inverted-residual blocks; CBAM insertion points and the "last two block groups" unfreeze must be mapped to that structure specifically (Section 5). Don't copy ResNet insertion logic.

**Removed:** the KNN / SVM / RandomForest classical baselines are dropped. They cannot receive the solution stack (nothing to turn on/off), they sit on frozen ResNet34 features rather than being independent, and they answer the old "classical vs deep" question rather than your "do my solutions help generalization" question. Instead of re-running them, **cite the established finding** (Khan et al. 2021; Tan et al. 2021) that CNNs outperform classical ML on this task, in one sentence, so their absence reads as deliberate scope rather than an oversight.

---

## 5. Precise definition of the "Full Solution Stack"

Enumerate these so "Stack ON" is unambiguous and reproducible. Mark which are architecture-agnostic vs architecture-specific.

| Component | Type | Notes for implementation |
|-----------|------|--------------------------|
| Advanced albumentations pipeline (shadow, fog, blur, noise, elastic distortion, perspective) | Architecture-agnostic | Train set only, **layered on top of the four basic augmentations** (does not replace them). Keep strength moderate enough to preserve lesion features. **Identical pipeline across all six architectures.** |
| Label smoothing on cross-entropy | Architecture-agnostic | **Same smoothing factor across all models.** |
| Strong two-layer classifier head with `BatchNorm1d` | Architecture-agnostic (head replaces final classifier) | Same head design pattern, adapted to each backbone's feature dimension (ResNet34: 512; ResNet50: 2048; VGG16: 4096 after its classifier; AlexNet: 4096; MobileNetV2: 1280; EfficientNetB0: 1280). |
| Deeper Stage-B fine-tuning (unfreeze last **two** block groups) | Architecture-specific | See per-architecture mapping table below. |
| CBAM attention blocks | Architecture-specific | See per-architecture mapping table below. Insertion points differ by design — adapt, do not copy blindly. |

### Per-architecture mapping for CBAM insertion and Stage-B unfreeze

| Architecture | Block structure | Stage-B unfreeze (last two groups) | CBAM insertion points | Notes |
|--------------|-----------------|------------------------------------|-----------------------|-------|
| ResNet34 | residual `layer1–4` | `layer3` + `layer4` | after `layer3` and `layer4` | Clean, canonical. |
| ResNet50 | residual `layer1–4` (bottleneck) | `layer3` + `layer4` | after `layer3` and `layer4` | Same as ResNet34; channel dims differ. |
| VGG16 | conv `block1–5` (no residuals) | `block5` + `block4` | after `block5` and `block4` conv stacks | No skip connections; straightforward manual insertion. |
| AlexNet | 5 sequential conv layers | last two conv layers | after the last two conv layers | Shallow; insertion trivial. |
| MobileNetV2 | inverted-residual bottleneck blocks | **last two bottleneck groups** | after the **last two inverted-residual stages** (respect the residual add — insert on the block output, not mid-shortcut) | Do **not** reuse ResNet insertion logic; map to the inverted-residual layout explicitly. |
| EfficientNetB0 | MBConv blocks (**contain SE attention**) | last two MBConv stages | after the last two MBConv stages | Adding CBAM on top of built-in squeeze-excite — justify in report; the OFF→ON delta is the evidence it helped or not. |

**Base configuration shared by BOTH Stack OFF and Stack ON** (i.e. the two-stage foundation, never removed):
- Two-stage training: Stage A trains new head with backbone frozen; Stage B unfreezes per above and fine-tunes.
- Stage-dependent LR: Stage A higher (1e-3…5e-4); Stage B lower (1e-4…1e-5).
- Validation-based checkpointing on **macro-F1** (save best, not last).
- **Basic augmentation (the four CP1 techniques) — applied in BOTH Stack OFF and Stack ON.** See the definition table below. In OFF these are the *only* augmentations; in ON the advanced albumentations pipeline layers *on top* of them (it does not replace them).
- Preprocessing: resize 224×224, normalize ImageNet mean `[0.485,0.456,0.406]` / std `[0.229,0.224,0.225]`.
- Aligned budget across ALL runs: same max epochs (Stage A + Stage B) and same early-stopping patience, so epoch count is never a confound.

### Definition of "basic augmentation" (the four CP1 techniques)

These are the four augmentations specified in CP1 Section 3.4.2. They form the shared augmentation baseline for every run. **Train set only — validation and test sets receive resize + ImageNet normalization and nothing else.** Keep transform *strengths* identical across all 12 runs so the only augmentation difference between OFF and ON is the advanced pipeline.

| Technique | Purpose (from CP1) | Implementation guidance |
|-----------|--------------------|-------------------------|
| **Random rotation** | Simulate different camera angles / leaf orientations | Moderate range only (e.g. ±15–30°) to preserve natural leaf structure; avoid extreme angles that distort lesion geometry. |
| **Horizontal flipping** | Double visual variation; valid since diseases have no fixed left–right orientation | Random horizontal flip, ~0.5 probability. Do **not** add vertical flip unless justified — it can create unnatural leaf orientations. |
| **Random cropping and scaling** | Simulate varying camera distance / framing, partial visibility | Random resized crop that still yields the 224×224 input; keep the scale range moderate so the leaf and lesions remain identifiable (avoid cropping away the diseased region entirely). |
| **Brightness and contrast adjustment** | Simulate real-world lighting variation (sun, shade, indoor/outdoor) | Mild jitter on brightness and contrast so the model relies on relative texture/pattern rather than absolute intensity; avoid extremes that wash out symptoms. |

**Ordering note:** apply geometric transforms (rotation, flip, crop/scale) and photometric transforms (brightness/contrast) before the final resize-to-224 and normalization. In Stack ON, the advanced albumentations pipeline (Section 5) is inserted into this same train-only transform chain, on top of these four — never applied to val/test.

**Guardrail:** the augmentation-on-val assertion (Sections 3.3, 6.1, 7) covers *both* the basic four and the advanced pipeline. Neither may ever touch validation or test dataloaders.

> **Important for interpretation:** because the stack bundles several components, a positive OFF→ON delta on any architecture shows *the stack as a whole* helps that architecture — not which component. That is acceptable for the primary and breadth claims. A component-wise ablation (Section 9) is the way to attribute the effect to individual pieces, if compute allows. EfficientNetB0's delta additionally answers "did CBAM-on-top-of-SE help?" — call that out explicitly in the write-up.

---

## 6. Implementation steps

### 6.1 Config-driven experiment runner
Refactor training so each matrix row is a **single config file** (YAML/JSON). One config = one run = one results directory. Suggested fields:

```yaml
run_name: resnet34_stack_off
seed: 42
backbone: resnet34            # resnet34 | vgg16 | ...
split_file: splits/frozen_split.json
solution_stack:
  advanced_augmentation: false   # true for Stack ON
  label_smoothing: 0.0           # e.g. 0.1 for Stack ON
  strong_head_bn: false          # true for Stack ON
  cbam: false                    # true for Stack ON
  stage_b_unfreeze: ["layer4"]   # Stack ON: ["layer3","layer4"] (resnet) / last two conv blocks (vgg)
training:
  stage_a_epochs: 15
  stage_b_epochs: 25
  patience: 7
  stage_a_lr: 1.0e-3
  stage_b_lr: 1.0e-4
  batch_size: 32
eval:
  plantvillage_test: true
  realworld_test: true           # evaluated ONCE at the end
  tta: false                     # optional; see Section 9
```

- [ ] The runner must fail loudly if augmentation is accidentally enabled on val/test.
- [ ] The runner must log the resolved config into the results directory.

### 6.2 Build the NEW runs (the missing arms)
The matrix needs both OFF and ON for all six CNNs. Typically the six **Stack OFF** runs already exist (your current baselines) plus **ResNet34 Stack ON** (your proposed model). That leaves the NEW arms to build:

- [ ] **ResNet34 Stack OFF** (run 1) — the control for your primary claim. Identical treatment to the plain baselines: two-stage, basic aug, `layer4`-only unfreeze, plain single-layer head, no CBAM, no label smoothing.
- [ ] **Stack ON for VGG16, ResNet50, AlexNet, MobileNetV2, EfficientNetB0** (runs 4, 6, 8, 10, 12) — port each stack component using the per-architecture mapping table in Section 5. Sequence: ResNet50 → VGG16 → AlexNet (unambiguous insertion) first, then MobileNetV2 and EfficientNetB0 (mind the inverted-residual mapping and the SE-plus-CBAM justification).
- [ ] **Verify every existing arm** (the six Stack OFF runs and ResNet34 Stack ON) used the **same frozen split, seed policy, and epoch/patience budget**. Any arm that did not must be **re-run** under the unified config. An ablation across mismatched splits is not an ablation — this applies to all 12 runs, not just the new ones.
- [ ] Confirm each OFF/ON pair differs by **nothing except the stack** (same split, seed, budget, backbone). This is what makes each pair's delta attributable.

### 6.3 Evaluation
For every model, compute on **both** the PlantVillage test set and the real-world test set:
- Accuracy, macro-Precision, macro-Recall, macro-F1.
- **Generalization gap** = PlantVillage-test accuracy − real-world-test accuracy (and the same for macro-F1).
- **Per-class metrics** on the real-world test set (precision/recall/F1 per disease class).
- **Confusion matrix** on the real-world test set (raw counts + row-normalized).

- [ ] Extend/reuse `compare_models.py` to append the new runs and emit a single comparison table (the format you already have: Acc, MacroF1, RealWorldAcc, RealWorldF1, Gap).

### 6.4 Per-class failure analysis (the explanatory contribution)
- [ ] For each Stack ON model, generate the row-normalized real-world confusion matrix as a heatmap figure.
- [ ] Identify the classes with the largest real-world recall collapse.
- [ ] Check specifically whether confusions concentrate among **visually-similar diseases** (bacterial spot ↔ early blight ↔ leaf mold ↔ septoria). This directly tests the proposal's secondary motivation.
- [ ] Compare the Stack OFF vs Stack ON confusion matrices for the same architecture — does the stack fix specific class confusions, or shift them?

**Acceptance criteria for Section 6:** four ablation runs share one frozen split/seed/budget; each emits both-dataset metrics + gap + per-class + confusion matrix; comparison table regenerated.

---

## 7. Reproducibility & anti-leakage checklist

- [ ] Single seed logged per run; deterministic where feasible.
- [ ] One frozen split file loaded by every run (verify by hashing the split assignment).
- [ ] De-duplication check across subsets (no near-duplicate images spanning train/val/test).
- [ ] Assertion in code: augmentation transforms are absent from val/test dataloaders.
- [ ] Assertion in code: no file under `train/` or `val/` originates from the real-world source.
- [ ] Real-world test evaluated once per final model; no hyperparameter decision references real-world metrics.

---

## 8. Deliverables (what feeds the CP2 report)

1. **Master comparison table** — all 12 CNN runs, both datasets, gap column. (Extends your existing table.)
2. **Ablation table** — the six OFF/ON pairs, each showing Δ(real-world accuracy) and Δ(real-world macro-F1) attributable to the stack. This is the centrepiece: one row per architecture, delta per architecture, and whether the effect is consistent across all six.
3. **Same-treatment architecture comparison** — the Stack-ON models ranked against each other (this is the *fair* ResNet34-vs-ResNet50 comparison that replaces the old confounded one).
4. **Per-class real-world tables** — for each Stack ON model.
5. **Confusion-matrix heatmaps** — real-world, Stack OFF vs Stack ON, per architecture (at least for ResNet34; ideally the two best backbones).
6. **Gap-narrowing figure** — grouped bar chart of real-world macro-F1, OFF vs ON, for all six architectures side by side. This single figure visually carries the breadth claim.
7. **Written findings** — the honest result (does the stack help universally, partially, or not) + the mechanistic explanation of the residual gap + one sentence citing Khan/Tan for the omitted classical baselines.

### Mapping to report sections
- Ablation design & rationale → **Methodology 3.5** (evaluation & comparison).
- Solution-stack component definitions → **3.2.2 (head/CBAM), 3.2.3 (Stage-B depth), 3.2.4.1 (label smoothing), 3.4.2 (augmentation)**.
- Results tables & confusion matrices → **Results chapter**.
- "Solutions help but a field gap persists, and here's why" → **Discussion/Limitations**, tied back to Mohanty (2016), Fenu & Malloci (2022), Tang (2025), Ahmad (2023).

---

## 9. Optional / stretch (only if compute allows — clearly separable from core)

Ordered by value-for-effort. None of these are required for a valid result; each is an enhancement.

1. **Background-randomization augmentation.** Highest-leverage single addition for closing the field gap: segment the leaf from PlantVillage's uniform background and composite onto varied natural backgrounds. If added, treat it as an *extra* stack component and ablate it too (otherwise you can't attribute its effect). Train-set only.
2. **Fine-tuning-depth ablation** on ResNet34: head-only vs `layer4` vs `layer3+layer4`, selected on PlantVillage val, real-world read once. May reveal that deeper fine-tuning *raised* lab accuracy but *hurt* generalization — a strong CP2 finding in itself.
3. **Test-time augmentation (TTA)** at inference: average predictions over flips/crops of each real-world test image. Inference-only, condition-1 legal, ~1–3 pts. Report as a separate row so it's not conflated with the training-side stack.
4. **Component-wise ablation** of the stack (turn on one component at a time) to attribute the effect to individual pieces. Costly (many runs) but the strongest possible version of the contribution.
5. **Ensemble (ResNet34 + ResNet50)** — fulfils the proposal's problem-statement promise of "combining multiple architectures," and ensembles reliably help out-of-distribution. Report as a separate contribution, not as part of the single-model ablation.

---

## 10. Execution order (suggested)

1. Section 3 cleanup (sampler removal, report reconciliation, seed/split scaffolding).
2. Build config runner (6.1) with the augmentation-on-val guardrail.
3. Verify all existing arms (six Stack OFF + ResNet34 Stack ON) share one frozen split/seed/budget; re-run any that don't.
4. Build **ResNet34 Stack OFF** (the primary control) and confirm the 1-vs-2 pair is clean — this alone fixes the biggest current weakness, so get it working end-to-end first.
5. Build the remaining Stack ON runs in porting-risk order: **ResNet50 → VGG16 → AlexNet**, then **MobileNetV2 → EfficientNetB0**.
6. Evaluation + per-class + confusion matrices for every run (6.3, 6.4).
7. Regenerate master comparison + six-pair ablation tables + gap-narrowing figure (Section 8).
7. (Optional) Section 9 items, each as clearly separated experiments.
8. Write findings honestly — including the persistent gap and its mechanism.

---

## 11. Guardrails the agent must respect (summary)

- Never put real-world images in train/val.
- Never augment val/test.
- Never select or tune against the real-world test set.
- Between control and treatment, change only the solution stack.
- Keep splits/seed/budget identical across all four ablation arms.
- Remove dead sampler/PlantDoc code; keep report consistent with code.
- Report the result the data gives — do not engineer a preferred narrative.
