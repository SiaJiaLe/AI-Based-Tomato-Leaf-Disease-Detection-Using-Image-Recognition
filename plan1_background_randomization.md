# Plan 1 — Background-Randomization Augmentation (EfficientNetB0)

**Owner:** Sia Jia Le (22062566)
**Depends on:** the unified 12-run ablation (frozen split `data/processed`, seed 42, budget stage_a 15 / stage_b 25 / patience 7).
**Goal:** Test whether replacing PlantVillage's uniform backgrounds with varied natural backgrounds during training reduces the real-world generalization gap. This is a **train-set-only augmentation**, added as one new row; it does not modify architecture.

> **Why this first.** Your ~57-point gap is dominated by a background/lighting shortcut: PlantVillage leaves sit on clean, uniform backgrounds, so the network learns "clean background → confident prediction" and collapses on cluttered field images. Background randomization attacks that shortcut directly — higher expected payoff on the real-world column than any attention block. It also keeps the "solutions" framing intact (it's an augmentation, not an architecture claim).

---

## 1. The single claim this tests

*Does compositing segmented PlantVillage leaves onto varied backgrounds during training improve real-world macro-F1 (and shrink the gap) for EfficientNetB0, relative to the same model with the existing Stack-ON augmentation only?*

**Judged on:** real-world macro-F1 and generalization gap. **Not** PlantVillage lab accuracy — a change that lifts lab but not real-world has not helped your problem.

---

## 2. Hygiene rules (identical to the main ablation)

1. **Train-set only.** Background randomization touches the training transform chain *only*. Val and test get resize + ImageNet normalization, nothing else. (Same guardrail as the four basic augmentations and the advanced pipeline.)
2. **One variable.** The new row differs from `efficientnetb0_stack_on` by **exactly one thing**: the background-randomization transform added to the train pipeline. Same backbone, split, seed (42), budget, head, CBAM, label smoothing, basic + advanced augmentation.
3. **Frozen split, seed 42, same budget/engine.** Reuse `data/processed`, `common/seeding.py`, `common/engine.py` unchanged.
4. **Select on PlantVillage val macro-F1; read real-world once** at the end for this row.
5. **No real-world images enter training.** Backgrounds composited in are *generic natural textures*, NOT images from your real-world test source. (Critical — see §3.2. Using field images as backgrounds would leak the test domain into training and break condition 1.)

---

## 3. Method

### 3.1 The core idea
For a training image: (a) segment the leaf from its uniform PlantVillage background, (b) paste the leaf onto a randomly chosen natural background, (c) feed the composite through the rest of the existing train transforms. Apply stochastically (only to a fraction of images per epoch) so the model still sees some clean images too.

### 3.2 Background source — must be domain-neutral
- Use a generic texture/scene set that is **not** your real-world tomato test data. Good options: a small curated folder of soil, foliage, grass, wood, hands, sky, greenhouse textures (e.g. from a free texture dataset or a handful of CC0 images). 30–100 background images is plenty.
- [ ] **Assert** none of these backgrounds come from the real-world test source. Log the background directory path in the run config.
- Rationale: you're teaching background-invariance in general, not memorizing the specific field the test set came from. The latter would be same-source leakage.

### 3.3 Leaf segmentation (PlantVillage is easy to segment)
PlantVillage backgrounds are near-uniform, so segmentation does **not** need a neural model. Two viable routes:

**Route A — classical mask (fast, recommended first):**
- Convert to HSV; threshold on the green/leaf range; morphological close+open to clean the mask; take the largest connected component as the leaf.
- Works because the background is uniform and the leaf is the dominant non-background object.
- [ ] Sanity-check masks on ~20 images per class before running full training — bad masks poison training.

**Route B — pretrained segmentation (fallback if Route A masks are noisy):**
- Use a lightweight pretrained segmentation model (e.g. a U²-Net / rembg-style background remover) to get the leaf mask.
- Heavier, but more robust to leaves that share color with a background.

### 3.4 Compositing
- Resize/scale the leaf to a random scale, random position on the background canvas, then continue into the existing transform chain (geometric → photometric → resize-224 → normalize).
- Optional realism touches (keep mild): slight Gaussian blur at the leaf boundary to avoid a hard "cut-out" edge; random background brightness so lighting varies.
- **Preserve the lesion.** Do not scale so small or crop so hard that the diseased region is lost — the label must still be justified by visible symptoms.

### 3.5 Application probability (this is the key hyperparameter)
- Apply background replacement to a **fraction `p`** of training images each epoch; leave `1−p` as original (clean-background) images.
- Reason: you want the model robust to *both* clean and cluttered backgrounds, not to overfit to composites. Start `p = 0.5`.
- `p` is the one thing you may tune — but tune it **on PlantVillage validation macro-F1**, never on real-world. Candidate values: `p ∈ {0.3, 0.5, 0.7}`. Pick the best on val, then read real-world once.

---

## 4. Config

Add to the EfficientNetB0 Stack-ON config a new block; everything else identical:

```yaml
run_name: efficientnetb0_stack_on_bgrand
seed: 42
backbone: efficientnetb0
split_file: data/processed              # unchanged, frozen
realworld_dir: data/processed/real_environment_test
solution_stack:                          # unchanged from stack_on
  advanced_augmentation: true
  label_smoothing: 0.1
  strong_head_bn: true
  cbam: true
  stage_b_unfreeze: [last_two_mbconv]
background_randomization:                # the ONE new thing
  enabled: true
  prob: 0.5                              # p; tune on val only
  background_dir: data/backgrounds_generic   # NOT real-world test data
  segmentation: hsv_threshold            # hsv_threshold | pretrained
  boundary_blur: true
training:                                # unchanged budget
  stage_a_epochs: 15
  stage_b_epochs: 25
  patience: 7
  stage_a_lr: 1.0e-3
  stage_b_lr: 1.0e-4
  batch_size: 32
eval:
  plantvillage_test: true
  realworld_test: true                   # read ONCE at end
```

- [ ] Runner asserts `background_randomization` transforms are absent from val/test loaders.
- [ ] Runner asserts `background_dir` is disjoint from the real-world test source.

---

## 5. Steps

1. [ ] Build/curate `data/backgrounds_generic` (30–100 domain-neutral images). Log the path; assert disjoint from real-world source.
2. [ ] Implement the segmentation+composite transform (Route A first). Unit-test on ~20 images/class; eyeball the masks and composites.
3. [ ] Wire it into the **train-only** transform chain, gated by `prob`.
4. [ ] Run `p = 0.5` first. Evaluate on PlantVillage val (macro-F1) + PlantVillage test + real-world test.
5. [ ] If promising, sweep `p ∈ {0.3, 0.7}`, **selecting on val**. Read real-world once for the chosen `p`.
6. [ ] Emit the same metric set as every other row: Acc, MacroF1, RealWorldAcc, RealWorldF1, Gap, per-class real-world, confusion matrix.

---

## 6. Interpreting the result

- **Real-world macro-F1 ↑ and gap ↓** → background shortcut confirmed; this becomes part of your best augmentation config and the new baseline for Plan 2. Strong, clean contribution.
- **Real-world flat, lab flat** → the shortcut wasn't background (or the composites were unrealistic). Check mask/composite quality before concluding; then report as a tried-and-bounded negative.
- **Lab ↓ but real-world ↑** → *ideal* outcome to discuss: the model gave up a lab-specific shortcut for genuine robustness. This is exactly the domain-shift story your literature predicts.

**Deliverable:** one new row (`efficientnetb0_stack_on_bgrand`) in the master table + its per-class real-world confusion matrix, compared against `efficientnetb0_stack_on`. The delta is attributable to background randomization alone.

---

## 7. Report placement
- Method → extend **3.4.2 (data augmentation)** as an advanced, domain-targeted augmentation; cite the background-bias / domain-shift evidence (Fenu & Malloci 2022) as motivation.
- Result → Results chapter, as an augmentation-ablation row on EfficientNetB0.
- Interpretation → Discussion, tied to "the field gap is partly a background-shortcut problem."
