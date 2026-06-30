# Real-World Generalization Improvement Plan
## Tomato Leaf Disease Detection — ResNet34 Transfer Learning Model
**By Sia Jia Le (22062566) | Supervisor: Dr Chin Teck Min | Sunway University**

---

## The Problem in Plain Terms

Your model was trained almost entirely on **PlantVillage** — a dataset of isolated leaves photographed under controlled studio conditions: uniform white/grey backgrounds, consistent lighting, clean single-leaf framing, and no occlusion. When a farmer takes a photo in a field, none of those conditions exist. The model has learned to classify *PlantVillage-style images*, not *tomato leaf diseases*. This is called **domain shift** or the **lab-to-field generalization gap**, and your own literature review (Mohanty et al., 2016; Fenu & Malloci, 2022) documents exactly this failure mode.

This is not a bug in your code. It is a fundamental limitation of the training data and augmentation strategy.

---

## Root Cause Analysis

Before jumping to solutions, it is important to understand exactly *why* your model fails on real-world images. Each root cause maps to a specific fix.

| Root Cause | What Happens in the Field | What Your Model Learned Instead |
|---|---|---|
| **Background complexity** | Leaves photographed against soil, grass, other plants, fences | Plain white/grey backgrounds only |
| **Lighting variability** | Harsh sunlight, shadows, overcast diffused light, golden hour colour cast | Controlled, flat, uniform studio lighting |
| **Partial occlusion** | Other leaves overlapping, part of leaf cut off at frame edge | Full, centered, isolated leaf always visible |
| **Camera distance and angle** | Photos from above, from the side, close-up or far away | Standardized single viewing angle and distance |
| **Multiple leaves in frame** | Farmer points phone at a plant, captures several leaves at once | Always one leaf per image |
| **Image quality variation** | Motion blur, lens distortion, JPEG compression, low resolution | High-resolution, sharp, clean images |
| **Symptom overlap with non-disease appearance** | Dust, water droplets, physical damage look like disease | Only clean disease symptoms from lab specimens |
| **Symptom at different growth stages** | Early, mid, and late-stage symptoms look very different | Mostly mid-to-late stage, consistent severity |

---

## The Strategy: Five Pillars

The fix is not one single thing. It requires a multi-pronged approach across five areas:

```
Pillar 1: Real-World Data Collection
    → Get actual field images into your training pipeline

Pillar 2: Advanced Augmentation (Domain Simulation)
    → Synthetically simulate field conditions during training

Pillar 3: Mixed Dataset Training Strategy
    → Correct way to combine PlantVillage + real-world data

Pillar 4: Model-Level Improvements
    → Architecture and training changes that help generalization

Pillar 5: Evaluation and Validation on Real Data
    → Correctly measure real-world performance
```

---

## Pillar 1: Real-World Data Collection

### Should You Collect More Real-World Images?

**Yes, but strategically.** Raw quantity is less important than diversity. 50 carefully collected, well-varied real-world images per class are worth more than 500 nearly-identical ones.

### Option A — Public Real-World Datasets (Start Here)

These datasets already exist and can be used immediately, without any field collection:

| Dataset | Source | What It Contains |
|---|---|---|
| **PlantDoc** | GitHub: pratikkayal/PlantDoc-Dataset | 2,569 images across 13 plant species, 17 disease classes, captured from Google Images and field conditions |
| **Mendeley Tomato Field Dataset** | data.mendeley.com | Tomato leaf images from actual fields in diverse conditions |
| **AI Challenger 2018 Plant Disease** | Kaggle mirror available | Real-environment plant images from agricultural competitions |
| **iNaturalist Tomato Disease Subset** | inaturalist.org | Community-contributed field photos with disease labels |
| **Roboflow Plant Disease Universe** | universe.roboflow.com | Multiple real-world plant disease datasets, some tomato-specific |

**Recommended first step:** Download PlantDoc immediately. It is the most commonly used benchmark for real-world plant disease generalization and will allow you to compare your results against published literature.

### Option B — Self-Collected Field Images

If you want to collect your own images for originality (useful for your FYP report), here is a structured protocol:

#### Collection Protocol

**What to capture:**
- Leaves at different stages of disease progression (early, mid, late)
- Leaves at different times of day (8am, 12pm, 4pm) — different lighting angles
- Images from at least 3 different distances: close-up (leaf fills 80% of frame), medium (full plant section), far (multiple plants visible)
- At least 3 different angles: straight-on, 45° from left, 45° from right
- Both sunny and overcast conditions
- Leaves with partial overlaps from other leaves
- Leaves with natural variation: dust, water droplets, insect damage alongside disease

**Target counts (minimum viable):**

| Disease Class | Minimum New Field Images |
|---|---|
| Bacterial Spot | 60 |
| Early Blight | 60 |
| Late Blight | 60 |
| Leaf Mold | 50 |
| Septoria Leaf Spot | 50 |
| Spider Mites | 50 |
| Target Spot | 50 |
| Yellow Leaf Curl Virus | 50 |
| Mosaic Virus | 50 |
| Healthy | 80 |

**What equipment to use:**
- Any modern smartphone camera (iPhone 12+, Samsung Galaxy S20+, or similar)
- No need for a professional camera — the point is to simulate how farmers actually take photos
- Do NOT use a plain background or special lighting — the messier the better for generalization

**Labelling:**
- Label by an expert (agronomist, plant pathologist) or cross-reference with lab-confirmed samples
- Store labels in a simple CSV: `filename, class_label, lighting_condition, background_type, distance`

**Where to collect in Malaysia:**
- Cameron Highlands or Kundasang tomato farms (most accessible for Malaysian students)
- Contact MARDI (Malaysian Agricultural Research and Development Institute) — they maintain experimental plots that students can photograph with permission
- Sunway University's own plant biology labs may have infected leaf specimens that can be photographed

---

## Pillar 2: Advanced Augmentation (Domain Simulation)

This is the most impactful change you can make **right now**, without collecting a single new image. The idea is to apply transformations that force your model to learn disease features rather than background/lighting features.

### What Your Current Augmentation Covers

Your proposal (Section 3.4.2) includes: random rotation, horizontal flip, random cropping/scaling, brightness/contrast adjustment. These are **basic geometric and photometric** augmentations. They are necessary but not sufficient for field generalization.

### What You Need to Add

#### A. Background Replacement / Cutout

PlantVillage's biggest distinguishing feature is its plain background. Teach your model to ignore backgrounds entirely.

**CutOut / Random Erasing:**
```python
# Randomly mask rectangular regions of the image
# Forces model to classify without relying on any single region
transforms.RandomErasing(
    p=0.5,
    scale=(0.02, 0.2),
    ratio=(0.3, 3.3),
    value='random'   # fill with random noise, not black
)
```

**Copy-Paste Background Augmentation (advanced):**
Segment the leaf from PlantVillage images (using GrabCut or SAM) and paste it onto random natural backgrounds (soil images, grass, farm photos). This directly eliminates the background bias.

```python
# Pseudocode for background paste augmentation
def paste_on_random_background(leaf_img, background_img):
    mask = segment_leaf(leaf_img)          # GrabCut or U2Net
    background = load_random_background()  # from a background image pool
    background.paste(leaf_img, mask=mask)
    return background
```

A pool of 50–100 natural background images (soil, grass, farm settings) is sufficient.

#### B. Realistic Lighting Augmentation

```python
# Simulate harsh direct sunlight
transforms.ColorJitter(
    brightness=(0.4, 1.8),   # wider range than your current setting
    contrast=(0.4, 1.6),
    saturation=(0.5, 1.5),
    hue=(-0.1, 0.1)          # slight green/yellow shift from chlorophyll
)

# Simulate shadows — apply random gradient darkening to part of the image
class RandomShadow:
    def __call__(self, img):
        # Draw a random dark polygon over part of the image
        # Simulates a shadow cast by another leaf or structure
        ...
```

#### C. Blur and Noise (Camera Realism)

```python
import albumentations as A

field_augmentations = A.Compose([
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),      # out-of-focus shots
    A.MotionBlur(blur_limit=5, p=0.2),              # shaky hand
    A.GaussNoise(var_limit=(10, 50), p=0.3),        # sensor noise
    A.ISONoise(color_shift=(0.01, 0.05), p=0.2),    # camera ISO noise
    A.ImageCompression(quality_lower=60, p=0.2),    # JPEG compression artifacts
    A.Downscale(scale_min=0.5, scale_max=0.9, p=0.2),  # low resolution capture
])
```

#### D. Perspective and Distortion

```python
A.OneOf([
    A.Perspective(scale=(0.05, 0.15), p=1.0),       # shooting at an angle
    A.GridDistortion(num_steps=5, p=1.0),            # lens barrel distortion
    A.ElasticTransform(alpha=120, p=1.0),            # leaf curvature
], p=0.4)
```

#### E. MixUp and CutMix (Regularization)

These are training-time techniques, not image transforms. They mix two images together during training and have been shown to significantly improve generalization:

```python
# MixUp: blend two images and blend their labels
def mixup(img1, label1, img2, label2, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    mixed_img = lam * img1 + (1 - lam) * img2
    mixed_label = lam * label1 + (1 - lam) * label2
    return mixed_img, mixed_label
```

#### F. Style Transfer Augmentation (LeafNST — already in your literature review)

Your literature review already covers LeafNST by Khare et al. (2024), which transfers disease texture patterns onto healthy leaves to generate new synthetic diseased images. This is worth implementing since you already cited it — it gives you a direct link from literature to implementation.

```python
# Conceptual implementation using neural style transfer
# Transfer the disease "style" from one leaf onto the "content" of another
# Use fast-neural-style (Johnson et al.) for efficiency
```

### Complete Recommended Augmentation Pipeline

```python
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Use albumentations for the full pipeline (more options than torchvision)
train_transform = A.Compose([
    # --- Spatial ---
    A.RandomResizedCrop(224, 224, scale=(0.6, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.Rotate(limit=35, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=20, p=0.4),

    # --- Perspective / Distortion ---
    A.Perspective(scale=(0.05, 0.15), p=0.3),
    A.ElasticTransform(alpha=80, sigma=80 * 0.05, p=0.2),

    # --- Lighting / Color ---
    A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1, p=0.7),
    A.RandomShadow(p=0.3),
    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.15),
    A.RandomRain(p=0.1),          # rain on lens effect

    # --- Camera Realism ---
    A.GaussianBlur(blur_limit=(3, 5), p=0.25),
    A.MotionBlur(blur_limit=5, p=0.15),
    A.GaussNoise(var_limit=(5, 30), p=0.25),
    A.ImageCompression(quality_lower=70, quality_upper=100, p=0.2),

    # --- Occlusion ---
    A.CoarseDropout(
        max_holes=8, max_height=20, max_width=20,
        min_holes=1, fill_value=0, p=0.3
    ),

    # --- Normalization ---
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2(),
])
```

**Key rule:** Apply this only to the training set. Validation and test sets continue to use only resize + center crop + normalize.

---

## Pillar 3: Mixed Dataset Training Strategy

Simply combining PlantVillage and real-world data naively often hurts performance because PlantVillage has ~10,000+ images per class while real-world datasets may have 50–200. The model will continue to overfit to PlantVillage.

### Strategy 1: Weighted Sampling (Recommended First Step)

Do not change the dataset ratio — change how often each sample is seen during training.

```python
from torch.utils.data import WeightedRandomSampler

# Give real-world images a higher sampling weight
# so they are seen proportionally more often per epoch
def build_weighted_sampler(dataset, plantvillage_weight=1.0, realworld_weight=5.0):
    weights = []
    for idx in range(len(dataset)):
        source = dataset.get_source(idx)  # 'plantvillage' or 'realworld'
        if source == 'realworld':
            weights.append(realworld_weight)
        else:
            weights.append(plantvillage_weight)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
```

### Strategy 2: Two-Phase Training (Curriculum Learning)

Train in deliberate phases from clean to noisy data:

```
Phase 1 (Epochs 1–15):  Train on PlantVillage only
                         → Model learns what disease features look like
Phase 2 (Epochs 16–30): Train on PlantVillage + real-world data (50:50 mix)
                         → Model adapts to real-world variation
Phase 3 (Epochs 31–40): Fine-tune on real-world data only
                         → Final adaptation, use low LR (1e-5)
```

This is curriculum learning — start easy (clean data), progress to hard (noisy field data). It consistently outperforms training on the mixed dataset from scratch.

### Strategy 3: Domain Adaptation — Feature Alignment

If you have enough real-world images, you can use **Domain Adversarial Training (DANN)** to explicitly force the feature extractor to learn domain-invariant representations. The model is trained to extract features that a domain classifier *cannot* distinguish as coming from PlantVillage or the field.

This is an advanced technique but highly relevant to your FYP's stated aim of improving generalization. Even a simplified version provides measurable improvement.

```
Standard classification loss:   Classify disease correctly
+
Domain confusion loss:          Confuse domain classifier (source vs. field)
=
Domain-invariant feature space
```

---

## Pillar 4: Model-Level Improvements

### 4.1 Unfreeze More Layers During Fine-Tuning

Your current Stage B fine-tuning only unfreezes `layer4` (last residual block). For real-world generalization, unfreezing `layer3` as well allows the model to re-learn mid-level features (textures, patterns) that are more relevant to field images.

```python
# Stage B extended: unfreeze layer3 and layer4
for name, param in model.named_parameters():
    if 'layer3' in name or 'layer4' in name or 'fc' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
```

Use an even lower learning rate for `layer3` than `layer4` to avoid damaging pretrained features:
- `fc`: LR = 1e-3
- `layer4`: LR = 1e-4
- `layer3`: LR = 1e-5

This is done using a **parameter group** in the optimizer:
```python
optimizer = torch.optim.Adam([
    {'params': model.layer3.parameters(), 'lr': 1e-5},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(),     'lr': 1e-3},
])
```

### 4.2 Add Dropout Before the Classifier Head

PlantVillage-trained models often memorize specific pixel patterns. Dropout after the global average pooling layer forces the model to use redundant, distributed representations — which are more robust to field variation.

```python
import torch.nn as nn
from torchvision import models

model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
model.fc = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(model.fc.in_features, NUM_CLASSES)
)
```

### 4.3 Label Smoothing

When training on mixed data, some real-world labels may be slightly uncertain (boundary cases between two diseases). Label smoothing prevents the model from becoming overconfident, which improves calibration and generalization.

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

### 4.4 Test-Time Augmentation (TTA)

At inference time, instead of predicting on one version of the image, predict on multiple augmented versions and average the probabilities. This is a zero-cost improvement to real-world accuracy.

```python
def predict_with_tta(model, image, n_augments=5):
    tta_transforms = [
        val_transform,                         # original
        apply_horizontal_flip(val_transform),  # flipped
        apply_brightness_boost(val_transform), # brighter
        apply_brightness_drop(val_transform),  # darker
        apply_slight_rotate(val_transform),    # slightly rotated
    ]
    probs = []
    for transform in tta_transforms:
        tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            logit = model(tensor)
        probs.append(F.softmax(logit, dim=1))
    return torch.stack(probs).mean(0)  # average across augmentations
```

TTA alone can recover 3–7% accuracy on real-world test sets without any retraining.

---

## Pillar 5: Evaluation and Validation on Real Data

### The Core Problem with Your Current Evaluation

Your current evaluation uses a test split from the same PlantVillage distribution as your training data. This tells you how well the model memorizes PlantVillage — not how well it detects diseases in real conditions.

### Fix 1: Hold Out a Real-World Test Set Completely

```
Dataset Construction:
├── PlantVillage (train: 70%, val: 15%, test: 15%)     ← existing
└── Real-World Images
    ├── Real-World Validation Set (15%)                ← used during training for domain monitoring
    └── Real-World Test Set (85%) — NEVER touched      ← final evaluation only
```

Report **two separate test set results** in your FYP:
1. PlantVillage test set accuracy (comparison with literature baselines)
2. Real-world test set accuracy (your actual contribution — the improvement over the generalization gap)

### Fix 2: Per-Class Real-World Breakdown

The confusion matrix on real-world data will show different failure modes than on PlantVillage. Diseases that are easy to distinguish in lab conditions (e.g., Late Blight vs. Mosaic Virus) may become confused in the field. Report per-class precision, recall, and F1 for both test sets in a comparative table.

### Fix 3: Domain Gap Metric

Quantify the generalization gap explicitly:
```
Generalization Gap = PlantVillage Test Accuracy − Real-World Test Accuracy
```

Tracking this metric across your model versions (baseline → augmentation improvements → domain adaptation) is a strong result narrative for your FYP.

### Baseline to Compare Against

Your comparison should include:
1. **Baseline**: ResNet34 trained on PlantVillage only, tested on real-world images (your current failing model)
2. **Improved v1**: ResNet34 + advanced augmentation
3. **Improved v2**: ResNet34 + augmentation + mixed training
4. **Improved v3** (if time permits): ResNet34 + augmentation + mixed training + domain adaptation

---

## Implementation Roadmap

Given that you are already in CP2 and have a timeline, here is a prioritized sequence:

### Week 1 (Immediate): Quick Wins

These changes can be made to your existing training pipeline this week with the highest return on investment:

1. **Install albumentations** (`pip install albumentations`)
2. **Replace your current augmentation pipeline** with the advanced pipeline from Pillar 2
3. **Add dropout** before the classifier head (Section 4.2)
4. **Add label smoothing** to the loss function (Section 4.3)
5. **Retrain** your model with these changes on the existing PlantVillage dataset
6. **Implement TTA** for inference (Section 4.4) — no retraining needed

Expected improvement: 15–30% accuracy gain on field images from augmentation alone (consistent with Khare et al., 2024 results cited in your literature review).

### Week 2: Get Real-World Data

1. Download **PlantDoc** dataset (freely available on GitHub)
2. Map its tomato disease classes to your 10 PlantVillage classes
3. Set aside a locked real-world test set (do not touch it until final evaluation)
4. Implement **weighted sampler** (Pillar 3, Strategy 1)
5. Retrain with mixed dataset + weighted sampling

### Week 3: Curriculum Training + Extended Fine-Tuning

1. Implement **two-phase curriculum training** (Pillar 3, Strategy 2)
2. Unfreeze `layer3` as well as `layer4` using parameter groups (Section 4.1)
3. Retrain full pipeline

### Week 4: Evaluate and Document

1. Run final evaluation on **both** PlantVillage test set and real-world test set
2. Generate confusion matrices for both
3. Compute domain gap metric for each model version
4. Write results analysis: what improved, what still fails, why

### Code Change Location Map

| What to Change | File in Your Project |
|---|---|
| Augmentation pipeline | `resnet34_model/src/dataset.py` |
| Model head (add dropout) | `resnet34_model/src/model.py` |
| Loss function (label smoothing) | `resnet34_model/src/train.py` |
| Optimizer (parameter groups) | `resnet34_model/src/train.py` |
| Weighted sampler | `resnet34_model/src/dataset.py` |
| TTA inference | `backend/infrastructure/ml/resnet34_inferencer.py` |
| Real-world test evaluation | `resnet34_model/src/evaluate.py` |
| Dataset preparation script | `resnet34_model/scripts/prepare_dataset.py` |

---

## Common Mistakes to Avoid

**Do not apply augmentation to validation or test sets.** Only the training set is augmented. Validation and test sets use only resize + center crop + normalize, always. Applying augmentation to the test set gives artificially noisy evaluation results.

**Do not mix real-world images into your test set that were also used in training.** This is data leakage. The real-world test set must be locked before any training begins.

**Do not use PlantVillage test accuracy as your primary result.** Your FYP explicitly aims to improve real-world generalization. The real-world test accuracy is your main result. PlantVillage accuracy is the baseline comparison.

**Do not reduce batch size too much when adding augmentation.** Aggressive augmentation increases gradient variance. If batch size drops below 16 with augmentation, training may become unstable. Keep batch size at 32 if GPU memory permits.

**Do not completely replace PlantVillage with real-world data.** PlantVillage provides clean, diverse disease features that real-world data cannot fully replicate in small quantities. The goal is to complement, not replace.

---

## How This Maps to Your FYP Objectives

| FYP Objective | How This Plan Addresses It |
|---|---|
| Objective 4: Optimized augmentation | Pillar 2 directly implements and extends Section 3.4.2 of your methodology |
| Objective 5: Evaluate under field conditions | Pillar 5 implements the real-world evaluation described in Section 3.5.2 |
| Problem Statement: generalization gap | Domain gap metric quantifies and reports the improvement you achieve |
| Abstract: improve generalization to real-world field conditions | The entire plan is the practical implementation of your stated research aim |

Your FYP proposal already identifies all of these problems (Section 1.2 cites Mohanty et al., Fenu & Malloci, Ahmad et al. on exactly this issue). This plan gives you the concrete implementation to close the gap you identified in your own literature review.

---

## Summary Table

| Pillar | Effort | Impact | When |
|---|---|---|---|
| Advanced augmentation (albumentations) | Low | High | Week 1 |
| Dropout + label smoothing | Low | Medium | Week 1 |
| Test-Time Augmentation (TTA) | Low | Medium | Week 1 (no retraining) |
| Download PlantDoc + weighted sampler | Medium | High | Week 2 |
| Unfreeze layer3 + parameter groups | Low | Medium | Week 3 |
| Curriculum training (2-phase) | Medium | High | Week 3 |
| Background replacement augmentation | High | High | If time permits |
| Domain adversarial training (DANN) | High | Very High | If time permits |
| Self-collected Malaysian field images | High | Very High | If time permits |
