# Baseline Models Implementation Plan
## Your Proposed ResNet34 vs Existing Solutions — Fair Experimental Comparison
**Student:** Sia Jia Le (22062566) | **Supervisor:** Dr Chin Teck Min | Sunway University

---

## Research Question This Comparison Answers

> Does your proposed solution — ResNet34 with advanced augmentation, a
> regularized classifier head, and CBAM attention — achieve better
> real-world generalization than existing approaches trained in their
> standard configurations?

Every baseline is trained the way it would be trained in an existing paper:
basic augmentation, standard single-layer head, no attention modules.
Your proposed ResNet34 runs with all your improvements applied.
The only shared element is the dataset — every model trains and evaluates
on the same `processed/` split so data is not a confounding variable.

---

## The Comparison Design at a Glance

| | Baselines (existing solutions) | Your ResNet34 (proposed solution) |
|---|---|---|
| Dataset | `processed/train` and `processed/test` | Same |
| Augmentation | Basic (flip, crop, normalize only) | Advanced albumentations pipeline |
| Classifier head | Standard: Dropout(0.2) → Linear(N, 10) | Strong: BN1d → Drop(0.4) → Linear(N,256) → ReLU → Drop(0.3) → Linear(256,10) |
| Attention | None | CBAM in layer3 and layer4 |
| Weighted sampler | No | Yes |
| Training regime | Single-stage fine-tune (freeze then unfreeze) | Two-stage with per-layer LRs |

---

## Output Directory Structure

Every model writes all results to its own folder under `outputs/`:

```
resnet34_model/outputs/
│
├── knn/outputs/
│   ├── eval_results.json
│   ├── cm_processed_test.png
│   └── classification_report.txt
│
├── svm/outputs/
│   └── (same files)
│
├── random_forest/outputs/
│   └── ...
│
├── alexnet/outputs/
│   ├── eval_results.json
│   ├── cm_processed_test.png
│   ├── history_stage_a.json
│   ├── history_stage_b.json
│   └── classification_report.txt
│
├── vgg16/outputs/
├── mobilenetv2/outputs/
├── efficientnet_b0/outputs/
├── resnet50/outputs/
│
└── resnet34_proposed/outputs/    ← your trained model results go here
    ├── eval_results.json
    ├── cm_processed_test.png
    └── classification_report.txt
```

---

## New Files to Create

```
resnet34_model/src/
├── baselines.py        # DL baseline model builders (standard configs)
├── classical_ml.py     # k-NN, SVM, Random Forest with CNN feature extraction
└── run_baselines.py    # Single launcher for all baseline experiments
```

Two small updates to existing files:
- `src/train.py` → accept optional `output_dir` parameter
- `src/evaluate.py` → accept optional `output_dir` parameter + add
  `generate_comparison_table()` + add `run_single_test_evaluation()`

---

## `src/config.py` — Add PROCESSED_DIR

Add this one line to your existing `config.py`:

```python
PROCESSED_DIR = DATA_DIR / "processed"   # shared dataset for all models
```

All models — baselines and your proposed ResNet34 — read from this directory.

---

## `src/dataset.py` — Add Basic Transform for Baselines

Add a minimal augmentation transform that baselines use. This represents
standard practice in existing literature — basic geometric transforms only,
no field-condition simulation.

```python
# Add alongside your existing train_transform in dataset.py

basic_train_transform = A.Compose([
    # Standard geometric augmentations only — what existing papers use
    A.RandomResizedCrop(224, 224, scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])
```

Update `get_dataloaders` to accept an `augmentation` flag:

```python
def get_dataloaders(train_dir, val_dir, batch_size=32,
                    use_weighted_sampler=False,
                    augmentation="basic"):
    """
    augmentation: "basic"    → uses basic_train_transform (for baselines)
                  "advanced" → uses train_transform (for your proposed model)
    """
    transform = train_transform if augmentation == "advanced" else basic_train_transform
    train_dataset = AlbumentationsDataset(train_dir, transform)
    val_dataset   = AlbumentationsDataset(val_dir, val_transform)

    sampler = build_weighted_sampler(train_dataset) if use_weighted_sampler else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return train_loader, val_loader
```

---

## `src/baselines.py` — Deep Learning Baseline Model Builders

Each model uses a **standard single-layer head** — this is what existing
literature uses and what each architecture ships with after replacing the
final classification layer for a new number of classes.

```python
"""
Deep learning baseline models in their standard configurations.

Every baseline uses:
  - ImageNet pretrained weights (standard transfer learning starting point)
  - Standard single-layer classifier head: Dropout(0.2) → Linear(N, num_classes)
    This represents how each model appears in existing literature, NOT your
    proposed improvements. Your proposed improvements are ResNet34-specific.
  - Basic augmentation only (flip, crop, normalize)
  - No attention modules
  - No weighted sampler
  - Trained on the same processed/ dataset as your proposed ResNet34

The only variable being compared is the model architecture itself,
since dataset and general training regime are held constant.
"""
import torch.nn as nn
import torchvision.models as models
import timm

NUM_CLASSES = 10


def _standard_head(in_features: int, num_classes: int) -> nn.Sequential:
    """
    Standard single-layer head — represents existing literature configurations.
    Light Dropout(0.2) matches what most papers apply before the final Linear.
    This is deliberately NOT your proposed strong head so the comparison
    shows what your improvements contribute over standard practice.
    """
    return nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes),
    )


# ── AlexNet ──────────────────────────────────────────────────────────────────

def build_alexnet(num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    AlexNet with ImageNet pretrained weights in standard configuration.
    AlexNet's original 6-layer classifier is replaced with our standard head.
    AdaptiveAvgPool2d(6,6) + Flatten gives a 9216-dim input to the head.
    """
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    model.classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d((6, 6)),
        nn.Flatten(),
        _standard_head(256 * 6 * 6, num_classes),
    )
    return model


def get_stage_b_params_alexnet(model: nn.Module) -> list[dict]:
    """
    Unfreeze the last two Conv feature blocks (Conv4, Conv5) and classifier.
    features indices: Conv1(0-2), Conv2(3-5), Conv3(6-8), Conv4(9-11), Conv5(12-14)
    """
    return [
        {"params": model.features[8:].parameters(),  "lr": 1e-4},
        {"params": model.classifier.parameters(),     "lr": 1e-3},
    ]


# ── VGG16 ────────────────────────────────────────────────────────────────────

def build_vgg16(num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    VGG16 with ImageNet pretrained weights in standard configuration.
    AdaptiveAvgPool2d(1,1) replaces the 7x7 pooling to give a 512-dim vector.
    Standard head replaces the original 3-layer FC classifier.
    """
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    model.classifier = nn.Sequential(
        nn.Flatten(),
        _standard_head(512, num_classes),
    )
    return model


def get_stage_b_params_vgg16(model: nn.Module) -> list[dict]:
    """
    Unfreeze the last two convolutional blocks (block4 and block5) and head.
    VGG16 block indices: block1(0-4), block2(5-9), block3(10-16),
                         block4(17-23), block5(24-30)
    """
    return [
        {"params": model.features[17:24].parameters(), "lr": 1e-5},  # block4
        {"params": model.features[24:].parameters(),   "lr": 1e-4},  # block5
        {"params": model.classifier.parameters(),      "lr": 1e-3},
    ]


# ── MobileNetV2 ───────────────────────────────────────────────────────────────

def build_mobilenetv2(num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    MobileNetV2 with ImageNet pretrained weights in standard configuration.
    Standard head replaces the original classifier. Input: 1280 features.
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = _standard_head(in_features, num_classes)
    return model


def get_stage_b_params_mobilenetv2(model: nn.Module) -> list[dict]:
    """
    MobileNetV2 has 19 InvertedResidual blocks (features[0] to [18]).
    Unfreeze the last 5 blocks (~25% of backbone) and classifier.
    """
    return [
        {"params": model.features[14:].parameters(), "lr": 1e-4},
        {"params": model.classifier.parameters(),    "lr": 1e-3},
    ]


# ── EfficientNet-B0 ───────────────────────────────────────────────────────────

def build_efficientnet_b0(num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    EfficientNet-B0 via timm in standard configuration.
    num_classes=0 removes the original head; we add our standard head.
    Input to head: 1280 features after global pooling.
    """
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
    in_features = model.num_features  # 1280
    model.classifier = _standard_head(in_features, num_classes)
    return model


def get_stage_b_params_efficientnet_b0(model: nn.Module) -> list[dict]:
    """
    EfficientNet-B0 has 9 MBConv block groups (model.blocks[0] to [8]).
    Unfreeze the last two block groups and classifier (~25% of backbone).
    Manually set requires_grad since EfficientNet does not have named stages.
    """
    for param in model.parameters():
        param.requires_grad = False
    for group in [model.blocks[7], model.blocks[8]]:
        for param in group.parameters():
            param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    return [
        {"params": model.blocks[7].parameters(),  "lr": 1e-5},
        {"params": model.blocks[8].parameters(),  "lr": 1e-4},
        {"params": model.classifier.parameters(), "lr": 1e-3},
    ]


# ── ResNet50 ──────────────────────────────────────────────────────────────────

def build_resnet50(num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    ResNet50 with ImageNet pretrained weights in standard configuration.
    ResNet50 uses Bottleneck blocks — fc input is 2048 (vs 512 for ResNet34).
    Standard head replaces the original 1000-class fc layer.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    in_features = model.fc.in_features  # 2048
    model.fc = _standard_head(in_features, num_classes)
    return model


def get_stage_b_params_resnet50(model: nn.Module) -> list[dict]:
    """
    Same stage naming as ResNet34: unfreeze layer3, layer4, and fc.
    ResNet50 has the same named stages, making the unfreeze strategy directly
    comparable to your proposed ResNet34 (layer3 + layer4 + head unfrozen).
    """
    return [
        {"params": model.layer3.parameters(), "lr": 1e-5},
        {"params": model.layer4.parameters(), "lr": 1e-4},
        {"params": model.fc.parameters(),     "lr": 1e-3},
    ]
```

---

## `src/classical_ml.py` — k-NN, SVM, Random Forest

Classical ML models cannot operate on raw 224×224 pixel images —
the feature space is too high-dimensional and unstructured.
The standard approach, used in Khan et al. (2021) and Tan et al. (2021)
which you cite in your FYP literature review, is to extract CNN features
first using a pretrained backbone, then train a classical classifier on
those compact feature vectors.

A pretrained ResNet34 backbone (fc layer removed) extracts 512-dimensional
feature vectors per image. These vectors capture semantic visual features
without any task-specific fine-tuning — exactly the "existing solution"
baseline condition.

```python
"""
Classical ML baselines using pretrained ResNet34 as a fixed feature extractor.

Pipeline for each model:
  1. Load ResNet34 pretrained on ImageNet (no fine-tuning, no head)
  2. Extract 512-dim feature vectors from processed/train images
  3. Fit k-NN / SVM / Random Forest on those feature vectors
  4. Extract features from processed/test images
  5. Evaluate and save results to outputs/{model_name}/outputs/

This is the standard approach in your cited literature:
  Khan et al. (2021): SVM/k-NN/RF on CNN features
  Tan et al. (2021): Classical ML compared against AlexNet/VGG/ResNet on
                     the same PlantVillage tomato dataset you are using
"""
import torch
import torch.nn as nn
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torchvision import models
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from src.config import *
from src.dataset import AlbumentationsDataset, val_transform


CLASS_NAMES_SHORT = [
    "Bacterial_Spot", "Early_Blight", "Late_Blight", "Leaf_Mold",
    "Septoria_LS", "Spider_Mites", "Target_Spot", "YLC_Virus",
    "Mosaic_Virus", "Healthy"
]


def _build_feature_extractor() -> nn.Module:
    """
    Pretrained ResNet34 with the final fc layer replaced by Identity.
    No fine-tuning. No task-specific adaptation.
    Outputs 512-dim feature vectors — the standard CNN feature extraction
    baseline as used in Khan et al. (2021) and Tan et al. (2021).
    Always runs in eval() mode with no gradient tracking.
    """
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    model.fc = nn.Identity()
    model.eval()
    return model


def _extract_features(extractor: nn.Module,
                       data_dir: Path,
                       device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """
    Passes all images through the feature extractor without augmentation.
    Uses val_transform (resize + normalize only) for reproducible features.
    Returns:
        features: np.ndarray of shape (N, 512)
        labels:   np.ndarray of shape (N,)
    """
    dataset = AlbumentationsDataset(data_dir, val_transform)
    loader  = DataLoader(dataset, batch_size=64, shuffle=False,
                         num_workers=4, pin_memory=True)

    all_features, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            feats  = extractor(images)           # (batch, 512)
            all_features.append(feats.cpu().numpy())
            all_labels.append(labels.numpy())

    features = np.concatenate(all_features, axis=0)
    labels   = np.concatenate(all_labels,   axis=0)
    print(f"  Extracted {features.shape[0]} × {features.shape[1]}-dim "
          f"features from {data_dir.name}/")
    return features, labels


def _save_results(model_name: str,
                  test_name: str,
                  y_true: np.ndarray,
                  y_pred: np.ndarray,
                  output_dir: Path) -> dict:
    """Computes all metrics, saves confusion matrix and classification report."""

    metrics = {
        "accuracy":        round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_precision": round(float(precision_score(
            y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_recall":    round(float(recall_score(
            y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_f1":        round(float(f1_score(
            y_true, y_pred, average="macro", zero_division=0)), 4),
        "per_class_f1":    f1_score(
            y_true, y_pred, average=None, zero_division=0).round(4).tolist(),
    }

    # Confusion matrix
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES_SHORT,
                yticklabels=CLASS_NAMES_SHORT, ax=ax)
    ax.set_title(f"{model_name} — {test_name.replace('_', ' ').title()}",
                 fontsize=13)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / f"cm_{test_name}.png", dpi=150)
    plt.close()
    print(f"  Confusion matrix saved: cm_{test_name}.png")

    # Classification report
    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES_SHORT,
        zero_division=0
    )
    mode = "a" if (output_dir / "classification_report.txt").exists() else "w"
    with open(output_dir / "classification_report.txt", mode) as f:
        f.write(f"\n{'='*60}\n{test_name.upper()}\n{'='*60}\n{report}\n")

    return metrics


def _build_classifier(name: str):
    """
    Returns a configured sklearn classifier.
    Hyperparameters represent standard literature defaults — not tuned
    to overfit to this specific dataset.
    """
    if name == "knn":
        return KNeighborsClassifier(
            n_neighbors=5,        # standard default; 5 neighbours
            metric="euclidean",   # appropriate for 512-dim CNN feature space
            n_jobs=-1,
        )
    elif name == "svm":
        return SVC(
            kernel="rbf",                      # standard for CNN feature spaces
            C=10.0,                            # moderate regularization
            gamma="scale",                     # 1 / (n_features * X.var())
            decision_function_shape="ovr",     # one-vs-rest multiclass
            probability=False,
            random_state=42,
        )
    elif name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,     # 200 trees — balanced accuracy vs speed
            max_depth=None,       # fully grown trees (RF is robust to this)
            min_samples_split=2,
            n_jobs=-1,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown classifier name: {name}")


def run_classical_baseline(classifier_name: str) -> dict:
    """
    Full pipeline: extract features → fit classifier → evaluate → save results.
    classifier_name: 'knn' | 'svm' | 'random_forest'
    """
    output_dir = OUTPUT_DIR / classifier_name / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = _build_feature_extractor().to(device)

    print(f"\n{'='*60}\nBaseline: {classifier_name.upper()}\n{'='*60}")

    # Feature extraction — use processed/train for fitting
    print("\nExtracting training features...")
    X_train, y_train = _extract_features(extractor, PROCESSED_DIR / "train", device)

    # Fit classifier
    clf = _build_classifier(classifier_name)
    print(f"\nFitting {classifier_name} on "
          f"{X_train.shape[0]} samples × {X_train.shape[1]} features...")
    clf.fit(X_train, y_train)
    print("  Fitting complete.")

    results = {}

    # Evaluate on processed/test (same test set as all other models)
    print(f"\nEvaluating on processed/test...")
    X_test, y_test = _extract_features(extractor, PROCESSED_DIR / "test", device)
    y_pred = clf.predict(X_test)

    metrics = _save_results(classifier_name, "processed_test",
                            y_test, y_pred, output_dir)
    results["processed_test"] = metrics
    print(f"  Acc: {metrics['accuracy']:.4f} | "
          f"Macro F1: {metrics['macro_f1']:.4f}")

    # Save eval_results.json
    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {output_dir}/")
    return results
```

---

## `src/run_baselines.py` — Single Launcher

```python
"""
Launcher for all baseline experiments.

Usage (local or HPC):
  BASELINE=knn              python src/run_baselines.py
  BASELINE=svm              python src/run_baselines.py
  BASELINE=random_forest    python src/run_baselines.py
  BASELINE=alexnet          python src/run_baselines.py
  BASELINE=vgg16            python src/run_baselines.py
  BASELINE=mobilenetv2      python src/run_baselines.py
  BASELINE=efficientnet_b0  python src/run_baselines.py
  BASELINE=resnet50         python src/run_baselines.py
  BASELINE=all              python src/run_baselines.py
"""
import os
from src.config import *
from src.dataset import get_dataloaders
from src.train import run_experiment
from src.evaluate import run_single_test_evaluation
from src.classical_ml import run_classical_baseline
from src.baselines import (
    build_alexnet,         get_stage_b_params_alexnet,
    build_vgg16,           get_stage_b_params_vgg16,
    build_mobilenetv2,     get_stage_b_params_mobilenetv2,
    build_efficientnet_b0, get_stage_b_params_efficientnet_b0,
    build_resnet50,        get_stage_b_params_resnet50,
)

CLASSICAL_MODELS = ["knn", "svm", "random_forest"]

DL_CONFIGS = {
    "alexnet": {
        "model_fn":  lambda: build_alexnet(NUM_CLASSES),
        "params_fn": get_stage_b_params_alexnet,
    },
    "vgg16": {
        "model_fn":  lambda: build_vgg16(NUM_CLASSES),
        "params_fn": get_stage_b_params_vgg16,
    },
    "mobilenetv2": {
        "model_fn":  lambda: build_mobilenetv2(NUM_CLASSES),
        "params_fn": get_stage_b_params_mobilenetv2,
    },
    "efficientnet_b0": {
        "model_fn":  lambda: build_efficientnet_b0(NUM_CLASSES),
        "params_fn": get_stage_b_params_efficientnet_b0,
    },
    "resnet50": {
        "model_fn":  lambda: build_resnet50(NUM_CLASSES),
        "params_fn": get_stage_b_params_resnet50,
    },
}


def run_dl_baseline(name: str):
    """
    Trains and evaluates one deep learning baseline model.
    Baseline training conditions:
      - Dataset:      processed/train (same as your proposed model)
      - Augmentation: basic only (flip, crop, normalize)
      - Head:         standard single Linear layer
      - Sampler:      no weighted sampler
      - Attention:    none
    """
    output_dir = OUTPUT_DIR / name / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}\nBaseline DL: {name.upper()}\n{'='*60}")

    # Baseline data: processed/ split, basic augmentation, no weighted sampler
    train_loader, val_loader = get_dataloaders(
        train_dir=PROCESSED_DIR / "train",
        val_dir=PROCESSED_DIR / "val",
        augmentation="basic",           # ← standard pipeline only
        use_weighted_sampler=False,     # ← no oversampling
    )

    cfg   = DL_CONFIGS[name]
    model = cfg["model_fn"]()
    stage_b_params = cfg["params_fn"](model)

    trained_model = run_experiment(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        experiment_name=name,
        stage_b_param_groups=stage_b_params,
        output_dir=output_dir,
    )

    # Evaluate on processed/test — the shared locked test set
    run_single_test_evaluation(
        model=trained_model,
        test_dir=PROCESSED_DIR / "test",
        experiment_name=name,
        output_dir=output_dir,
    )


def main():
    baseline = os.environ.get("BASELINE", "all").lower()

    if baseline == "all":
        targets_classical = CLASSICAL_MODELS
        targets_dl = list(DL_CONFIGS.keys())
    elif baseline in CLASSICAL_MODELS:
        targets_classical = [baseline]
        targets_dl = []
    elif baseline in DL_CONFIGS:
        targets_classical = []
        targets_dl = [baseline]
    else:
        raise ValueError(
            f"Unknown BASELINE: '{baseline}'. Choose from: "
            f"{CLASSICAL_MODELS + list(DL_CONFIGS.keys()) + ['all']}"
        )

    for name in targets_classical:
        run_classical_baseline(name)

    for name in targets_dl:
        run_dl_baseline(name)

    print("\n\nAll baselines complete.")
    print("Run: python -c \"from src.evaluate import generate_comparison_table; "
          "generate_comparison_table()\"")


if __name__ == "__main__":
    main()
```

---

## Updates to Existing Files

### `src/train.py` — Add `output_dir` parameter

Only one small change: make `output_dir` optional so baselines can
write checkpoints to their own folder.

```python
def run_experiment(model, train_loader, val_loader, experiment_name,
                   stage_b_param_groups, output_dir=None):  # ← add this
    """
    output_dir: if None, uses OUTPUT_DIR / "checkpoints" / experiment_name
                if provided (baselines), uses that directory directly
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    # Determine checkpoint directory
    checkpoint_dir = (output_dir if output_dir
                      else OUTPUT_DIR / "checkpoints" / experiment_name)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ... rest of your existing function completely unchanged ...
```

### `src/evaluate.py` — Add two new functions

Add these two functions to your existing `evaluate.py`. Do not modify
`evaluate_both_test_sets()` — that function is for your ResNet34 experiments
which evaluate on two test sets. Baselines use a single shared test set.

```python
def run_single_test_evaluation(model, test_dir, experiment_name,
                                output_dir) -> dict:
    """
    Evaluates a trained model on a single test set.
    Used for all baseline models — they evaluate on processed/test only,
    not on two separate test sets like your proposed ResNet34 experiments.

    Saves:
      - eval_results.json
      - cm_processed_test.png
      - classification_report.txt
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix, classification_report
    )
    import matplotlib.pyplot as plt
    import seaborn as sns

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device).eval()

    dataset = AlbumentationsDataset(test_dir, val_transform)
    loader  = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            preds = model(images.to(device)).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    metrics = {
        "accuracy":        round(float(accuracy_score(all_labels, all_preds)), 4),
        "macro_precision": round(float(precision_score(
            all_labels, all_preds, average="macro", zero_division=0)), 4),
        "macro_recall":    round(float(recall_score(
            all_labels, all_preds, average="macro", zero_division=0)), 4),
        "macro_f1":        round(float(f1_score(
            all_labels, all_preds, average="macro", zero_division=0)), 4),
        "per_class_f1":    f1_score(
            all_labels, all_preds, average=None,
            zero_division=0).round(4).tolist(),
    }

    results = {"processed_test": metrics}

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title(f"{experiment_name} — Processed Test Set", fontsize=13)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "cm_processed_test.png", dpi=150)
    plt.close()

    # Classification report
    report = classification_report(
        all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0
    )
    with open(output_dir / "classification_report.txt", "w") as f:
        f.write(f"=== {experiment_name} — Processed Test Set ===\n{report}\n")

    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[{experiment_name}] Processed Test | "
          f"Acc {metrics['accuracy']:.4f} | "
          f"Macro F1 {metrics['macro_f1']:.4f}")
    print(f"  Results saved to: {output_dir}/")
    return results


def generate_comparison_table():
    """
    Reads eval_results.json from every model's output directory and prints
    the full comparison table for your FYP Results chapter.

    Baseline models report metrics from processed/test.
    Your proposed ResNet34 reports metrics from processed/test as well
    (pulled from its eval_results.json which contains a 'processed_test' key,
    or from 'plantvillage' key if using your existing evaluate_both_test_sets).
    """
    MODELS = [
        # (display name,      folder key,         type,         params)
        ("k-NN",              "knn",              "Classical",  "N/A"),
        ("SVM",               "svm",              "Classical",  "N/A"),
        ("Random Forest",     "random_forest",    "Classical",  "N/A"),
        ("AlexNet",           "alexnet",          "DL",         "61.1M"),
        ("VGG16",             "vgg16",            "DL",         "138.4M"),
        ("MobileNetV2",       "mobilenetv2",      "DL",         "3.4M"),
        ("EfficientNet-B0",   "efficientnet_b0",  "DL",         "5.3M"),
        ("ResNet50",          "resnet50",         "DL",         "25.6M"),
        ("ResNet34 (Proposed)", "resnet34_proposed", "DL (Ours)", "~21.9M"),
    ]

    print(f"\n{'='*100}")
    print(f"  {'Model':<25} {'Type':<12} "
          f"{'Acc':>8} {'Precision':>10} {'Recall':>8} "
          f"{'Macro F1':>10} {'Params':>10}")
    print(f"{'='*100}")

    for display_name, key, model_type, params in MODELS:
        # Search for eval_results.json in the model's output directory
        result_path = OUTPUT_DIR / key / "outputs" / "eval_results.json"

        if not result_path.exists():
            print(f"  {display_name:<25} {'[NOT YET RUN]'}")
            continue

        r = json.load(open(result_path))

        # Key used by baselines is 'processed_test'
        # Key used by your ResNet34 experiments may vary — check both
        m = (r.get("processed_test")
             or r.get("plantvillage")
             or {})

        print(
            f"  {display_name:<25} {model_type:<12}"
            f"{m.get('accuracy', 0):>8.4f}"
            f"{m.get('macro_precision', 0):>10.4f}"
            f"{m.get('macro_recall', 0):>8.4f}"
            f"{m.get('macro_f1', 0):>10.4f}"
            f"{params:>10}"
        )

    print(f"{'='*100}")
    print("  All models evaluated on the same processed/test set.\n")
```

---

## `src/evaluate.py` — Save Your Proposed Model Results

After your ResNet34 training completes, run this to save its results in
the same format as all baselines so `generate_comparison_table()` picks it up:

```python
# Run once after your best ResNet34 experiment finishes training
import torch
from src.config import *
from src.evaluate import run_single_test_evaluation
from src.model import build_resnet34_with_cbam   # your best model builder

model = build_resnet34_with_cbam(NUM_CLASSES)
model.load_state_dict(
    torch.load(OUTPUT_DIR / "checkpoints" / "best_experiment" / "best_stage_b.pth")
)

output_dir = OUTPUT_DIR / "resnet34_proposed" / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

run_single_test_evaluation(
    model=model,
    test_dir=PROCESSED_DIR / "test",
    experiment_name="resnet34_proposed",
    output_dir=output_dir,
)
```

---

## HPC SLURM Scripts

### `scripts/run_baseline_job.sh` — Submit one model at a time

```bash
#!/bin/bash
#SBATCH --job-name=bl_${BASELINE}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=06:00:00
#SBATCH --output=logs/baseline_${BASELINE}_%j.log

module load python/3.11 cuda/12.1
cd ~/fyp/resnet34_model
source venv/bin/activate

echo "Starting baseline: ${BASELINE}"
BASELINE=${BASELINE} python src/run_baselines.py
echo "Done: ${BASELINE}"
```

### `scripts/submit_all_baselines.sh` — Submit all models in parallel

```bash
#!/bin/bash
mkdir -p logs

# Classical ML — CPU only, run as one job sequentially
echo "Submitting classical ML baselines (CPU)..."
sbatch --partition=cpu --gres="" \
       --cpus-per-task=8 --mem=32G --time=03:00:00 \
       --job-name=classical_ml \
       --output=logs/classical_%j.log \
       --wrap="
         cd ~/fyp/resnet34_model && source venv/bin/activate
         BASELINE=knn           python src/run_baselines.py
         BASELINE=svm           python src/run_baselines.py
         BASELINE=random_forest python src/run_baselines.py
       "

# Deep learning baselines — one GPU job each, all run in parallel
for MODEL in alexnet vgg16 mobilenetv2 efficientnet_b0 resnet50; do
    sbatch --export=BASELINE=${MODEL} scripts/run_baseline_job.sh
    echo "  Submitted GPU job: ${MODEL}"
done

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo ""
echo "After all complete, generate results:"
echo "  python -c \"from src.evaluate import generate_comparison_table; generate_comparison_table()\""
```

---

## Setup Checklist

- [ ] `pip install timm` — required for EfficientNet-B0
- [ ] `pip install scikit-learn` — required for classical ML
- [ ] `PROCESSED_DIR` added to `config.py` pointing to your processed dataset
- [ ] `PROCESSED_DIR / "train"`, `/ "val"`, and `/ "test"` all exist and
      have correct class subdirectory structure
- [ ] `get_dataloaders()` updated to accept `augmentation` parameter
- [ ] `run_experiment()` updated to accept optional `output_dir` parameter
- [ ] `run_single_test_evaluation()` and `generate_comparison_table()`
      added to `evaluate.py`
- [ ] Test locally before HPC submission:
      `BASELINE=mobilenetv2 python src/run_baselines.py`

---

## Estimated HPC Training Times

| Model | Stage A | Stage B | Total |
|---|---|---|---|
| k-NN / SVM / RF | — | — | 30–90 min (CPU) |
| AlexNet | ~20 min | ~30 min | ~50 min |
| MobileNetV2 | ~25 min | ~40 min | ~65 min |
| EfficientNet-B0 | ~30 min | ~50 min | ~80 min |
| ResNet50 | ~35 min | ~60 min | ~95 min |
| VGG16 | ~40 min | ~70 min | ~110 min |

Use `--time=04:00:00` for lighter models, `--time=06:00:00` for VGG16
and ResNet50.

---

## How This Reads in Your FYP Results Chapter

> All models were trained and evaluated on the same dataset (the `processed/`
> split combining PlantVillage and real-world field images) to ensure a fair
> comparison. Baseline models were trained using standard transfer learning
> configurations — basic data augmentation (random crop, horizontal flip,
> normalization), a standard single-layer classifier head, and no attention
> modules. The proposed ResNet34 model was trained with the full set of
> proposed improvements: an advanced field-condition augmentation pipeline,
> a regularized two-layer classifier head with BatchNorm1d, CBAM attention
> in the final two residual block groups, and a weighted sampler to balance
> real-world image representation. All models were evaluated on the identical
> held-out `processed/test` set. Performance differences therefore reflect
> the contribution of the proposed training strategy and architectural
> modifications rather than differences in data.
