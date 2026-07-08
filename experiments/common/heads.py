"""Classifier heads — the "strong head" component of the solution stack.

Stack OFF uses a plain head (light dropout -> single linear), representing how
these backbones are used in standard transfer-learning literature. Stack ON
uses a stronger two-layer head with BatchNorm1d that re-centers backbone
features before classification, which helps adapt to the noisier real-world
distribution. The head is the only place the two conditions may legitimately
differ in classifier design; everything upstream is governed by the backbone
and CBAM flags.

Both heads are parameterized purely by `in_features` so the identical design
pattern applies to every backbone (ResNet34 512, ResNet50 2048, VGG16 4096,
AlexNet 4096, MobileNetV2 1280, EfficientNetB0 1280).
"""
import torch.nn as nn


def plain_head(in_features: int, num_classes: int) -> nn.Sequential:
    """Standard baseline head: Dropout(0.2) -> Linear."""
    return nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes),
    )


def strong_head(in_features: int, num_classes: int) -> nn.Sequential:
    """Solution-stack head: BatchNorm1d -> Dropout -> Linear -> ReLU ->
    Dropout -> Linear. Same design as the proposed ResNet34 model."""
    return nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )


def build_head(in_features: int, num_classes: int, strong: bool) -> nn.Sequential:
    return strong_head(in_features, num_classes) if strong else plain_head(in_features, num_classes)
