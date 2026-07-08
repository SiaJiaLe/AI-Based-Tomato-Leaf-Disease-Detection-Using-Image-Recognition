"""Backbone construction for all six CNN architectures under the OFF/ON stack.

Each builder returns a BuiltModel that hides per-architecture structure behind
one interface the engine can use uniformly:

  * head swap          — plain vs strong head (heads.py), by feature dim.
  * CBAM insertion     — Stack ON only, appended after the last two backbone
                         stages (cbam.Sequential_CBAM). Insertion points are
                         mapped per architecture (CP2 spec section 5); the
                         attention is applied to each stage's *output*, so any
                         internal residual add is respected.
  * Stage-B unfreeze   — one_group (OFF: deepest stage only) or two_group
                         (ON: deepest two stages), with per-group learning
                         rates (deeper stage warms faster than the shallower).

CBAM channel widths are inferred lazily on first forward, so a single dummy
pass (warm_up) must run before the optimizer is built.
"""
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from .cbam import Sequential_CBAM
from .heads import build_head

# Stage-B learning rates: (shallower-of-the-two, deeper, head).
STAGE_B_LR_DEEP = 1e-4      # deepest stage / single unfrozen stage
STAGE_B_LR_SHALLOW = 1e-5   # the second (shallower) stage when two are unfrozen
STAGE_B_LR_HEAD = 1e-3      # classifier head always fastest


@dataclass
class BuiltModel:
    module: nn.Module
    head: nn.Module
    # Ordered shallow -> deep. Each group is a list of modules whose params are
    # toggled together (a "stage" may span several submodules, e.g. a conv plus
    # its BN, or an MBConv stage plus the final conv head).
    groups: List[List[nn.Module]] = field(default_factory=list)

    def warm_up(self, device) -> None:
        """One dummy forward to materialize lazy CBAM channel branches so all
        parameters exist before the optimizer is constructed."""
        self.module.eval()
        with torch.no_grad():
            self.module(torch.zeros(2, 3, 224, 224, device=device))

    def freeze_to_head(self):
        for p in self.module.parameters():
            p.requires_grad = False
        for p in self.head.parameters():
            p.requires_grad = True
        return self.head.parameters()

    def configure_stage_b(self, two_group: bool):
        """Unfreeze the deepest one or two stages (+head) and return Adam param
        groups with per-stage learning rates."""
        for p in self.module.parameters():
            p.requires_grad = False
        for p in self.head.parameters():
            p.requires_grad = True

        chosen = self.groups[-2:] if two_group else self.groups[-1:]
        # LR schedule: with two stages the shallower gets 1e-5, deeper 1e-4;
        # with one stage it gets 1e-4.
        lrs = [STAGE_B_LR_SHALLOW, STAGE_B_LR_DEEP] if two_group else [STAGE_B_LR_DEEP]

        param_dicts = []
        for group, lr in zip(chosen, lrs):
            params = []
            for module in group:
                for p in module.parameters():
                    p.requires_grad = True
                    params.append(p)
            param_dicts.append({"params": params, "lr": lr})
        param_dicts.append({"params": list(self.head.parameters()), "lr": STAGE_B_LR_HEAD})
        return param_dicts


def _split_vgg_blocks(features: nn.Sequential) -> List[nn.Sequential]:
    """Split VGG features into conv blocks delimited by MaxPool2d (5 blocks)."""
    blocks, current = [], []
    for m in features:
        current.append(m)
        if isinstance(m, nn.MaxPool2d):
            blocks.append(nn.Sequential(*current))
            current = []
    if current:
        blocks.append(nn.Sequential(*current))
    return blocks


def _build_resnet(num_classes, strong, cbam, depth):
    if depth == 34:
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = build_head(in_features, num_classes, strong)
    if cbam:
        model.layer3 = Sequential_CBAM(model.layer3)
        model.layer4 = Sequential_CBAM(model.layer4)
    groups = [[model.layer1], [model.layer2], [model.layer3], [model.layer4]]
    return BuiltModel(module=model, head=model.fc, groups=groups)


def _build_vgg16(num_classes, strong, cbam):
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    blocks = _split_vgg_blocks(model.features)
    model.features = nn.Sequential(*blocks)
    if cbam:
        model.features[4] = Sequential_CBAM(model.features[4])
        model.features[3] = Sequential_CBAM(model.features[3])
    # Keep the pretrained 25088->4096->4096 classifier body; replace only the
    # final Linear with our head on the 4096-dim features.
    pre = list(model.classifier.children())[:-1]
    head_tail = build_head(4096, num_classes, strong)
    model.classifier = nn.Sequential(*pre, head_tail)
    groups = [[model.features[i]] for i in range(len(model.features))]
    return BuiltModel(module=model, head=model.classifier, groups=groups)


def _build_alexnet(num_classes, strong, cbam):
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    f = model.features
    stages = [
        nn.Sequential(f[0], f[1], f[2]),   # conv1 + pool
        nn.Sequential(f[3], f[4], f[5]),   # conv2 + pool
        nn.Sequential(f[6], f[7]),         # conv3
        nn.Sequential(f[8], f[9]),         # conv4
        nn.Sequential(f[10], f[11], f[12]),  # conv5 + pool
    ]
    model.features = nn.Sequential(*stages)
    if cbam:
        model.features[4] = Sequential_CBAM(model.features[4])
        model.features[3] = Sequential_CBAM(model.features[3])
    pre = list(model.classifier.children())[:-1]
    head_tail = build_head(4096, num_classes, strong)
    model.classifier = nn.Sequential(*pre, head_tail)
    groups = [[model.features[i]] for i in range(len(model.features))]
    return BuiltModel(module=model, head=model.classifier, groups=groups)


def _build_mobilenetv2(num_classes, strong, cbam):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    feats = list(model.features)  # 0 stem, 1..17 inverted residuals, 18 final conv (1280)
    shallow = nn.Sequential(*feats[0:14])
    mid = nn.Sequential(*feats[14:17])   # 160-channel inverted-residual stage
    deep = nn.Sequential(*feats[17:19])  # 320 IR + final 1280 conv
    model.features = nn.Sequential(shallow, mid, deep)
    if cbam:
        model.features[2] = Sequential_CBAM(model.features[2])
        model.features[1] = Sequential_CBAM(model.features[1])
    in_features = model.classifier[-1].in_features  # 1280
    model.classifier = build_head(in_features, num_classes, strong)
    groups = [[model.features[0]], [model.features[1]], [model.features[2]]]
    return BuiltModel(module=model, head=model.classifier, groups=groups)


def _build_efficientnetb0(num_classes, strong, cbam):
    import timm
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=num_classes)
    in_features = model.num_features  # 1280
    model.classifier = build_head(in_features, num_classes, strong)
    if cbam:
        model.blocks[-1] = Sequential_CBAM(model.blocks[-1])
        model.blocks[-2] = Sequential_CBAM(model.blocks[-2])
    # Deepest group folds in conv_head/bn2 (post-blocks stem) so deep
    # fine-tuning actually reaches the final feature projection.
    groups = [[model.blocks[i]] for i in range(len(model.blocks) - 2)]
    groups.append([model.blocks[-2]])
    groups.append([model.blocks[-1], model.conv_head, model.bn2])
    return BuiltModel(module=model, head=model.classifier, groups=groups)


_BUILDERS = {
    "resnet34": lambda n, s, c: _build_resnet(n, s, c, 34),
    "resnet50": lambda n, s, c: _build_resnet(n, s, c, 50),
    "vgg16": _build_vgg16,
    "alexnet": _build_alexnet,
    "mobilenetv2": _build_mobilenetv2,
    "efficientnetb0": _build_efficientnetb0,
}


def build_backbone(name: str, num_classes: int, strong_head: bool, cbam: bool) -> BuiltModel:
    if name not in _BUILDERS:
        raise ValueError(f"Unknown backbone '{name}'. Choices: {sorted(_BUILDERS)}")
    return _BUILDERS[name](num_classes, strong_head, cbam)
