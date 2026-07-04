import torch.nn as nn
import torchvision.models as models


def _standard_head(in_features: int, num_classes: int) -> nn.Sequential:
    """
    Standard single-layer head, representing how ResNet50 is used in
    existing literature for transfer learning — light Dropout(0.2) then
    a single Linear layer. This is deliberately NOT the proposed model's
    stronger BatchNorm/Dropout/2-layer head, and there is no CBAM
    attention injected into layer3/layer4 as the proposed ResNet34 does.
    """
    return nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes),
    )


def build_model(num_classes: int) -> nn.Module:
    """
    ResNet50 with ImageNet-pretrained weights in its standard configuration.
    ResNet50 uses Bottleneck blocks — fc input is 2048 (vs 512 for ResNet34).
    """
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    in_features = model.fc.in_features  # 2048
    model.fc = _standard_head(in_features, num_classes)
    return model


def classifier_parameters(model: nn.Module):
    return model.fc.parameters()


def get_stage_b_params(model: nn.Module) -> list:
    """
    ResNet50 shares the same named stages as the proposed ResNet34
    (layer1-4 + fc), so the fairest apples-to-apples unfreeze target is
    the same layer3/layer4/fc combination. Unlike the proposed model,
    there is no CBAM attention and no label smoothing — only the
    unfreeze target names are comparable, not the regularization.
    """
    for p in model.parameters():
        p.requires_grad = False
    for p in model.layer3.parameters():
        p.requires_grad = True
    for p in model.layer4.parameters():
        p.requires_grad = True
    for p in model.fc.parameters():
        p.requires_grad = True

    return [
        {"params": model.layer3.parameters(), "lr": 1e-5},
        {"params": model.layer4.parameters(), "lr": 1e-4},
        {"params": model.fc.parameters(), "lr": 1e-3},
    ]
