import torch.nn as nn
import torchvision.models as models


def _standard_head(in_features: int, num_classes: int) -> nn.Sequential:
    """
    Standard single-layer head, representing how MobileNetV2 is used in
    existing literature for transfer learning — light Dropout(0.2) then
    a single Linear layer. This is deliberately NOT the proposed model's
    stronger BatchNorm/Dropout/2-layer head.
    """
    return nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes),
    )


def build_model(num_classes: int) -> nn.Module:
    """MobileNetV2 with ImageNet-pretrained weights in its standard configuration."""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = _standard_head(in_features, num_classes)
    return model


def classifier_parameters(model: nn.Module):
    return model.classifier.parameters()


def get_stage_b_params(model: nn.Module) -> list:
    """
    MobileNetV2 has 19 InvertedResidual blocks (features[0] to [18]).
    Standard fine-tuning: unfreeze the last 5 blocks (~25% of the
    backbone) plus the classifier — the conventional partial-unfreeze
    ratio used for this architecture, independent of the proposed
    ResNet34's layer3/layer4 targets.
    """
    for p in model.parameters():
        p.requires_grad = False
    for p in model.features[14:].parameters():
        p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True

    return [
        {"params": model.features[14:].parameters(), "lr": 1e-4},
        {"params": model.classifier.parameters(), "lr": 1e-3},
    ]
