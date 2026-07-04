import torch.nn as nn
import torchvision.models as models


def _standard_head(in_features: int, num_classes: int) -> nn.Sequential:
    """
    Standard single-layer head, representing how AlexNet is used in
    existing literature for transfer learning — light Dropout(0.2) then
    a single Linear layer. This is deliberately NOT the proposed model's
    stronger BatchNorm/Dropout/2-layer head.
    """
    return nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes),
    )


def build_model(num_classes: int) -> nn.Module:
    """
    AlexNet with ImageNet-pretrained weights in its standard configuration.
    torchvision's AlexNet.forward() already applies model.avgpool
    (AdaptiveAvgPool2d((6, 6))) and flattens to 2D before calling
    model.classifier, so the replacement classifier must NOT repeat
    that pooling/flatten step — it receives an already-flat 256*6*6
    vector.
    """
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    model.classifier = _standard_head(256 * 6 * 6, num_classes)
    return model


def classifier_parameters(model: nn.Module):
    return model.classifier.parameters()


def get_stage_b_params(model: nn.Module) -> list:
    """
    Standard fine-tuning: unfreeze the last two conv feature blocks
    (Conv4, Conv5) plus the classifier head.
    features indices: Conv1(0-2) Conv2(3-5) Conv3(6-8) Conv4(9-11) Conv5(12-14)
    """
    for p in model.parameters():
        p.requires_grad = False
    for p in model.features[9:].parameters():
        p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True

    return [
        {"params": model.features[9:].parameters(), "lr": 1e-4},
        {"params": model.classifier.parameters(), "lr": 1e-3},
    ]
