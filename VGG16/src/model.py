import torch.nn as nn
import torchvision.models as models


def _standard_head(in_features: int, num_classes: int) -> nn.Sequential:
    """
    Standard single-layer head, representing how VGG16 is used in
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
    VGG16 with ImageNet-pretrained weights in its standard configuration.
    AdaptiveAvgPool2d(1,1) replaces the 7x7 pooling to give a 512-dim
    vector into the standard head.
    """
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    model.classifier = nn.Sequential(
        nn.Flatten(),
        _standard_head(512, num_classes),
    )
    return model


def classifier_parameters(model: nn.Module):
    return model.classifier.parameters()


def get_stage_b_params(model: nn.Module) -> list:
    """
    Standard fine-tuning: unfreeze the last two convolutional blocks
    (block4, block5) plus the classifier head.
    VGG16 block indices: block1(0-4) block2(5-9) block3(10-16)
                         block4(17-23) block5(24-30)
    """
    for p in model.parameters():
        p.requires_grad = False
    for p in model.features[17:].parameters():
        p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True

    return [
        {"params": model.features[17:24].parameters(), "lr": 1e-5},  # block4
        {"params": model.features[24:].parameters(), "lr": 1e-4},    # block5
        {"params": model.classifier.parameters(), "lr": 1e-3},
    ]
