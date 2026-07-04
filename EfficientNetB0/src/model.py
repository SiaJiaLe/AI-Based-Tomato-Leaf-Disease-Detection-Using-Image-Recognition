import torch.nn as nn
import timm


def _standard_head(in_features: int, num_classes: int) -> nn.Sequential:
    """
    Standard single-layer head, representing how EfficientNet-B0 is used
    in existing literature for transfer learning — light Dropout(0.2)
    then a single Linear layer. This is deliberately NOT the proposed
    model's stronger BatchNorm/Dropout/2-layer head.
    """
    return nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes),
    )


def build_model(num_classes: int) -> nn.Module:
    """
    EfficientNet-B0 via timm in its standard configuration.
    num_classes=0 removes the original head; the standard head is
    attached afterwards. Input to head: 1280 features after global
    pooling (timm's forward_head applies global_pool then classifier).
    """
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
    in_features = model.num_features  # 1280
    model.classifier = _standard_head(in_features, num_classes)
    return model


def classifier_parameters(model: nn.Module):
    return model.classifier.parameters()


def get_stage_b_params(model: nn.Module) -> list:
    """
    EfficientNet-B0 has 9 MBConv block groups (model.blocks[0] to [8]).
    Standard fine-tuning: unfreeze the last two block groups (~25% of the
    backbone) plus the classifier — timm has no named layer3/layer4
    stages, so requires_grad is set manually per block group.
    """
    for p in model.parameters():
        p.requires_grad = False
    for group in [model.blocks[7], model.blocks[8]]:
        for p in group.parameters():
            p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True

    return [
        {"params": model.blocks[7].parameters(), "lr": 1e-5},
        {"params": model.blocks[8].parameters(), "lr": 1e-4},
        {"params": model.classifier.parameters(), "lr": 1e-3},
    ]
