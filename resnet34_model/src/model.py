import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights

class TomatoResNet34(nn.Module):
    def __init__(self, num_classes):
        super(TomatoResNet34, self).__init__()
        self.model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        
        in_features = self.model.fc.in_features
        # Advanced Regularization: Dropout to prevent pixel memorization
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)
        
    def freeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True
            
    def unfreeze_layer3_and_4(self):
        """Unfreeze layer3 and layer4 for deeper fine-tuning on real-world textures"""
        for param in self.model.layer3.parameters():
            param.requires_grad = True
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        for param in self.model.fc.parameters():
            param.requires_grad = True
