import torch
import torch.nn as nn
import types
from torchvision.models import resnet34, ResNet34_Weights
import torchvision

class CBAM(nn.Module):
    """Combines channel attention (SE-style) with spatial attention.
    The spatial attention component helps suppress background clutter
    in real-world field photos by learning where to look on the leaf."""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Channel attention (same as SE)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        # Spatial attention — learns WHERE to focus on the feature map
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        b, c, _, _ = x.shape
        ca = self.channel_attention(x).view(b, c, 1, 1)
        x = x * ca

        # Spatial attention: pool across channels to get spatial map
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_in = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(spatial_in)
        return x * sa

def add_cbam_to_pretrained_resnet34(model):
    """Insert CBAM blocks into layer3 and layer4 only.
    Early layers (layer1, layer2) are left untouched to preserve
    low-level texture features from ImageNet pretraining."""
    for name, module in model.named_modules():
        if isinstance(module, torchvision.models.resnet.BasicBlock):
            layer_name = name.split('.')[0]
            if layer_name in ('layer3', 'layer4'):
                channels = module.conv2.out_channels
                cbam = CBAM(channels)
                
                # Initialize channel attention so it starts as near-identity
                nn.init.zeros_(cbam.channel_attention[-2].weight)
                # Initialize spatial attention so it starts as near-identity
                nn.init.zeros_(cbam.spatial_attention[0].weight)
                
                module.cbam = cbam
                
                # Patch the forward method to include CBAM before the residual addition
                def new_forward(self, x):
                    identity = x
                    out = self.conv1(x)
                    out = self.bn1(out)
                    out = self.relu(out)
                    
                    out = self.conv2(out)
                    out = self.bn2(out)
                    
                    # Apply CBAM
                    out = self.cbam(out)
                    
                    if self.downsample is not None:
                        identity = self.downsample(x)
                        
                    out += identity
                    out = self.relu(out)
                    
                    return out
                
                # Bind the new forward method to the instance
                module.forward = types.MethodType(new_forward, module)
    return model

class TomatoResNet34(nn.Module):
    def __init__(self, num_classes):
        super(TomatoResNet34, self).__init__()
        self.model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        
        # Inject CBAM attention into layer3 and layer4
        self.model = add_cbam_to_pretrained_resnet34(self.model)
        
        in_features = self.model.fc.in_features
        # Advanced Regularization: BatchNorm1d + Dropout + 2-layer FC
        # Re-centers features to adapt to noisy real-world distribution
        self.model.fc = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
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
