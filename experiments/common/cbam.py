"""CBAM (Convolutional Block Attention Module) — the attention component of
the solution stack.

Channel attention (SE-style) recalibrates *which* feature channels matter;
spatial attention learns *where* on the feature map to look, which helps
suppress background clutter in real-world field photos. Both branches are
initialized near-identity (final weight zeroed) so inserting CBAM into a
pretrained backbone does not disrupt the ImageNet features at the start of
training — the attention is learned, not imposed.

The channel branch is built lazily on the first forward pass, inferring the
channel count from the input tensor. This lets a single implementation be
dropped after any stage of any backbone (ResNet, VGG, AlexNet, MobileNet,
EfficientNet) with no per-architecture channel bookkeeping. Callers must run
one dummy forward at build time (backbones.warm_up) so all lazy parameters
exist before the optimizer is constructed.
"""
import torch
import torch.nn as nn


class CBAM(nn.Module):
    def __init__(self, channels: int = None, reduction: int = 16):
        super().__init__()
        self.reduction = reduction
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.spatial_attention[0].weight)
        self.channel_attention = None
        if channels is not None:
            self._build_channel(channels)

    def _build_channel(self, channels: int) -> None:
        hidden = max(channels // self.reduction, 1)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.channel_attention[-2].weight)

    def forward(self, x):
        if self.channel_attention is None:
            self._build_channel(x.shape[1])
            self.channel_attention.to(x.device)

        b, c, _, _ = x.shape
        ca = self.channel_attention(x).view(b, c, 1, 1)
        x = x * ca

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        return x * sa


class Sequential_CBAM(nn.Module):
    """Runs `module`, then attends to its output with CBAM. Used to append
    attention after a backbone stage while leaving the stage itself intact
    (respects any internal residual add, since CBAM sees the post-add output).
    """
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.cbam = CBAM()  # channels inferred lazily

    def forward(self, x):
        return self.cbam(self.module(x))
