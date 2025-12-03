from typing import Tuple

import torch
from torch import nn
from torchvision import models


class CameraBackbone(nn.Module):
    """
    Simple ResNet-based encoder for per-image features.
    """

    def __init__(self, out_channels: int = 64, pretrained: bool = False) -> None:
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])  # up to conv5
        self.proj = nn.Conv2d(512, out_channels, kernel_size=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        return self.proj(features)

    def images_to_bev(self, images: torch.Tensor, bev_shape: Tuple[int, int]) -> torch.Tensor:
        """
        Coarsely align camera features to BEV by adaptive pooling.
        This ignores calibration but keeps the fusion interface alive.
        """
        feats = self.forward(images)
        return nn.functional.adaptive_avg_pool2d(feats, output_size=bev_shape)
