# Updated by AI
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from src.models.graph_module_placeholder import StaticGraphModule
from src.models.backbone_camera import CameraBackbone
from src.models.backbone_lidar import SimpleLidarBackbone


class ConcatFusion(nn.Module):
    def forward(self, lidar_feat: torch.Tensor, camera_feat: Optional[torch.Tensor]) -> torch.Tensor:
        if camera_feat is None:
            return lidar_feat
        if camera_feat.shape[-2:] != lidar_feat.shape[-2:]:
            camera_feat = F.interpolate(
                camera_feat,
                size=lidar_feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return torch.cat([lidar_feat, camera_feat], dim=1)


class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4, camera_channels: int | None = None) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.camera_proj = None
        if camera_channels and camera_channels != embed_dim:
            self.camera_proj = nn.Conv2d(camera_channels, embed_dim, kernel_size=1)

    def forward(self, lidar_feat: torch.Tensor, camera_feat: Optional[torch.Tensor]) -> torch.Tensor:
        if camera_feat is None:
            return lidar_feat
        if self.camera_proj:
            camera_feat = self.camera_proj(camera_feat)
        if camera_feat.shape[-2:] != lidar_feat.shape[-2:]:
            camera_feat = F.interpolate(
                camera_feat,
                size=lidar_feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        b, c, h, w = lidar_feat.shape
        lidar_tokens = lidar_feat.flatten(2).transpose(1, 2)  # (B, HW, C)
        camera_tokens = camera_feat.flatten(2).transpose(1, 2)
        fused_tokens, _ = self.attn(lidar_tokens, camera_tokens, camera_tokens)
        return fused_tokens.transpose(1, 2).view(b, c, h, w)


class SimpleDetectionHead(nn.Module):
    """
    Minimal detection head producing a heatmap and box regression.
    """

    def __init__(self, in_channels: int, hidden: int = 64, num_classes: int = 10) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.class_head = nn.Conv2d(hidden, num_classes, kernel_size=1)
        # CenterNet prior: initialize bias so sigmoid outputs ~0.1 initially
        # This prevents exploding detection counts at the start of training
        # -2.19 corresponds to sigmoid(-2.19) ≈ 0.1
        nn.init.constant_(self.class_head.bias, -2.19)
        self.box_head = nn.Conv2d(hidden, 7, kernel_size=1)  # x, y, z, w, l, h, yaw

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.body(x)
        heatmap = torch.sigmoid(self.class_head(features))
        box = self.box_head(features)
        return {"heatmap": heatmap, "box": box}


class FusionBaselineModel(nn.Module):
    """
    LiDAR + camera baseline with selectable fusion strategy.
    """

    def __init__(
        self,
        lidar_in_channels: int = 2,
        lidar_feat_channels: int = 64,
        camera_feat_channels: int = 64,
        fusion_mode: str = "concat",
        num_classes: int = 10,
        use_graph: bool = False,          # <-- NEW
        graph_k: int = 8,                 # <-- NEW (unused now but nice for later)
    ) -> None:
        super().__init__()
        self.lidar_backbone = SimpleLidarBackbone(
            in_channels=lidar_in_channels,
            out_channels=lidar_feat_channels,
        )
        self.camera_backbone = CameraBackbone(out_channels=camera_feat_channels, pretrained=False)
        self.fusion_mode = fusion_mode

        if fusion_mode == "cross_attn":
            self.fusion = CrossAttentionFusion(
                embed_dim=lidar_feat_channels,
                camera_channels=camera_feat_channels,
            )
            head_in_channels = lidar_feat_channels
        else:
            self.fusion = ConcatFusion()
            head_in_channels = lidar_feat_channels + camera_feat_channels

        # NEW: graph module after fusion, before detection head
        self.use_graph = use_graph
        if use_graph:
            self.graph = StaticGraphModule(in_channels=head_in_channels, k_neighbors=graph_k)
        else:
            self.graph = None

        self.head = SimpleDetectionHead(in_channels=head_in_channels, num_classes=num_classes)

    def forward(
        self,
        lidar_bev: torch.Tensor,
        camera_bev: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        lidar_features = self.lidar_backbone(lidar_bev)
        fused = self.fusion(lidar_features, camera_bev)

        if self.graph is not None:
            fused = self.graph(fused)

        return self.head(fused)
