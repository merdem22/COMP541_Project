"""
Camera Backbone - LSS (Lift-Splat-Shoot) Style BEV Projection
Projects multi-view camera features to BEV using depth distribution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    resnet18,
    resnet34,
    resnet50,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
)
from typing import List, Tuple
import numpy as np


class EfficientNetExtractor(nn.Module):
    """Lightweight feature extractor based on ResNet."""
    
    def __init__(self, backbone: str = "resnet34", pretrained: bool = True):
        super().__init__()
        
        if backbone == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            base = resnet18(weights=weights)
            self.out_channels = 512
        elif backbone == "resnet34":
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            base = resnet34(weights=weights)
            self.out_channels = 512
        else:
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            base = resnet50(weights=weights)
            self.out_channels = 2048
        
        # Remove final pooling and fc
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x


class DepthNet(nn.Module):
    """Predicts depth distribution for each pixel."""
    
    def __init__(
        self,
        in_channels: int,
        depth_channels: int = 64,
        feat_channels: int = 64,
        depth_bins: int = 64,
    ):
        super().__init__()
        
        self.depth_bins = depth_bins
        
        # Reduce channel dimension
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Depth prediction branch
        self.depth_conv = nn.Sequential(
            nn.Conv2d(256, depth_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(depth_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(depth_channels, depth_bins, 1),
        )
        
        # Feature branch (context encoding)
        self.feat_conv = nn.Sequential(
            nn.Conv2d(256, feat_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
        )
        
        self.feat_channels = feat_channels
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            depth: (B, D, H, W) depth distribution
            feat: (B, C, H, W) context features
        """
        x = self.reduce(x)
        depth = self.depth_conv(x).softmax(dim=1)
        feat = self.feat_conv(x)
        return depth, feat


class ViewTransformer(nn.Module):
    """Transforms camera features to BEV using LSS-style depth projection."""
    
    def __init__(
        self,
        feat_channels: int,
        depth_bins: int = 64,
        depth_min: float = 1.0,
        depth_max: float = 60.0,
        bev_x_bound: List[float] = [-51.2, 51.2, 0.4],
        bev_y_bound: List[float] = [-51.2, 51.2, 0.4],
        bev_z_bound: List[float] = [-5.0, 3.0, 8.0],
        downsample: int = 16,
    ):
        super().__init__()
        
        self.feat_channels = feat_channels
        self.depth_bins = depth_bins
        self.downsample = downsample
        
        # Depth bins
        self.depth_bins_values = torch.linspace(depth_min, depth_max, depth_bins)
        
        # BEV grid params
        self.bev_x_bound = bev_x_bound
        self.bev_y_bound = bev_y_bound
        self.bev_z_bound = bev_z_bound
        
        self.bev_h = int((bev_y_bound[1] - bev_y_bound[0]) / bev_y_bound[2])
        self.bev_w = int((bev_x_bound[1] - bev_x_bound[0]) / bev_x_bound[2])
    
    def forward(
        self,
        feat: torch.Tensor,
        depth: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        img_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Args:
            feat: (B, N_cams, C, H_feat, W_feat) camera features
            depth: (B, N_cams, D, H_feat, W_feat) depth distributions
            intrinsics: (B, N_cams, 3, 3) camera intrinsics
            extrinsics: (B, N_cams, 4, 4) camera extrinsics (sensor to ego)
            img_size: (H, W) original image size
        Returns:
            bev: (B, C, bev_H, bev_W) BEV features
        """
        B, N_cams, C, H_feat, W_feat = feat.shape
        D = self.depth_bins
        device = feat.device
        
        depth_bins = self.depth_bins_values.to(device)
        
        # Create frustum grid
        # (D, H_feat, W_feat, 3) - 3D points in camera frame for each pixel
        frustum = self._create_frustum(H_feat, W_feat, depth_bins, device)
        
        # Initialize BEV accumulator
        bev = torch.zeros(B, C, self.bev_h, self.bev_w, device=device)
        bev_count = torch.zeros(B, 1, self.bev_h, self.bev_w, device=device)
        
        for cam_idx in range(N_cams):
            cam_feat = feat[:, cam_idx]  # (B, C, H, W)
            cam_depth = depth[:, cam_idx]  # (B, D, H, W)
            cam_K = intrinsics[:, cam_idx]  # (B, 3, 3)
            cam_E = extrinsics[:, cam_idx]  # (B, 4, 4)
            
            # Lift features with depth
            # feat_lifted: (B, C, D, H, W)
            feat_lifted = cam_feat.unsqueeze(2) * cam_depth.unsqueeze(1)
            
            # Project frustum to ego frame
            points_ego = self._frustum_to_ego(
                frustum, cam_K, cam_E, img_size, H_feat, W_feat
            )  # (B, D, H, W, 3)
            
            # Splat to BEV
            bev, bev_count = self._splat_to_bev(
                feat_lifted, points_ego, bev, bev_count
            )
        
        # Normalize by count
        bev = bev / (bev_count + 1e-5)
        
        return bev
    
    def _create_frustum(
        self, H: int, W: int, depth_bins: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """Create frustum of 3D points for each pixel."""
        D = len(depth_bins)
        
        # Pixel coordinates
        xs = torch.linspace(0, W - 1, W, device=device)
        ys = torch.linspace(0, H - 1, H, device=device)
        ys, xs = torch.meshgrid(ys, xs, indexing='ij')
        
        # Add depth dimension
        xs = xs.unsqueeze(0).expand(D, -1, -1)
        ys = ys.unsqueeze(0).expand(D, -1, -1)
        ds = depth_bins.view(D, 1, 1).expand(-1, H, W)
        
        # Stack: (D, H, W, 3) with (u, v, depth)
        frustum = torch.stack([xs, ys, ds], dim=-1)
        
        return frustum
    
    def _frustum_to_ego(
        self,
        frustum: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        img_size: Tuple[int, int],
        H_feat: int,
        W_feat: int,
    ) -> torch.Tensor:
        """Project frustum points from image to ego coordinates."""
        B = intrinsics.shape[0]
        D, H, W, _ = frustum.shape
        device = frustum.device
        
        # Scale intrinsics for feature map size
        scale_x = W_feat / (img_size[1] / self.downsample)
        scale_y = H_feat / (img_size[0] / self.downsample)
        
        K = intrinsics.clone()
        K[:, 0, :] *= self.downsample / scale_x
        K[:, 1, :] *= self.downsample / scale_y
        
        # Flatten frustum
        frustum_flat = frustum.view(-1, 3)  # (D*H*W, 3)
        u, v, d = frustum_flat[:, 0], frustum_flat[:, 1], frustum_flat[:, 2]
        
        # Unproject to camera coordinates
        points_cam = []
        for b in range(B):
            K_inv = torch.inverse(K[b])
            
            # Homogeneous pixel coords
            uv1 = torch.stack([u * d, v * d, d], dim=-1)  # (D*H*W, 3)
            
            # To camera frame
            pts_cam = torch.matmul(K_inv, uv1.T).T  # (D*H*W, 3)
            
            # To ego frame
            E = extrinsics[b]
            pts_ego = torch.matmul(E[:3, :3], pts_cam.T).T + E[:3, 3]
            
            points_cam.append(pts_ego.view(D, H, W, 3))
        
        return torch.stack(points_cam)  # (B, D, H, W, 3)
    
    def _splat_to_bev(
        self,
        feat: torch.Tensor,
        points: torch.Tensor,
        bev: torch.Tensor,
        bev_count: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Splat lifted features to BEV grid."""
        B, C, D, H, W = feat.shape
        device = feat.device
        
        # Flatten spatial dimensions
        feat_flat = feat.permute(0, 2, 3, 4, 1).reshape(B, -1, C)  # (B, D*H*W, C)
        points_flat = points.reshape(B, -1, 3)  # (B, D*H*W, 3)
        
        # Compute BEV indices
        bev_x = ((points_flat[..., 0] - self.bev_x_bound[0]) / self.bev_x_bound[2]).long()
        bev_y = ((points_flat[..., 1] - self.bev_y_bound[0]) / self.bev_y_bound[2]).long()
        
        # Valid mask (within BEV bounds and z range)
        valid = (
            (bev_x >= 0) & (bev_x < self.bev_w) &
            (bev_y >= 0) & (bev_y < self.bev_h) &
            (points_flat[..., 2] >= self.bev_z_bound[0]) &
            (points_flat[..., 2] <= self.bev_z_bound[1])
        )
        
        # Scatter to BEV
        for b in range(B):
            mask = valid[b]
            if mask.sum() == 0:
                continue
            
            x_idx = bev_x[b][mask]
            y_idx = bev_y[b][mask]
            feat_valid = feat_flat[b][mask]  # (N_valid, C)
            
            # Scatter add
            idx = y_idx * self.bev_w + x_idx
            
            bev_flat = bev[b].view(C, -1)
            bev_flat.scatter_add_(1, idx.unsqueeze(0).expand(C, -1), feat_valid.T)
            
            count_flat = bev_count[b].view(1, -1)
            count_flat.scatter_add_(1, idx.unsqueeze(0), torch.ones_like(idx).unsqueeze(0).float())
        
        return bev, bev_count


class CameraBackbone(nn.Module):
    """Full camera backbone: Image encoder + Depth prediction + BEV projection."""
    
    def __init__(
        self,
        backbone: str = "resnet34",
        pretrained: bool = True,
        depth_channels: int = 64,
        depth_bins: int = 64,
        depth_min: float = 1.0,
        depth_max: float = 60.0,
        feat_channels: int = 64,
        bev_x_bound: List[float] = [-51.2, 51.2, 0.4],
        bev_y_bound: List[float] = [-51.2, 51.2, 0.4],
        bev_z_bound: List[float] = [-5.0, 3.0, 8.0],
        out_channels: int = 128,
        img_size: Tuple[int, int] = (256, 704),
        downsample: int = 32,
    ):
        super().__init__()
        
        self.img_size = img_size
        
        # Image backbone
        self.encoder = EfficientNetExtractor(backbone, pretrained)
        
        # Depth and feature prediction
        self.depth_net = DepthNet(
            in_channels=self.encoder.out_channels,
            depth_channels=depth_channels,
            feat_channels=feat_channels,
            depth_bins=depth_bins,
        )
        
        # View transformer
        self.view_transformer = ViewTransformer(
            feat_channels=feat_channels,
            depth_bins=depth_bins,
            depth_min=depth_min,
            depth_max=depth_max,
            bev_x_bound=bev_x_bound,
            bev_y_bound=bev_y_bound,
            bev_z_bound=bev_z_bound,
            downsample=downsample,
        )
        
        # BEV refinement
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(feat_channels, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        self.out_channels = out_channels
    
    def forward(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            images: (B, N_cams, 3, H, W) multi-view images
            intrinsics: (B, N_cams, 3, 3) camera intrinsics
            extrinsics: (B, N_cams, 4, 4) camera extrinsics
        Returns:
            bev: (B, C, H_bev, W_bev) BEV features
        """
        B, N_cams, C_img, H, W = images.shape
        
        # Reshape for batch processing
        images_flat = images.view(B * N_cams, C_img, H, W)
        
        # Extract features
        feat = self.encoder(images_flat)
        
        # Predict depth and context
        depth, context = self.depth_net(feat)
        
        # Reshape back
        _, C_feat, H_feat, W_feat = context.shape
        D = depth.shape[1]
        
        context = context.view(B, N_cams, C_feat, H_feat, W_feat)
        depth = depth.view(B, N_cams, D, H_feat, W_feat)
        
        # Transform to BEV
        bev = self.view_transformer(
            context, depth, intrinsics, extrinsics, (H, W)
        )
        
        # Refine BEV
        bev = self.bev_encoder(bev)
        
        return bev
