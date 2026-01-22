"""
LiDAR Backbone - Pillar-based BEV Feature Extraction
Inspired by PointPillars but with enhanced feature learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import numpy as np


class PillarVFE(nn.Module):
    """Pillar Feature Encoder - extracts features from point pillars."""
    
    def __init__(
        self,
        in_channels: int = 5,
        feat_channels: int = 64,
        voxel_size: List[float] = [0.1, 0.1, 8.0],
        point_cloud_range: List[float] = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        max_points_per_pillar: int = 32,
        max_pillars: int = 30000,
    ):
        super().__init__()
        
        self.voxel_size = torch.tensor(voxel_size)
        self.point_cloud_range = torch.tensor(point_cloud_range)
        self.max_points_per_pillar = max_points_per_pillar
        self.max_pillars = max_pillars
        
        # Pillar dimensions
        self.nx = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
        self.ny = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])
        
        # Input: [x, y, z, intensity, ring, x_c, y_c, z_c, x_p, y_p] 
        # x_c, y_c, z_c: offset from pillar center
        # x_p, y_p: offset from pillar x,y location
        augmented_channels = in_channels + 5
        
        self.pfn = nn.Sequential(
            nn.Linear(augmented_channels, feat_channels, bias=False),
            nn.BatchNorm1d(feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels, bias=False),
            nn.BatchNorm1d(feat_channels),
            nn.ReLU(inplace=True),
        )
        
        self.feat_channels = feat_channels
    
    def forward(self, points: torch.Tensor, points_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: (B, N, 5) point cloud [x, y, z, intensity, ring]
            points_mask: (B, N) valid point mask
        Returns:
            pillar_features: (B, C, H, W) BEV feature map
        """
        batch_size = points.shape[0]
        device = points.device
        
        # Move constants to device
        voxel_size = self.voxel_size.to(device)
        pc_range = self.point_cloud_range.to(device)
        
        bev_features = []
        
        for b in range(batch_size):
            pts = points[b][points_mask[b]]  # (N_valid, 5)
            
            if pts.shape[0] == 0:
                bev_features.append(torch.zeros(self.feat_channels, self.ny, self.nx, device=device))
                continue
            
            # Compute pillar indices
            pillar_x = ((pts[:, 0] - pc_range[0]) / voxel_size[0]).long()
            pillar_y = ((pts[:, 1] - pc_range[1]) / voxel_size[1]).long()
            
            # Clip to valid range
            pillar_x = torch.clamp(pillar_x, 0, self.nx - 1)
            pillar_y = torch.clamp(pillar_y, 0, self.ny - 1)
            
            # Flatten pillar indices
            pillar_idx = pillar_y * self.nx + pillar_x
            
            # Get unique pillars and their counts
            unique_pillars, inverse_idx, counts = torch.unique(
                pillar_idx, return_inverse=True, return_counts=True
            )
            
            # Limit number of pillars
            if len(unique_pillars) > self.max_pillars:
                perm = torch.randperm(len(unique_pillars))[:self.max_pillars]
                keep_mask = torch.isin(inverse_idx, perm)
                pts = pts[keep_mask]
                pillar_idx = pillar_idx[keep_mask]
                pillar_x = pillar_x[keep_mask]
                pillar_y = pillar_y[keep_mask]
                unique_pillars, inverse_idx, counts = torch.unique(
                    pillar_idx, return_inverse=True, return_counts=True
                )
            
            num_pillars = len(unique_pillars)
            
            # Compute pillar centers
            pillar_centers_x = (unique_pillars % self.nx).float() * voxel_size[0] + pc_range[0] + voxel_size[0] / 2
            pillar_centers_y = (unique_pillars // self.nx).float() * voxel_size[1] + pc_range[1] + voxel_size[1] / 2
            
            # Augment point features
            # Offset from pillar center
            x_c = pts[:, 0] - pillar_centers_x[inverse_idx]
            y_c = pts[:, 1] - pillar_centers_y[inverse_idx]
            z_c = pts[:, 2] - pts[:, 2].mean()  # offset from mean z
            
            # Offset from pillar location
            x_p = pts[:, 0] - (pillar_x.float() * voxel_size[0] + pc_range[0])
            y_p = pts[:, 1] - (pillar_y.float() * voxel_size[1] + pc_range[1])
            
            # Concatenate augmented features
            pts_aug = torch.cat([pts, x_c.unsqueeze(1), y_c.unsqueeze(1), z_c.unsqueeze(1),
                                x_p.unsqueeze(1), y_p.unsqueeze(1)], dim=1)
            
            # Apply PFN
            pts_feat = self.pfn(pts_aug)  # (N_points, C)
            
            # Scatter max to get pillar features
            pillar_feats = torch.zeros(num_pillars, self.feat_channels, device=device, dtype=pts_feat.dtype)
            pillar_feats.scatter_reduce_(
                0,
                inverse_idx.unsqueeze(1).expand(-1, self.feat_channels),
                pts_feat,
                reduce='amax',
                include_self=False
            )
            
            # Scatter to BEV
            bev = torch.zeros(self.feat_channels, self.ny * self.nx, device=device, dtype=pts_feat.dtype)
            bev[:, unique_pillars] = pillar_feats.T
            bev = bev.view(self.feat_channels, self.ny, self.nx)
            
            bev_features.append(bev)
        
        return torch.stack(bev_features)


class LiDARBackbone(nn.Module):
    """Full LiDAR backbone: Pillar VFE + 2D CNN for BEV processing."""
    
    def __init__(
        self,
        in_channels: int = 5,
        voxel_size: List[float] = [0.4, 0.4, 8.0],
        point_cloud_range: List[float] = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        encoder_channels: List[int] = [64, 128, 256],
        out_channels: int = 128,
    ):
        super().__init__()
        
        self.pillar_vfe = PillarVFE(
            in_channels=in_channels,
            feat_channels=encoder_channels[0],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_points_per_pillar=32,
            max_pillars=30000,
        )
        
        # 2D CNN backbone for BEV feature extraction
        self.blocks = nn.ModuleList()
        in_ch = encoder_channels[0]
        
        for i, out_ch in enumerate(encoder_channels):
            block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=2 if i > 0 else 1, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
            self.blocks.append(block)
            in_ch = out_ch
        
        # Multi-scale fusion with upsampling
        self.deblocks = nn.ModuleList()
        self.lateral_convs = nn.ModuleList()
        
        for i, ch in enumerate(reversed(encoder_channels[:-1])):
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(in_ch, ch, 2, stride=2, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
            ))
            self.lateral_convs.append(nn.Sequential(
                nn.Conv2d(ch, ch, 1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
            ))
            in_ch = ch
        
        # Final projection
        self.out_conv = nn.Sequential(
            nn.Conv2d(encoder_channels[0], out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        self.out_channels = out_channels
    
    def forward(self, points: torch.Tensor, points_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: (B, N, 5) point cloud
            points_mask: (B, N) valid point mask
        Returns:
            bev_features: (B, C, H, W) BEV features
        """
        # Pillar encoding
        x = self.pillar_vfe(points, points_mask)
        
        # Encoder with skip connections
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        
        # Decoder with skip connections
        for i, (deblock, lateral) in enumerate(zip(self.deblocks, self.lateral_convs)):
            x = deblock(x)
            skip = lateral(features[-(i+2)])
            x = x + skip
        
        # Final projection
        x = self.out_conv(x)
        
        return x
