from typing import Tuple

import numpy as np
import torch
from torch import nn


def points_to_bev(
    points: torch.Tensor,
    x_range: Tuple[float, float] = (-50.0, 50.0),
    y_range: Tuple[float, float] = (-50.0, 50.0),
    resolution: float = 0.5,
) -> torch.Tensor:
    """
    Rasterize lidar points into a coarse BEV feature map with two channels:
    - max height per cell
    - normalized point count
    """
    h = int((y_range[1] - y_range[0]) / resolution)
    w = int((x_range[1] - x_range[0]) / resolution)
    if points.numel() == 0:
        return torch.zeros((2, h, w), dtype=torch.float32)

    pts = points.detach().cpu().numpy()
    xy = pts[:, :2]
    z = pts[:, 2]

    mask = (
        (xy[:, 0] >= x_range[0])
        & (xy[:, 0] <= x_range[1])
        & (xy[:, 1] >= y_range[0])
        & (xy[:, 1] <= y_range[1])
    )
    xy = xy[mask]
    z = z[mask]

    if xy.size == 0:
        return torch.zeros((2, h, w), dtype=torch.float32)

    grid_x = ((xy[:, 0] - x_range[0]) / resolution).astype(np.int64)
    grid_y = ((xy[:, 1] - y_range[0]) / resolution).astype(np.int64)

    height_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    np.maximum.at(height_map, (grid_y, grid_x), z)
    np.add.at(count_map, (grid_y, grid_x), 1.0)

    if count_map.max() > 0:
        count_map = count_map / count_map.max()

    bev = np.stack([height_map, count_map], axis=0)
    return torch.from_numpy(bev)


class SimpleLidarBackbone(nn.Module):
    """
    A tiny lidar BEV backbone: 3 conv blocks.
    """

    def __init__(self, in_channels: int = 2, base_channels: int = 32, out_channels: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        return self.encoder(bev)

    def forward_points(
        self,
        lidar_points: torch.Tensor,
        x_range: Tuple[float, float] = (-50.0, 50.0),
        y_range: Tuple[float, float] = (-50.0, 50.0),
        resolution: float = 0.5,
    ) -> torch.Tensor:
        bev = points_to_bev(lidar_points, x_range=x_range, y_range=y_range, resolution=resolution)
        bev = bev.unsqueeze(0) if bev.ndim == 3 else bev
        return self.forward(bev)
