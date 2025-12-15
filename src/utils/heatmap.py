"""
Utilities for creating target heatmaps from ground truth boxes.
Used for training object detection on BEV features.
"""

import math
from typing import List, Dict, Tuple

import torch
import numpy as np


def gaussian_2d(shape: Tuple[int, int], sigma: float = 1.0) -> np.ndarray:
    """
    Generate a 2D Gaussian kernel.
    
    Args:
        shape: (height, width) of the kernel
        sigma: standard deviation of the Gaussian
    
    Returns:
        2D numpy array with Gaussian values, peak=1 at center
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap: np.ndarray, center: Tuple[int, int], radius: int, k: float = 1.0) -> np.ndarray:
    """
    Draw a Gaussian peak on the heatmap at the given center.
    
    Args:
        heatmap: (H, W) array to draw on (modified in place)
        center: (x, y) pixel coordinates of the center
        radius: radius of the Gaussian (determines sigma)
        k: peak value (default 1.0)
    
    Returns:
        Modified heatmap
    """
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)
    
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape
    
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    
    if left + right <= 0 or top + bottom <= 0:
        return heatmap
    
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    
    if masked_heatmap.shape[0] > 0 and masked_heatmap.shape[1] > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    
    return heatmap


def gaussian_radius(det_size: Tuple[float, float], min_overlap: float = 0.7) -> int:
    """
    Compute Gaussian radius based on object size (CenterNet style).
    
    Args:
        det_size: (height, width) of the object in pixels
        min_overlap: minimum IoU overlap required
    
    Returns:
        Gaussian radius in pixels
    """
    height, width = det_size
    
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2
    
    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2
    
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    
    return max(0, int(min(r1, r2, r3)))


def boxes_to_heatmap(
    boxes: List[Dict],
    target_class: str = "car",
    heatmap_size: Tuple[int, int] = (100, 100),
    x_range: Tuple[float, float] = (-50.0, 50.0),
    y_range: Tuple[float, float] = (-50.0, 50.0),
    min_radius: int = 2,
) -> torch.Tensor:
    """
    Convert ground truth boxes to a Gaussian heatmap for a single class.
    
    Args:
        boxes: List of box dictionaries from NuScenes dataset
               Each box has 'translation' (x,y,z), 'size' (w,l,h), 'name'
        target_class: Class name to filter for (e.g., "car", "pedestrian")
        heatmap_size: (H, W) of the output heatmap
        x_range: (min_x, max_x) in meters for BEV
        y_range: (min_y, max_y) in meters for BEV
        min_radius: Minimum Gaussian radius in pixels
    
    Returns:
        Tensor of shape (1, H, W) with Gaussian peaks at object centers
    """
    H, W = heatmap_size
    heatmap = np.zeros((H, W), dtype=np.float32)
    
    # Resolution: meters per pixel
    res_x = (x_range[1] - x_range[0]) / W
    res_y = (y_range[1] - y_range[0]) / H
    
    for box in boxes:
        # Filter by class name (nuScenes uses hierarchical names like "vehicle.car")
        name = box["name"].lower()
        if target_class.lower() not in name:
            continue
        
        # Get box center in world coordinates
        x, y, z = box["translation"]
        
        # Check if box is within range
        if not (x_range[0] <= x <= x_range[1] and y_range[0] <= y <= y_range[1]):
            continue
        
        # Convert to pixel coordinates
        px = int((x - x_range[0]) / res_x)
        py = int((y - y_range[0]) / res_y)
        
        # Clamp to valid range
        px = max(0, min(W - 1, px))
        py = max(0, min(H - 1, py))
        
        # Get object size in pixels for Gaussian radius
        w, l, h = box["size"]  # width, length, height in meters
        obj_w_px = w / res_x
        obj_l_px = l / res_y
        
        # Compute Gaussian radius based on object size
        radius = gaussian_radius((obj_l_px, obj_w_px), min_overlap=0.7)
        radius = max(min_radius, radius)
        
        # Draw Gaussian at center
        draw_gaussian(heatmap, (px, py), radius)
    
    return torch.from_numpy(heatmap).unsqueeze(0)  # (1, H, W)


def batch_boxes_to_heatmap(
    boxes_batch: List[List[Dict]],
    target_class: str = "car",
    heatmap_size: Tuple[int, int] = (100, 100),
    x_range: Tuple[float, float] = (-50.0, 50.0),
    y_range: Tuple[float, float] = (-50.0, 50.0),
) -> torch.Tensor:
    """
    Convert a batch of box lists to heatmaps.
    
    Args:
        boxes_batch: List of box lists, one per sample in batch
        target_class: Class to detect
        heatmap_size: Output size
        x_range, y_range: BEV coordinate ranges
    
    Returns:
        Tensor of shape (B, 1, H, W)
    """
    heatmaps = []
    for boxes in boxes_batch:
        hm = boxes_to_heatmap(
            boxes, 
            target_class=target_class,
            heatmap_size=heatmap_size,
            x_range=x_range,
            y_range=y_range,
        )
        heatmaps.append(hm)
    
    return torch.stack(heatmaps, dim=0)  # (B, 1, H, W)

