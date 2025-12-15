"""
Utilities for creating target heatmaps and box regression targets from ground truth boxes.
Used for training object detection on BEV features.
"""

import math
from typing import List, Dict, Tuple, Optional

import torch
import numpy as np


# ============== Class Mapping ==============
# NuScenes has 23 detailed classes, we group them into 10 detection classes

NUSCENES_CLASSES = [
    "car",              # 0
    "truck",            # 1
    "bus",              # 2
    "trailer",          # 3
    "construction",     # 4
    "pedestrian",       # 5
    "motorcycle",       # 6
    "bicycle",          # 7
    "traffic_cone",     # 8
    "barrier",          # 9
]

# Map from NuScenes category names to our class indices
CATEGORY_TO_CLASS = {
    # Vehicles
    "vehicle.car": 0,
    "vehicle.truck": 1,
    "vehicle.bus.bendy": 2,
    "vehicle.bus.rigid": 2,
    "vehicle.trailer": 3,
    "vehicle.construction": 4,
    "vehicle.emergency.ambulance": 0,  # treat as car
    "vehicle.emergency.police": 0,     # treat as car
    # Two-wheelers
    "vehicle.motorcycle": 6,
    "vehicle.bicycle": 7,
    # Pedestrians (all types)
    "human.pedestrian.adult": 5,
    "human.pedestrian.child": 5,
    "human.pedestrian.wheelchair": 5,
    "human.pedestrian.stroller": 5,
    "human.pedestrian.personal_mobility": 5,
    "human.pedestrian.police_officer": 5,
    "human.pedestrian.construction_worker": 5,
    # Movable objects
    "movable_object.trafficcone": 8,
    "movable_object.barrier": 9,
    "movable_object.pushable_pullable": -1,  # ignore
    "movable_object.debris": -1,              # ignore
    # Other
    "animal": -1,                             # ignore
    "static_object.bicycle_rack": -1,         # ignore
}


def get_class_index(category_name: str) -> int:
    """Get class index from NuScenes category name. Returns -1 if should be ignored."""
    return CATEGORY_TO_CLASS.get(category_name, -1)


# ============== Gaussian Utilities ==============

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


# ============== Target Generation ==============

def boxes_to_targets(
    boxes: List[Dict],
    num_classes: int = 10,
    heatmap_size: Tuple[int, int] = (100, 100),
    x_range: Tuple[float, float] = (-50.0, 50.0),
    y_range: Tuple[float, float] = (-50.0, 50.0),
    min_radius: int = 2,
) -> Dict[str, torch.Tensor]:
    """
    Convert ground truth boxes to heatmaps and box regression targets.
    
    This creates:
    - heatmap: (num_classes, H, W) Gaussian peaks at object centers per class
    - box_targets: (7, H, W) regression targets [x_off, y_off, z, w, l, h, sin_yaw]
    - box_mask: (1, H, W) mask indicating which cells have valid box targets
    
    Args:
        boxes: List of box dictionaries from NuScenes dataset
        num_classes: Number of detection classes
        heatmap_size: (H, W) of the output
        x_range, y_range: BEV coordinate ranges in meters
        min_radius: Minimum Gaussian radius in pixels
    
    Returns:
        Dictionary with 'heatmap', 'box_targets', 'box_mask' tensors
    """
    H, W = heatmap_size
    
    # Initialize targets
    heatmap = np.zeros((num_classes, H, W), dtype=np.float32)
    box_targets = np.zeros((7, H, W), dtype=np.float32)  # x_off, y_off, z, w, l, h, sin_yaw
    box_mask = np.zeros((1, H, W), dtype=np.float32)
    
    # Resolution: meters per pixel
    res_x = (x_range[1] - x_range[0]) / W
    res_y = (y_range[1] - y_range[0]) / H
    
    for box in boxes:
        # Get class index
        class_idx = get_class_index(box["name"])
        if class_idx < 0 or class_idx >= num_classes:
            continue  # Skip ignored classes
        
        # Get box center in world coordinates
        x, y, z = box["translation"]
        
        # Check if box is within range
        if not (x_range[0] <= x <= x_range[1] and y_range[0] <= y <= y_range[1]):
            continue
        
        # Convert to pixel coordinates (float for sub-pixel offset)
        px_float = (x - x_range[0]) / res_x
        py_float = (y - y_range[0]) / res_y
        
        # Integer pixel location
        px = int(px_float)
        py = int(py_float)
        
        # Clamp to valid range
        px = max(0, min(W - 1, px))
        py = max(0, min(H - 1, py))
        
        # Get object size
        w, l, h = box["size"]  # width, length, height in meters
        obj_w_px = w / res_x
        obj_l_px = l / res_y
        
        # Compute Gaussian radius based on object size
        radius = gaussian_radius((obj_l_px, obj_w_px), min_overlap=0.7)
        radius = max(min_radius, radius)
        
        # Draw Gaussian on class heatmap
        draw_gaussian(heatmap[class_idx], (px, py), radius)
        
        # Store box regression targets at center pixel
        # Sub-pixel offset from integer cell center
        x_offset = px_float - px
        y_offset = py_float - py
        
        # Get rotation (yaw) - NuScenes uses quaternion, we need yaw angle
        rotation = box.get("rotation", [1, 0, 0, 0])  # wxyz quaternion
        # Convert quaternion to yaw (rotation around z-axis)
        qw, qx, qy, qz = rotation
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        
        # Store targets (normalized)
        box_targets[0, py, px] = x_offset  # x offset in [0, 1)
        box_targets[1, py, px] = y_offset  # y offset in [0, 1)
        box_targets[2, py, px] = z / 5.0   # z normalized (typical range -2 to 3m)
        box_targets[3, py, px] = w / 10.0  # width normalized (typical 0-10m)
        box_targets[4, py, px] = l / 10.0  # length normalized
        box_targets[5, py, px] = h / 5.0   # height normalized (typical 0-5m)
        box_targets[6, py, px] = math.sin(yaw)  # sin(yaw) for rotation
        
        box_mask[0, py, px] = 1.0  # Mark this cell as having valid target
    
    return {
        "heatmap": torch.from_numpy(heatmap),
        "box_targets": torch.from_numpy(box_targets),
        "box_mask": torch.from_numpy(box_mask),
    }


def batch_boxes_to_targets(
    boxes_batch: List[List[Dict]],
    num_classes: int = 10,
    heatmap_size: Tuple[int, int] = (100, 100),
    x_range: Tuple[float, float] = (-50.0, 50.0),
    y_range: Tuple[float, float] = (-50.0, 50.0),
) -> Dict[str, torch.Tensor]:
    """
    Convert a batch of box lists to targets.
    
    Returns:
        Dictionary with:
        - 'heatmap': (B, num_classes, H, W)
        - 'box_targets': (B, 7, H, W)
        - 'box_mask': (B, 1, H, W)
    """
    heatmaps = []
    box_targets_list = []
    box_masks = []
    
    for boxes in boxes_batch:
        targets = boxes_to_targets(
            boxes,
            num_classes=num_classes,
            heatmap_size=heatmap_size,
            x_range=x_range,
            y_range=y_range,
        )
        heatmaps.append(targets["heatmap"])
        box_targets_list.append(targets["box_targets"])
        box_masks.append(targets["box_mask"])
    
    return {
        "heatmap": torch.stack(heatmaps, dim=0),
        "box_targets": torch.stack(box_targets_list, dim=0),
        "box_mask": torch.stack(box_masks, dim=0),
    }


# ============== Legacy single-class functions (for backward compatibility) ==============

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
    (Legacy function for backward compatibility)
    """
    H, W = heatmap_size
    heatmap = np.zeros((H, W), dtype=np.float32)
    
    res_x = (x_range[1] - x_range[0]) / W
    res_y = (y_range[1] - y_range[0]) / H
    
    for box in boxes:
        name = box["name"].lower()
        if target_class.lower() not in name:
            continue
        
        x, y, z = box["translation"]
        
        if not (x_range[0] <= x <= x_range[1] and y_range[0] <= y <= y_range[1]):
            continue
        
        px = int((x - x_range[0]) / res_x)
        py = int((y - y_range[0]) / res_y)
        px = max(0, min(W - 1, px))
        py = max(0, min(H - 1, py))
        
        w, l, h = box["size"]
        obj_w_px = w / res_x
        obj_l_px = l / res_y
        
        radius = gaussian_radius((obj_l_px, obj_w_px), min_overlap=0.7)
        radius = max(min_radius, radius)
        
        draw_gaussian(heatmap, (px, py), radius)
    
    return torch.from_numpy(heatmap).unsqueeze(0)


def batch_boxes_to_heatmap(
    boxes_batch: List[List[Dict]],
    target_class: str = "car",
    heatmap_size: Tuple[int, int] = (100, 100),
    x_range: Tuple[float, float] = (-50.0, 50.0),
    y_range: Tuple[float, float] = (-50.0, 50.0),
) -> torch.Tensor:
    """
    Convert a batch of box lists to heatmaps (single class).
    (Legacy function for backward compatibility)
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
    
    return torch.stack(heatmaps, dim=0)
