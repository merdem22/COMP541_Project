"""
Detection Head - CenterPoint-style Heatmap-based 3D Object Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class SeparateHead(nn.Module):
    """Separate prediction head for each task."""
    
    def __init__(
        self,
        in_channels: int,
        heads: Dict[str, int],
        head_channels: int = 64,
        final_kernel: int = 1,
    ):
        super().__init__()
        
        self.heads = heads
        
        for head_name, num_channels in heads.items():
            head = nn.Sequential(
                nn.Conv2d(in_channels, head_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(head_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_channels, num_channels, final_kernel),
            )
            
            # Initialize
            if head_name == 'heatmap':
                head[-1].bias.data.fill_(-2.19)  # focal loss init
            else:
                for m in head.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            
            setattr(self, head_name, head)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {name: getattr(self, name)(x) for name in self.heads}


class DetectionHead(nn.Module):
    """
    CenterPoint-style detection head.
    Predicts heatmaps and regression targets for each object class.
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        shared_channels: int = 256,
        num_classes: int = 10,
        common_heads: Dict[str, int] = None,
        tasks: List[Dict] = None,
        bev_x_bound: List[float] = [-51.2, 51.2, 0.4],
        bev_y_bound: List[float] = [-51.2, 51.2, 0.4],
    ):
        super().__init__()
        
        if common_heads is None:
            common_heads = {
                'reg': 2,      # x, y offset
                'height': 1,   # z
                'dim': 3,      # w, l, h
                'rot': 2,      # sin, cos
                'vel': 2,      # vx, vy
            }
        
        if tasks is None:
            # Default: one task per class
            tasks = [{'num_class': 1, 'class_names': [f'class_{i}']} for i in range(num_classes)]
        
        self.tasks = tasks
        self.num_classes = num_classes
        self.bev_x_bound = bev_x_bound
        self.bev_y_bound = bev_y_bound
        
        # Shared conv layers
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, shared_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(shared_channels),
            nn.ReLU(inplace=True),
        )
        
        # Task-specific heads
        self.task_heads = nn.ModuleList()
        for task in tasks:
            heads = {'heatmap': task['num_class']}
            heads.update(common_heads)
            self.task_heads.append(SeparateHead(shared_channels, heads))
    
    def forward(self, x: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """
        Args:
            x: (B, C, H, W) BEV features
        Returns:
            List of task predictions, each containing:
                - heatmap: (B, num_class, H, W)
                - reg: (B, 2, H, W)
                - height: (B, 1, H, W)
                - dim: (B, 3, H, W)
                - rot: (B, 2, H, W)
                - vel: (B, 2, H, W)
        """
        x = self.shared_conv(x)
        return [head(x) for head in self.task_heads]
    
    def get_targets(
        self,
        annotations: List[Dict],
        device: torch.device,
        bev_shape: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        """
        Generate training targets from annotations.
        
        Args:
            annotations: List of annotation dicts per batch item
            device: Target device
            bev_shape: (H, W) of BEV feature map
        """
        batch_size = len(annotations)
        H, W = bev_shape
        
        # Class to task mapping
        class_to_task = {}
        for task_idx, task in enumerate(self.tasks):
            for class_name in task['class_names']:
                class_to_task[class_name] = task_idx
        
        # Initialize targets
        num_tasks = len(self.tasks)
        targets = {
            'heatmaps': [torch.zeros(batch_size, task['num_class'], H, W, device=device) 
                        for task in self.tasks],
            'reg': torch.zeros(batch_size, 2, H, W, device=device),
            'height': torch.zeros(batch_size, 1, H, W, device=device),
            'dim': torch.zeros(batch_size, 3, H, W, device=device),
            'rot': torch.zeros(batch_size, 2, H, W, device=device),
            'vel': torch.zeros(batch_size, 2, H, W, device=device),
            'reg_mask': torch.zeros(batch_size, H, W, device=device, dtype=torch.bool),
            'task_reg_masks': [torch.zeros(batch_size, H, W, device=device, dtype=torch.bool)
                               for _ in self.tasks],
            'indices': [],
        }
        
        # BEV conversion params
        x_min, x_max, x_res = self.bev_x_bound
        y_min, y_max, y_res = self.bev_y_bound
        
        for b, ann in enumerate(annotations):
            boxes = ann['boxes']
            labels = ann['labels']
            vels = ann['velocities']
            
            if len(boxes) == 0:
                targets['indices'].append(torch.zeros(0, dtype=torch.long, device=device))
                continue
            
            # Convert box centers to BEV indices
            x_idx = ((boxes[:, 0] - x_min) / x_res).long()
            y_idx = ((boxes[:, 1] - y_min) / y_res).long()
            
            # Clip to valid range
            valid = (x_idx >= 0) & (x_idx < W) & (y_idx >= 0) & (y_idx < H)
            x_idx = x_idx[valid].clamp(0, W - 1)
            y_idx = y_idx[valid].clamp(0, H - 1)
            boxes = boxes[valid]
            labels = labels[valid]
            vels = vels[valid]
            
            if len(boxes) == 0:
                targets['indices'].append(torch.zeros(0, dtype=torch.long, device=device))
                continue
            
            # Create heatmap targets with Gaussian
            for i, (x, y, box, label, vel) in enumerate(zip(x_idx, y_idx, boxes, labels, vels)):
                # Find task for this class
                from src.data import CLASS_NAMES
                class_name = CLASS_NAMES[label.item()]
                task_idx = class_to_task.get(class_name, 0)
                
                # Get class index within task
                task_classes = self.tasks[task_idx]['class_names']
                class_idx_in_task = task_classes.index(class_name) if class_name in task_classes else 0
                
                # Generate Gaussian heatmap
                radius = self._gaussian_radius(box[3:5].cpu().numpy(), x_res)
                radius = max(1, int(radius))
                
                self._draw_gaussian(
                    targets['heatmaps'][task_idx][b, class_idx_in_task],
                    (x.item(), y.item()),
                    radius
                )
                
                # Regression targets at center
                targets['reg'][b, 0, y, x] = box[0] - (x.float() * x_res + x_min)
                targets['reg'][b, 1, y, x] = box[1] - (y.float() * y_res + y_min)
                targets['height'][b, 0, y, x] = box[2]
                targets['dim'][b, :, y, x] = box[3:6]
                targets['rot'][b, 0, y, x] = torch.sin(box[6])
                targets['rot'][b, 1, y, x] = torch.cos(box[6])
                targets['vel'][b, :, y, x] = vel
                targets['reg_mask'][b, y, x] = True
                targets['task_reg_masks'][task_idx][b, y, x] = True
            
            targets['indices'].append(y_idx * W + x_idx)
        
        return targets
    
    def _gaussian_radius(self, box_size: np.ndarray, voxel_size: float) -> float:
        """Compute Gaussian radius based on box size."""
        # Use minimum of width and length
        min_side = min(box_size[0], box_size[1]) / voxel_size
        return max(1.0, min_side / 2.0)
    
    def _draw_gaussian(
        self,
        heatmap: torch.Tensor,
        center: Tuple[int, int],
        radius: int,
    ):
        """Draw Gaussian on heatmap."""
        H, W = heatmap.shape
        x, y = center
        
        # Create Gaussian kernel
        diameter = 2 * radius + 1
        sigma = diameter / 6.0
        
        xs = torch.arange(diameter, device=heatmap.device) - radius
        ys = torch.arange(diameter, device=heatmap.device) - radius
        ys, xs = torch.meshgrid(ys, xs, indexing='ij')
        
        gaussian = torch.exp(-(xs**2 + ys**2) / (2 * sigma**2))
        
        # Clip to valid region
        left = max(0, x - radius)
        right = min(W, x + radius + 1)
        top = max(0, y - radius)
        bottom = min(H, y + radius + 1)
        
        g_left = max(0, radius - x)
        g_right = g_left + (right - left)
        g_top = max(0, radius - y)
        g_bottom = g_top + (bottom - top)
        
        # Apply maximum (don't overwrite larger values)
        masked_heatmap = heatmap[top:bottom, left:right]
        masked_gaussian = gaussian[g_top:g_bottom, g_left:g_right]
        
        if masked_heatmap.shape == masked_gaussian.shape:
            heatmap[top:bottom, left:right] = torch.maximum(masked_heatmap, masked_gaussian)


class DetectionLoss(nn.Module):
    """Loss function for CenterPoint-style detection."""
    
    def __init__(
        self,
        loss_weights: Dict[str, float] = None,
    ):
        super().__init__()
        
        if loss_weights is None:
            loss_weights = {
                'heatmap': 1.0,
                'reg': 2.0,
                'height': 0.2,
                'dim': 0.2,
                'rot': 0.2,
                'vel': 0.2,
            }
        
        self.loss_weights = loss_weights
    
    def forward(
        self,
        predictions: List[Dict[str, torch.Tensor]],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute detection loss.
        
        Args:
            predictions: List of task predictions
            targets: Target tensors from get_targets()
        """
        losses = {}

        device = predictions[0]['heatmap'].device

        # Heatmap loss (focal loss)
        heatmap_loss = torch.zeros((), device=device)
        for task_idx, pred in enumerate(predictions):
            hm = pred['heatmap']
            hm_target = targets['heatmaps'][task_idx]
            heatmap_loss += self._focal_loss(hm.sigmoid(), hm_target)
        
        losses['heatmap'] = heatmap_loss / len(predictions) * self.loss_weights['heatmap']
        
        # Regression losses (only at object centers, per task)
        reg_losses = {
            key: torch.zeros((), device=device)
            for key in ['reg', 'height', 'dim', 'rot', 'vel']
        }
        
        for task_idx, pred in enumerate(predictions):
            mask = targets['task_reg_masks'][task_idx]
            if mask.sum() == 0:
                # Ensure reg heads participate in the graph even with no positives.
                for key in ['reg', 'height', 'dim', 'rot', 'vel']:
                    reg_losses[key] += pred[key].sum() * 0.0
                continue
            num_pos = mask.sum().clamp(min=1).float()
            
            for key in ['reg', 'height', 'dim', 'rot', 'vel']:
                target = targets[key]
                loss = F.l1_loss(
                    pred[key].permute(0, 2, 3, 1)[mask],
                    target.permute(0, 2, 3, 1)[mask],
                    reduction='sum'
                ) / num_pos
                reg_losses[key] += loss
        
        for key in ['reg', 'height', 'dim', 'rot', 'vel']:
            losses[key] = (reg_losses[key] / max(1, len(predictions))) * self.loss_weights[key]
        
        losses['total'] = sum(losses.values())
        
        return losses
    
    def _focal_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        alpha: float = 2.0,
        beta: float = 4.0,
    ) -> torch.Tensor:
        """Focal loss for heatmap prediction."""
        pos_mask = target.eq(1).float()
        neg_mask = target.lt(1).float()
        
        neg_weights = (1 - target).pow(beta)
        
        pos_loss = -torch.log(pred.clamp(min=1e-6)) * (1 - pred).pow(alpha) * pos_mask
        neg_loss = -torch.log((1 - pred).clamp(min=1e-6)) * pred.pow(alpha) * neg_weights * neg_mask
        
        num_pos = pos_mask.sum().clamp(min=1)
        loss = (pos_loss.sum() + neg_loss.sum()) / num_pos
        
        return loss


def decode_predictions(
    predictions: List[Dict[str, torch.Tensor]],
    bev_x_bound: List[float],
    bev_y_bound: List[float],
    score_thresh: float = 0.1,
    nms_thresh: float = 0.2,
    max_dets: int = 300,
    pre_nms_topk: int | None = None,
) -> List[Dict[str, torch.Tensor]]:
    """
    Decode network predictions to 3D boxes.
    
    Returns:
        List of detection dicts per batch item:
            - boxes: (N, 7) [x, y, z, w, l, h, yaw]
            - scores: (N,)
            - labels: (N,)
            - velocities: (N, 2)
    """
    batch_size = predictions[0]['heatmap'].shape[0]
    device = predictions[0]['heatmap'].device
    
    x_min, x_max, x_res = bev_x_bound
    y_min, y_max, y_res = bev_y_bound
    
    results = []
    if pre_nms_topk is None:
        # Safety cap to avoid huge candidate sets early in training when heatmaps are noisy.
        pre_nms_topk = max(1000, max_dets * 10)
    
    for b in range(batch_size):
        all_boxes = []
        all_scores = []
        all_labels = []
        all_vels = []
        
        label_offset = 0
        
        for task_idx, pred in enumerate(predictions):
            heatmap = pred['heatmap'][b].sigmoid()  # (num_class, H, W)
            reg = pred['reg'][b]                      # (2, H, W)
            height = pred['height'][b, 0]             # (H, W)
            dim = pred['dim'][b]                      # (3, H, W)
            rot = pred['rot'][b]                      # (2, H, W)
            vel = pred['vel'][b]                      # (2, H, W)
            
            num_classes, H, W = heatmap.shape
            
            # Get top-k detections per class
            for c in range(num_classes):
                hm = heatmap[c]  # (H, W)

                # Memory-bounded candidate selection:
                # 1) top-k over the whole heatmap (no massive `where()` allocations),
                # 2) optional local-max check on those candidates.
                flat = hm.view(-1)
                k = min(int(pre_nms_topk), flat.numel())
                scores, idx = flat.topk(k)

                # Threshold
                keep_mask = scores > score_thresh
                if keep_mask.sum() == 0:
                    continue
                scores = scores[keep_mask]
                idx = idx[keep_mask]

                y_idx = torch.div(idx, W, rounding_mode='floor')
                x_idx = idx - y_idx * W

                # Local maxima check (3x3) to reduce duplicates/noise
                hm_max = F.max_pool2d(hm[None, None], 3, stride=1, padding=1).squeeze(0).squeeze(0)
                is_peak = hm[y_idx, x_idx] == hm_max[y_idx, x_idx]
                if is_peak.sum() == 0:
                    continue
                scores = scores[is_peak]
                y_idx = y_idx[is_peak]
                x_idx = x_idx[is_peak]
                
                # Decode boxes
                x = x_idx.float() * x_res + x_min + reg[0, y_idx, x_idx]
                y = y_idx.float() * y_res + y_min + reg[1, y_idx, x_idx]
                z = height[y_idx, x_idx]
                w = dim[0, y_idx, x_idx]
                l = dim[1, y_idx, x_idx]
                h = dim[2, y_idx, x_idx]
                
                sin_yaw = rot[0, y_idx, x_idx]
                cos_yaw = rot[1, y_idx, x_idx]
                yaw = torch.atan2(sin_yaw, cos_yaw)
                
                vx = vel[0, y_idx, x_idx]
                vy = vel[1, y_idx, x_idx]
                
                boxes = torch.stack([x, y, z, w, l, h, yaw], dim=1)
                labels = torch.full_like(scores, label_offset + c, dtype=torch.long)
                vels = torch.stack([vx, vy], dim=1)
                
                all_boxes.append(boxes)
                all_scores.append(scores)
                all_labels.append(labels)
                all_vels.append(vels)
            
            label_offset += num_classes
        
        if len(all_boxes) == 0:
            results.append({
                'boxes': torch.zeros(0, 7, device=device),
                'scores': torch.zeros(0, device=device),
                'labels': torch.zeros(0, dtype=torch.long, device=device),
                'velocities': torch.zeros(0, 2, device=device),
            })
            continue
        
        boxes = torch.cat(all_boxes, dim=0)
        scores = torch.cat(all_scores, dim=0)
        labels = torch.cat(all_labels, dim=0)
        vels = torch.cat(all_vels, dim=0)

        # Global cap before NMS (further protection if many classes contribute).
        if scores.numel() > pre_nms_topk:
            topk_scores, topk_idx = scores.topk(pre_nms_topk)
            scores = topk_scores
            boxes = boxes[topk_idx]
            labels = labels[topk_idx]
            vels = vels[topk_idx]
        
        # Apply NMS (class-agnostic for simplicity)
        keep = nms_bev(boxes, scores, nms_thresh)
        keep = keep[:max_dets]
        
        results.append({
            'boxes': boxes[keep],
            'scores': scores[keep],
            'labels': labels[keep],
            'velocities': vels[keep],
        })
    
    return results


def nms_bev(boxes: torch.Tensor, scores: torch.Tensor, thresh: float) -> torch.Tensor:
    """
    BEV NMS using center distance.
    
    Args:
        boxes: (N, 7) [x, y, z, w, l, h, yaw]
        scores: (N,)
        thresh: distance threshold
    """
    if len(boxes) == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)
    
    # Sort by score
    order = scores.argsort(descending=True)
    boxes = boxes[order]
    
    keep = []
    
    while len(order) > 0:
        i = order[0].item()
        keep.append(i)
        
        if len(order) == 1:
            break
        
        # Compute center distance to remaining boxes
        box_i = boxes[0]
        remaining = boxes[1:]
        
        dist = torch.sqrt(
            (box_i[0] - remaining[:, 0])**2 + 
            (box_i[1] - remaining[:, 1])**2
        )
        
        # Keep boxes that are far enough
        mask = dist > thresh
        order = order[1:][mask]
        boxes = remaining[mask]
    
    return torch.tensor(keep, dtype=torch.long, device=scores.device)
