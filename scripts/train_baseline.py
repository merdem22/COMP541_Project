"""
Training loop for multi-class object detection with bounding box regression.
Uses ground truth boxes to create Gaussian heatmap targets and box regression targets.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

# Ensure project root is on PYTHONPATH for direct script execution
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.nuscenes_dataset import NuScenesDetectionDataset, collate_nuscenes
from src.models.backbone_lidar import points_to_bev
from src.models.fusion_baseline import FusionBaselineModel
from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.utils.heatmap import batch_boxes_to_targets, NUSCENES_CLASSES


# ============== Loss Functions ==============

def focal_loss(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    alpha: float = 2.0, 
    beta: float = 4.0,
    neg_weight: float = 0.1,
) -> torch.Tensor:
    """
    Focal loss for heatmap regression (CenterNet style).
    
    With 10000 cells and ~50 objects total, negative samples dominate.
    neg_weight < 1 prevents the model from collapsing to all zeros.
    """
    pos_mask = target.eq(1).float()
    neg_mask = target.lt(1).float()
    
    pred = torch.clamp(pred, min=1e-4, max=1 - 1e-4)
    
    # Positive loss: -log(pred) * (1 - pred)^alpha
    pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_mask
    
    # Negative loss: -log(1 - pred) * pred^alpha * (1 - target)^beta
    neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * torch.pow(1 - target, beta) * neg_mask
    
    num_pos = pos_mask.sum().clamp(min=1)
    loss = -(pos_loss.sum() + neg_weight * neg_loss.sum()) / num_pos
    
    return loss


def smooth_l1_loss_with_mask(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Smooth L1 loss for box regression, only computed at positive locations.
    
    Args:
        pred: (B, 7, H, W) predicted box params
        target: (B, 7, H, W) target box params
        mask: (B, 1, H, W) binary mask where 1 = valid target
    """
    # Expand mask to match box channels
    mask = mask.expand_as(pred)  # (B, 7, H, W)
    
    # Only compute loss at masked locations
    diff = F.smooth_l1_loss(pred, target, reduction='none')
    masked_loss = diff * mask
    
    num_pos = mask[:, 0:1, :, :].sum().clamp(min=1)  # Count positive locations
    return masked_loss.sum() / (num_pos * 7)  # Normalize by num_pos and num_params


# ============== Metrics ==============

def compute_metrics(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    nms_kernel: int = 3,
    top_k: int = 50,  # Per class, per sample
    match_radius: int = 3,
) -> Dict[str, float]:
    """
    Compute detection metrics using adaptive thresholding.
    
    Instead of a fixed threshold, we use top-k selection which adapts to the model's
    current prediction distribution. This prevents both:
    - Explosion (when threshold is too low)
    - Collapse (when threshold is too high)
    """
    with torch.no_grad():
        B, C, H, W = pred.shape
        
        # NMS: keep local maxima
        pred_pooled = F.max_pool2d(pred, kernel_size=nms_kernel, stride=1, padding=nms_kernel // 2)
        is_peak = pred >= pred_pooled - 1e-6
        
        # Use adaptive threshold: mean of top 10 predictions per sample
        pred_flat = pred.view(B, C, -1)
        topk_for_thresh = torch.topk(pred_flat, k=min(10, H*W), dim=-1).values
        adaptive_thresh = topk_for_thresh.mean() * 0.5  # Half of avg top-10
        adaptive_thresh = max(adaptive_thresh.item(), 0.05)  # Minimum threshold
        
        is_peak = is_peak & (pred >= adaptive_thresh)
        gt_peaks_mask = target >= 0.99
        
        total_tp = 0
        total_pred = 0
        total_gt = 0
        
        for b in range(B):
            for c in range(C):
                pred_slice = pred[b, c]
                peak_mask = is_peak[b, c]
                gt_mask = gt_peaks_mask[b, c]
                
                # Top-K selection per class
                peak_values = pred_slice * peak_mask.float()
                peak_flat = peak_values.view(-1)
                
                k = min(top_k, (peak_flat > 0).sum().item())
                if k == 0:
                    total_gt += gt_mask.sum().item()
                    continue
                
                topk_values, topk_indices = torch.topk(peak_flat, k)
                valid_mask = topk_values >= adaptive_thresh
                topk_indices = topk_indices[valid_mask]
                num_pred_this = len(topk_indices)
                
                pred_ys = topk_indices // W
                pred_xs = topk_indices % W
                pred_coords = torch.stack([pred_ys, pred_xs], dim=1).float()
                
                gt_indices = gt_mask.nonzero(as_tuple=False)
                num_gt_this = gt_indices.shape[0]
                
                total_pred += num_pred_this
                total_gt += num_gt_this
                
                if num_pred_this == 0 or num_gt_this == 0:
                    continue
                
                # One-to-one matching
                gt_coords = gt_indices.float()
                dist = torch.cdist(pred_coords, gt_coords, p=2)
                
                matched_gt = set()
                tp_this = 0
                
                for pred_idx in range(num_pred_this):
                    distances_to_gt = dist[pred_idx]
                    for gt_idx in torch.argsort(distances_to_gt):
                        gt_idx = gt_idx.item()
                        if gt_idx in matched_gt:
                            continue
                        if distances_to_gt[gt_idx] <= match_radius:
                            tp_this += 1
                            matched_gt.add(gt_idx)
                            break
                
                total_tp += tp_this
        
        if total_pred == 0 and total_gt == 0:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "num_pred": 0, "num_gt": 0, "threshold": adaptive_thresh}
        if total_pred == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "num_pred": 0, "num_gt": int(total_gt), "threshold": adaptive_thresh}
        if total_gt == 0:
            return {"precision": 0.0, "recall": 1.0, "f1": 0.0, "num_pred": int(total_pred), "num_gt": 0, "threshold": adaptive_thresh}
        
        precision = total_tp / max(total_pred, 1)
        recall = total_tp / max(total_gt, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "num_pred": int(total_pred),
            "num_gt": int(total_gt),
            "threshold": adaptive_thresh,
        }


# ============== Data Loading ==============

def build_dataloader(cfg) -> DataLoader:
    dataset = NuScenesDetectionDataset(
        data_root=cfg.data.data_root,
        version=cfg.data.version,
        camera_channels=cfg.data.camera_channels,
        load_annotations=True,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=cfg.data.shuffle,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_nuscenes,
    )


def stack_camera_images(image_list) -> Optional[torch.Tensor]:
    tensors = []
    for images in image_list:
        if not images:
            return None
        if "CAM_FRONT" in images:
            img = images["CAM_FRONT"]
        else:
            first_key = next(iter(images))
            img = images[first_key]
        tensors.append(img)
    return torch.stack(tensors) if tensors else None


# ============== Main Training Loop ==============

def main() -> None:
    parser = argparse.ArgumentParser(description="Train multi-class detection with box regression.")
    parser.add_argument("--config", default="experiments/exp_001_baseline_mini.yaml", type=Path)
    parser.add_argument("--num-classes", default=10, type=int, help="Number of classes (10 for full, 1 for car-only)")
    parser.add_argument("--epochs", default=1, type=int, help="Number of epochs")
    parser.add_argument("--box-weight", default=1.0, type=float, help="Weight for box regression loss")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(name="train_baseline")
    device = torch.device(cfg.train.device)

    num_classes = args.num_classes
    logger.info("=" * 60)
    logger.info("Multi-class Detection Training with Box Regression")
    logger.info(f"Number of classes: {num_classes}")
    if num_classes == 10:
        logger.info(f"Classes: {NUSCENES_CLASSES}")
    logger.info(f"Graph module enabled: {cfg.model.use_graph}")
    logger.info(f"Box regression weight: {args.box_weight}")
    logger.info("=" * 60)

    dataloader = build_dataloader(cfg)
    
    model = FusionBaselineModel(
        lidar_in_channels=cfg.model.lidar_bev_channels,
        lidar_feat_channels=cfg.model.lidar_feat_channels,
        camera_feat_channels=cfg.model.camera_feat_channels,
        fusion_mode=cfg.model.fusion_mode,
        use_graph=cfg.model.use_graph,
        num_classes=num_classes,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    # BEV parameters
    x_range = (-50.0, 50.0)
    y_range = (-50.0, 50.0)
    heatmap_size = (100, 100)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        epoch_hm_loss = 0.0
        epoch_box_loss = 0.0
        epoch_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        total_gt = 0
        total_pred = 0
        step = 0

        for batch in dataloader:
            # Create input BEV from lidar points
            lidar_bevs = torch.stack([points_to_bev(p) for p in batch["lidar_points"]]).to(device)
            
            # Optional camera features
            camera_stack = stack_camera_images(batch["images"])
            camera_bev = None
            if camera_stack is not None:
                camera_stack = camera_stack.to(device)
                camera_bev = model.camera_backbone.images_to_bev(camera_stack, bev_shape=lidar_bevs.shape[-2:])

            # Create targets from GT boxes (multi-class + box regression)
            targets = batch_boxes_to_targets(
                batch["boxes"],
                num_classes=num_classes,
                heatmap_size=heatmap_size,
                x_range=x_range,
                y_range=y_range,
            )
            target_heatmap = targets["heatmap"].to(device)  # (B, num_classes, H, W)
            target_boxes = targets["box_targets"].to(device)  # (B, 7, H, W)
            target_mask = targets["box_mask"].to(device)  # (B, 1, H, W)

            # Forward pass
            outputs = model(lidar_bevs, camera_bev=camera_bev)
            pred_heatmap = outputs["heatmap"]  # (B, num_classes, H, W)
            pred_boxes = outputs["box"]  # (B, 7, H, W)

            # Losses
            hm_loss = focal_loss(pred_heatmap, target_heatmap)
            box_loss = smooth_l1_loss_with_mask(pred_boxes, target_boxes, target_mask)
            
            total_loss = hm_loss + args.box_weight * box_loss

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Compute metrics
            metrics = compute_metrics(pred_heatmap, target_heatmap)
            
            epoch_hm_loss += hm_loss.item()
            epoch_box_loss += box_loss.item()
            epoch_metrics["precision"] += metrics["precision"]
            epoch_metrics["recall"] += metrics["recall"]
            epoch_metrics["f1"] += metrics["f1"]
            total_pred += metrics["num_pred"]
            total_gt += metrics["num_gt"]
            step += 1

            if step % 5 == 0 or step == 1:
                max_pred = pred_heatmap.max().item()
                mean_pred = pred_heatmap.mean().item()
                num_pos = (target_heatmap >= 0.99).sum().item()
                
                logger.info(
                    f"[Epoch {epoch+1}] step={step} | "
                    f"hm_loss={hm_loss.item():.4f} box_loss={box_loss.item():.4f} | "
                    f"max={max_pred:.3f} mean={mean_pred:.3f} | "
                    f"P={metrics['precision']:.2f} R={metrics['recall']:.2f} F1={metrics['f1']:.2f} | "
                    f"pred={metrics['num_pred']} gt={num_pos} (thresh={metrics['threshold']:.3f})"
                )

            if step >= cfg.train.max_steps:
                break

        # Epoch summary
        avg_hm_loss = epoch_hm_loss / max(step, 1)
        avg_box_loss = epoch_box_loss / max(step, 1)
        avg_prec = epoch_metrics["precision"] / max(step, 1)
        avg_rec = epoch_metrics["recall"] / max(step, 1)
        avg_f1 = epoch_metrics["f1"] / max(step, 1)
        
        logger.info("=" * 60)
        logger.info(f"Epoch {epoch+1} Summary:")
        logger.info(f"  Heatmap Loss: {avg_hm_loss:.4f}")
        logger.info(f"  Box Loss: {avg_box_loss:.4f}")
        logger.info(f"  Precision: {avg_prec:.3f}")
        logger.info(f"  Recall: {avg_rec:.3f}")
        logger.info(f"  F1: {avg_f1:.3f}")
        logger.info(f"  Total Predictions: {total_pred}, Total GT: {total_gt}")
        logger.info("=" * 60)

    # Save trained model
    save_path = Path("outputs/checkpoints")
    save_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_path / "model_latest.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {
            "lidar_bev_channels": cfg.model.lidar_bev_channels,
            "lidar_feat_channels": cfg.model.lidar_feat_channels,
            "camera_feat_channels": cfg.model.camera_feat_channels,
            "fusion_mode": cfg.model.fusion_mode,
            "use_graph": cfg.model.use_graph,
            "num_classes": num_classes,
        }
    }, checkpoint_path)
    logger.info(f"Model saved to {checkpoint_path}")
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
