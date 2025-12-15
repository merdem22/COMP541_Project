"""
Training loop for the fusion baseline with real car detection targets.
Uses ground truth boxes to create Gaussian heatmap targets.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

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
from src.utils.heatmap import batch_boxes_to_heatmap


# ============== Loss Functions ==============

def focal_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """
    Focal loss for heatmap regression (CenterNet style).
    
    Args:
        pred: Predicted heatmap (B, C, H, W), values in [0, 1]
        target: Ground truth heatmap (B, C, H, W), values in [0, 1]
        alpha: Focal weight for positive samples
        beta: Focal weight for negative samples
    
    Returns:
        Scalar loss value
    """
    pos_mask = target.eq(1).float()
    neg_mask = target.lt(1).float()
    
    # Clamp predictions to avoid log(0)
    pred = torch.clamp(pred, min=1e-4, max=1 - 1e-4)
    
    # Positive loss: -log(pred) * (1 - pred)^alpha
    pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_mask
    
    # Negative loss: -log(1 - pred) * pred^alpha * (1 - target)^beta
    neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * torch.pow(1 - target, beta) * neg_mask
    
    num_pos = pos_mask.sum().clamp(min=1)
    loss = -(pos_loss.sum() + neg_loss.sum()) / num_pos
    
    return loss


# ============== Metrics ==============

def compute_metrics(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    threshold: float = 0.3,
    nms_kernel: int = 3,
    top_k: int = 100,
    match_radius: int = 3,
) -> Tuple[float, float, float, int, int]:
    """
    Compute precision, recall, and F1 for heatmap peak detection.
    Uses CenterNet-style peak extraction with top-k selection and one-to-one GT matching.
    
    Args:
        pred: Predicted heatmap (B, C, H, W)
        target: Ground truth heatmap (B, C, H, W)
        threshold: Detection threshold
        nms_kernel: Kernel size for local maximum detection (NMS)
        top_k: Maximum number of detections per sample
        match_radius: Radius in pixels for matching predictions to GT
    
    Returns:
        (precision, recall, f1, num_pred, num_gt)
    """
    with torch.no_grad():
        B, C, H, W = pred.shape
        
        # ===== NMS Peak Extraction =====
        # NOTE: When heatmap is flat/uniform (e.g., untrained model outputs ~0.5 everywhere),
        # pred == maxpool(pred) is True for EVERY pixel, causing explosion in detection count.
        # This is why we need:
        #   1. CenterNet bias init (-2.19) to start with low predictions (~0.1)
        #   2. Top-k selection to cap the number of detections
        pred_pooled = F.max_pool2d(pred, kernel_size=nms_kernel, stride=1, padding=nms_kernel // 2)
        is_peak = (pred == pred_pooled) & (pred >= threshold)
        
        # ===== Extract GT peak locations =====
        gt_peaks_mask = target >= 0.99
        
        total_tp = 0
        total_pred = 0
        total_gt = 0
        
        # Process each sample in the batch
        for b in range(B):
            for c in range(C):
                pred_slice = pred[b, c]  # (H, W)
                peak_mask = is_peak[b, c]  # (H, W)
                gt_mask = gt_peaks_mask[b, c]  # (H, W)
                
                # ===== Top-K Selection =====
                # Get peak values and apply top-k
                peak_values = pred_slice * peak_mask.float()  # Zero out non-peaks
                peak_flat = peak_values.view(-1)  # (H*W,)
                
                # Select top-k peaks
                k = min(top_k, (peak_flat > 0).sum().item())
                if k == 0:
                    # No predictions for this sample/class
                    total_gt += gt_mask.sum().item()
                    continue
                
                topk_values, topk_indices = torch.topk(peak_flat, k)
                
                # Filter by threshold (topk might include zeros if fewer than k peaks)
                valid_mask = topk_values >= threshold
                topk_indices = topk_indices[valid_mask]
                num_pred_this = len(topk_indices)
                
                # Convert flat indices to (y, x) coordinates
                pred_ys = topk_indices // W
                pred_xs = topk_indices % W
                pred_coords = torch.stack([pred_ys, pred_xs], dim=1).float()  # (num_pred, 2)
                
                # ===== Extract GT coordinates =====
                gt_indices = gt_mask.nonzero(as_tuple=False)  # (num_gt, 2) as (y, x)
                num_gt_this = gt_indices.shape[0]
                
                total_pred += num_pred_this
                total_gt += num_gt_this
                
                if num_pred_this == 0 or num_gt_this == 0:
                    continue
                
                # ===== One-to-One Matching =====
                # Compute pairwise distances between predictions and GT
                # pred_coords: (num_pred, 2), gt_indices: (num_gt, 2)
                gt_coords = gt_indices.float()
                
                # Pairwise L2 distance: (num_pred, num_gt)
                dist = torch.cdist(pred_coords, gt_coords, p=2)
                
                # Greedy matching: each GT can be matched at most once
                matched_gt = set()
                tp_this = 0
                
                # Sort predictions by confidence (already sorted by topk)
                for pred_idx in range(num_pred_this):
                    distances_to_gt = dist[pred_idx]  # (num_gt,)
                    
                    # Find closest unmatched GT within radius
                    for gt_idx in torch.argsort(distances_to_gt):
                        gt_idx = gt_idx.item()
                        if gt_idx in matched_gt:
                            continue
                        if distances_to_gt[gt_idx] <= match_radius:
                            tp_this += 1
                            matched_gt.add(gt_idx)
                            break
                
                total_tp += tp_this
        
        # ===== Compute Metrics =====
        if total_pred == 0 and total_gt == 0:
            return 1.0, 1.0, 1.0, 0, 0
        if total_pred == 0:
            return 0.0, 0.0, 0.0, 0, int(total_gt)
        if total_gt == 0:
            return 0.0, 1.0, 0.0, int(total_pred), 0
        
        precision = total_tp / max(total_pred, 1)
        recall = total_tp / max(total_gt, 1)  # Recall <= 1 due to one-to-one matching
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)
        
        return precision, recall, f1, int(total_pred), int(total_gt)


# ============== Data Loading ==============

def build_dataloader(cfg) -> DataLoader:
    dataset = NuScenesDetectionDataset(
        data_root=cfg.data.data_root,
        version=cfg.data.version,
        camera_channels=cfg.data.camera_channels,
        load_annotations=True,  # Enable GT loading
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
    parser = argparse.ArgumentParser(description="Train fusion baseline for car detection.")
    parser.add_argument("--config", default="experiments/exp_001_baseline_mini.yaml", type=Path)
    parser.add_argument("--target-class", default="car", type=str, help="Class to detect")
    parser.add_argument("--epochs", default=1, type=int, help="Number of epochs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(name="train_baseline")
    device = torch.device(cfg.train.device)

    logger.info("=" * 60)
    logger.info("Training car detection with StaticGraphModule")
    logger.info(f"Target class: {args.target_class}")
    logger.info(f"Graph module enabled: {cfg.model.use_graph}")
    logger.info("=" * 60)

    dataloader = build_dataloader(cfg)
    
    # Model outputs 10 classes, but we only use class 0 for cars
    model = FusionBaselineModel(
        lidar_in_channels=cfg.model.lidar_bev_channels,
        lidar_feat_channels=cfg.model.lidar_feat_channels,
        camera_feat_channels=cfg.model.camera_feat_channels,
        fusion_mode=cfg.model.fusion_mode,
        use_graph=cfg.model.use_graph,
        num_classes=1,  # Single class: car
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    
    # BEV parameters (must match backbone_lidar.py)
    x_range = (-50.0, 50.0)
    y_range = (-50.0, 50.0)
    # The backbone has stride 2, so output is 100x100 for 200x200 input
    heatmap_size = (100, 100)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_prec = 0.0
        epoch_rec = 0.0
        epoch_f1 = 0.0
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

            # Create target heatmap from GT boxes
            target_heatmap = batch_boxes_to_heatmap(
                batch["boxes"],
                target_class=args.target_class,
                heatmap_size=heatmap_size,
                x_range=x_range,
                y_range=y_range,
            ).to(device)  # (B, 1, H, W)

            # Forward pass
            outputs = model(lidar_bevs, camera_bev=camera_bev)
            pred_heatmap = outputs["heatmap"]  # (B, 1, H, W)

            # Compute focal loss
            loss = focal_loss(pred_heatmap, target_heatmap)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute metrics
            # Use lower threshold (0.1) for monitoring since CenterNet init starts predictions at ~0.1
            prec, rec, f1, n_pred, n_gt = compute_metrics(pred_heatmap, target_heatmap, threshold=0.1)
            
            epoch_loss += loss.item()
            epoch_prec += prec
            epoch_rec += rec
            epoch_f1 += f1
            total_pred += n_pred
            total_gt += n_gt
            step += 1

            if step % 5 == 0 or step == 1:
                logger.info(
                    f"[Epoch {epoch+1}] step={step} loss={loss.item():.4f} "
                    f"prec={prec:.3f} rec={rec:.3f} f1={f1:.3f} "
                    f"pred={n_pred} gt={n_gt}"
                )

            if step >= cfg.train.max_steps:
                break

        # Epoch summary
        avg_loss = epoch_loss / max(step, 1)
        avg_prec = epoch_prec / max(step, 1)
        avg_rec = epoch_rec / max(step, 1)
        avg_f1 = epoch_f1 / max(step, 1)
        
        logger.info("=" * 60)
        logger.info(f"Epoch {epoch+1} Summary:")
        logger.info(f"  Avg Loss: {avg_loss:.4f}")
        logger.info(f"  Avg Precision: {avg_prec:.3f}")
        logger.info(f"  Avg Recall: {avg_rec:.3f}")
        logger.info(f"  Avg F1: {avg_f1:.3f}")
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
        }
    }, checkpoint_path)
    logger.info(f"Model saved to {checkpoint_path}")
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
