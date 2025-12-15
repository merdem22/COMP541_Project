"""
Visualize ground truth and predicted heatmaps for car detection.
Saves images to outputs/heatmaps/ directory.
"""

import argparse
import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is on PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.nuscenes_dataset import NuScenesDetectionDataset, collate_nuscenes
from src.models.backbone_lidar import points_to_bev
from src.models.fusion_baseline import FusionBaselineModel
from src.utils.config import load_config
from src.utils.heatmap import batch_boxes_to_heatmap


def visualize_sample(
    lidar_bev: torch.Tensor,
    gt_heatmap: torch.Tensor,
    pred_heatmap: torch.Tensor,
    boxes: list,
    sample_idx: int,
    output_dir: Path,
    target_class: str = "car",
):
    """
    Create a visualization figure with 4 subplots:
    1. LiDAR BEV (height channel)
    2. LiDAR BEV (density channel)
    3. Ground Truth Heatmap
    4. Predicted Heatmap
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # LiDAR BEV - Height channel
    ax1 = axes[0, 0]
    height_map = lidar_bev[0].cpu().numpy()
    im1 = ax1.imshow(height_map, cmap='viridis', origin='lower')
    ax1.set_title('LiDAR BEV - Height', fontsize=12)
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    # LiDAR BEV - Density channel
    ax2 = axes[0, 1]
    density_map = lidar_bev[1].cpu().numpy()
    im2 = ax2.imshow(density_map, cmap='hot', origin='lower')
    ax2.set_title('LiDAR BEV - Point Density', fontsize=12)
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    # Ground Truth Heatmap
    ax3 = axes[1, 0]
    gt_map = gt_heatmap[0].cpu().numpy()
    im3 = ax3.imshow(gt_map, cmap='Reds', origin='lower', vmin=0, vmax=1)
    ax3.set_title(f'Ground Truth Heatmap ({target_class})', fontsize=12)
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    # Count cars in this sample
    num_cars = sum(1 for b in boxes if target_class.lower() in b["name"].lower())
    ax3.text(0.02, 0.98, f'{num_cars} cars', transform=ax3.transAxes, 
             fontsize=10, verticalalignment='top', color='white',
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    # Predicted Heatmap
    ax4 = axes[1, 1]
    pred_map = pred_heatmap[0].cpu().numpy()
    im4 = ax4.imshow(pred_map, cmap='Reds', origin='lower', vmin=0, vmax=1)
    ax4.set_title('Predicted Heatmap', fontsize=12)
    plt.colorbar(im4, ax=ax4, fraction=0.046)
    
    # Add max prediction value
    max_pred = pred_map.max()
    ax4.text(0.02, 0.98, f'max: {max_pred:.3f}', transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', color='white',
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    plt.suptitle(f'Sample {sample_idx}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / f'sample_{sample_idx:03d}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Visualize heatmaps for car detection")
    parser.add_argument("--config", default="experiments/exp_001_baseline_mini.yaml", type=Path)
    parser.add_argument("--checkpoint", default="outputs/checkpoints/model_latest.pt", type=Path,
                        help="Path to trained model checkpoint")
    parser.add_argument("--target-class", default="car", type=str)
    parser.add_argument("--num-samples", default=10, type=int, help="Number of samples to visualize")
    parser.add_argument("--output-dir", default="outputs/heatmaps", type=Path)
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    cfg = load_config(args.config)
    device = torch.device(cfg.train.device)
    
    print(f"Loading dataset...")
    dataset = NuScenesDetectionDataset(
        data_root=cfg.data.data_root,
        version=cfg.data.version,
        camera_channels=cfg.data.camera_channels,
        load_annotations=True,
    )
    
    print(f"Loading model (use_graph={cfg.model.use_graph})...")
    model = FusionBaselineModel(
        lidar_in_channels=cfg.model.lidar_bev_channels,
        lidar_feat_channels=cfg.model.lidar_feat_channels,
        camera_feat_channels=cfg.model.camera_feat_channels,
        fusion_mode=cfg.model.fusion_mode,
        use_graph=cfg.model.use_graph,
        num_classes=1,
    ).to(device)
    
    # Load trained weights if checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        print(f"Loading trained weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("  Loaded successfully!")
    else:
        print(f"  No checkpoint found at {checkpoint_path}, using random weights")
    
    model.eval()
    
    # BEV parameters
    x_range = (-50.0, 50.0)
    y_range = (-50.0, 50.0)
    heatmap_size = (100, 100)
    
    print(f"Visualizing {args.num_samples} samples...")
    
    for idx in range(min(args.num_samples, len(dataset))):
        sample = dataset[idx]
        
        # Create LiDAR BEV
        lidar_bev = points_to_bev(sample["lidar_points"]).to(device)  # (2, H, W)
        
        # Create GT heatmap
        gt_heatmap = batch_boxes_to_heatmap(
            [sample["boxes"]],
            target_class=args.target_class,
            heatmap_size=heatmap_size,
            x_range=x_range,
            y_range=y_range,
        ).to(device)  # (1, 1, H, W)
        
        # Get model prediction
        with torch.no_grad():
            lidar_bev_batch = lidar_bev.unsqueeze(0)  # (1, 2, H, W)
            # Create dummy camera BEV (zeros) to match expected input channels
            # The backbone outputs 100x100 (stride 2 on 200x200 input)
            dummy_camera_bev = torch.zeros(1, 64, 100, 100, device=device)
            outputs = model(lidar_bev_batch, camera_bev=dummy_camera_bev)
            pred_heatmap = outputs["heatmap"]  # (1, 1, H, W)
        
        # Visualize
        output_path = visualize_sample(
            lidar_bev=lidar_bev,
            gt_heatmap=gt_heatmap[0],  # Remove batch dim
            pred_heatmap=pred_heatmap[0],  # Remove batch dim
            boxes=sample["boxes"],
            sample_idx=idx,
            output_dir=output_dir,
            target_class=args.target_class,
        )
        
        print(f"  Saved: {output_path}")
    
    print(f"\nDone! Visualizations saved to {output_dir}/")
    print(f"Open with: open {output_dir}")


if __name__ == "__main__":
    main()

