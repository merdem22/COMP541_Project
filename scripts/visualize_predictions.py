"""
Visualize predicted bounding boxes on BEV.
Shows LiDAR BEV, GT boxes, and predicted boxes.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.nuscenes_dataset import NuScenesDetectionDataset, collate_nuscenes
from src.models.backbone_lidar import points_to_bev
from src.models.fusion_baseline import FusionBaselineModel
from src.utils.heatmap import batch_boxes_to_targets, NUSCENES_CLASSES, get_class_index
from torch.utils.data import DataLoader
from src.models.learned_graph_module import LearnedGraphModule

def extract_predictions(pred_heatmap, pred_boxes, threshold=0.15, top_k=50):
    """
    Extract predicted bounding boxes from heatmap and box outputs.
    
    Returns list of dicts with: class_id, x, y, w, l, score
    """
    import torch.nn.functional as F
    
    B, C, H, W = pred_heatmap.shape
    predictions = []
    
    # NMS
    pred_pooled = F.max_pool2d(pred_heatmap, kernel_size=3, stride=1, padding=1)
    is_peak = (pred_heatmap >= pred_pooled - 1e-6) & (pred_heatmap >= threshold)
    
    for b in range(B):
        batch_preds = []
        
        for c in range(C):
            peak_mask = is_peak[b, c]
            peak_values = pred_heatmap[b, c] * peak_mask.float()
            peak_flat = peak_values.view(-1)
            
            k = min(top_k, (peak_flat > 0).sum().item())
            if k == 0:
                continue
            
            topk_scores, topk_indices = torch.topk(peak_flat, k)
            
            for score, idx in zip(topk_scores, topk_indices):
                if score < threshold:
                    continue
                
                y = (idx // W).item()
                x = (idx % W).item()
                
                # Get box parameters
                box_params = pred_boxes[b, :, y, x]
                x_off = box_params[0].item()
                y_off = box_params[1].item()
                w = box_params[3].item() * 10.0  # Denormalize
                l = box_params[4].item() * 10.0
                
                batch_preds.append({
                    'class_id': c,
                    'x': x + x_off,
                    'y': y + y_off,
                    'w': max(w, 1.0),  # Minimum size
                    'l': max(l, 1.0),
                    'score': score.item(),
                })
        
        predictions.append(batch_preds)
    
    return predictions


def extract_gt_boxes(boxes_list, num_classes=10):
    """Extract GT boxes from raw annotation format."""
    gt_boxes = []
    
    for boxes in boxes_list:
        sample_gt = []
        for box in boxes:
            class_idx = get_class_index(box["name"])
            if class_idx < 0 or class_idx >= num_classes:
                continue
            
            x, y, z = box["translation"]
            w, l, h = box["size"]
            
            # FIX: Convert to BEV pixel coordinates correctly
            # World coords: x=[-50, 50], y=[-50, 50]
            # Pixel coords: [0, 100] for 100x100 grid
            # Note: The BEV is created with x->columns, y->rows
            px = (x + 50.0) / 100.0 * 100  # x in world -> column in image
            py = (y + 50.0) / 100.0 * 100  # y in world -> row in image
            
            # Size in pixels (1 pixel = 1 meter for 100x100 grid over 100m)
            pw = w  # width in meters ≈ pixels
            pl = l  # length in meters ≈ pixels
            
            # Only add if within bounds
            if 0 <= px <= 100 and 0 <= py <= 100:
                sample_gt.append({
                    'class_id': class_idx,
                    'x': px,
                    'y': py,
                    'w': max(pw, 1),  # minimum 1 pixel
                    'l': max(pl, 1),
                    'name': box["name"],
                })
        
        gt_boxes.append(sample_gt)
    
    return gt_boxes


def visualize_sample(lidar_bev, gt_boxes, pred_boxes, sample_idx=0, save_path=None):
    """Visualize a single sample with GT and predicted boxes."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Get LiDAR BEV image
    bev_img = lidar_bev[sample_idx].cpu().numpy()
    height_map = bev_img[0]  # Height channel
    density_map = bev_img[1] if bev_img.shape[0] > 1 else bev_img[0]
    
    # FIX: Normalize and enhance the BEV image for visibility
    # The raw values are often very small, need to enhance contrast
    density_map = np.clip(density_map, 0, None)  # Remove negatives
    if density_map.max() > 0:
        # Log transform for better visibility of sparse points
        density_map = np.log1p(density_map * 100)
        density_map = density_map / (density_map.max() + 1e-6)
    
    # Also create a combined visualization
    combined = np.maximum(height_map, density_map)
    if combined.max() > 0:
        combined = combined / (combined.max() + 1e-6)
    
    # Class colors
    class_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # ========== LiDAR BEV ==========
    ax = axes[0]
    # Use a better colormap and add some background
    bg = np.ones_like(combined) * 0.1  # Dark gray background
    display_img = np.maximum(bg, combined)
    ax.imshow(display_img, cmap='hot', origin='lower', vmin=0, vmax=1)
    ax.set_title('LiDAR BEV\n(Point Density)', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # ========== GT Boxes ==========
    ax = axes[1]
    ax.imshow(display_img, cmap='hot', origin='lower', vmin=0, vmax=1)
    ax.set_title(f'Ground Truth\n({len(gt_boxes[sample_idx])} objects)', fontsize=12, fontweight='bold')
    
    for box in gt_boxes[sample_idx]:
        color = class_colors[box['class_id'] % 10]
        rect = Rectangle(
            (box['x'] - box['w']/2, box['y'] - box['l']/2),
            box['w'], box['l'],
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        # Add class label
        ax.text(box['x'], box['y'] + box['l']/2 + 2, 
                NUSCENES_CLASSES[box['class_id']][:3],
                fontsize=8, ha='center', color=color, fontweight='bold')
    
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # ========== Predicted Boxes ==========
    ax = axes[2]
    ax.imshow(display_img, cmap='hot', origin='lower', vmin=0, vmax=1)
    
    # Filter top predictions
    top_preds = sorted(pred_boxes[sample_idx], key=lambda x: x['score'], reverse=True)[:20]
    ax.set_title(f'Predictions\n(top {len(top_preds)}, thresh>0.15)', fontsize=12, fontweight='bold')
    
    for box in top_preds:
        color = class_colors[box['class_id'] % 10]
        alpha = min(1.0, box['score'] + 0.3)
        rect = Rectangle(
            (box['x'] - box['w']/2, box['y'] - box['l']/2),
            box['w'], box['l'],
            linewidth=2, edgecolor=color, facecolor=color, alpha=0.3
        )
        ax.add_patch(rect)
        # Add score
        ax.text(box['x'], box['y'], f'{box["score"]:.2f}',
                fontsize=7, ha='center', va='center', color='white', fontweight='bold')
    
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Legend
    legend_patches = [patches.Patch(color=class_colors[i], label=NUSCENES_CLASSES[i]) 
                      for i in range(min(5, len(NUSCENES_CLASSES)))]  # First 5 classes
    fig.legend(handles=legend_patches, loc='lower center', ncol=5, fontsize=9)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    plt.close()

def main():
    print("Loading model and data...")
    
    # Load dataset
    dataset = NuScenesDetectionDataset(
        data_root="data/nuscenes",
        version="v1.0-mini",
        camera_channels=[],
        load_annotations=True,
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, 
                           num_workers=0, collate_fn=collate_nuscenes)
    
    # Load model structure matching the saved checkpoint
    device = torch.device("cpu")
    model = FusionBaselineModel(
        lidar_in_channels=2,
        lidar_feat_channels=64,
        camera_feat_channels=64,
        fusion_mode="concat",
        num_classes=10,
        use_graph=False, # Initialize without graph first
    ).to(device)
    
    # Replace with Learned Graph Module (as done in training)
    # Note: args must match training
    model.graph = LearnedGraphModule(
        in_channels=128, # head_channels from FusionBaselineModel
        max_edges_per_node=8,
        initial_threshold=-1.0
    ).to(device)
    model.use_graph = True
    
    # Try to load checkpoint
    checkpoint_path = Path("outputs/checkpoints/model_latest.pt")
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print("No checkpoint found, using random weights")
    
    model.eval()
    
    # Process a few samples
    output_dir = Path("outputs/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating visualizations...")
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 3:  # Only visualize 3 batches (6 samples)
            break
        
        # Prepare inputs
        lidar_bevs = torch.stack([points_to_bev(p) for p in batch["lidar_points"]]).to(device)
        B = lidar_bevs.shape[0]
        dummy_camera = torch.zeros(B, 64, 100, 100, device=device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(lidar_bevs, camera_bev=dummy_camera)
        
        pred_heatmap = outputs["heatmap"]
        pred_boxes_tensor = outputs["box"]
        
        # Extract predictions and GT
        predictions = extract_predictions(pred_heatmap, pred_boxes_tensor, threshold=0.15)
        gt_boxes = extract_gt_boxes(batch["boxes"])
        
        # Visualize each sample in batch
        for i in range(B):
            sample_num = batch_idx * B + i
            save_path = output_dir / f"sample_{sample_num:03d}.png"
            
            # Create single-sample lists
            single_pred = [predictions[i]]
            single_gt = [gt_boxes[i]]
            single_bev = lidar_bevs[i:i+1]
            
            visualize_sample(single_bev, single_gt, single_pred, 
                           sample_idx=0, save_path=save_path)
    
    print(f"\n✓ Visualizations saved to {output_dir}/")
    print("Files: sample_000.png, sample_001.png, ...")


if __name__ == "__main__":
    main()

