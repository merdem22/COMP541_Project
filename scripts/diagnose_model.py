"""
Diagnostic script to understand model prediction behavior.
Analyzes heatmap statistics, peak distributions, and visualizes problematic samples.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.nuscenes_dataset import NuScenesDetectionDataset, collate_nuscenes
from src.models.backbone_lidar import points_to_bev
from src.models.fusion_baseline import FusionBaselineModel
from src.utils.config import load_config
from src.utils.heatmap import batch_boxes_to_heatmap


def analyze_heatmap_statistics(pred_heatmap: torch.Tensor, target_heatmap: torch.Tensor):
    """Compute detailed statistics about heatmap predictions"""
    with torch.no_grad():
        stats = {
            'pred_min': pred_heatmap.min().item(),
            'pred_max': pred_heatmap.max().item(),
            'pred_mean': pred_heatmap.mean().item(),
            'pred_std': pred_heatmap.std().item(),
            'pred_median': pred_heatmap.median().item(),
            'target_mean': target_heatmap.mean().item(),
            'target_peaks': (target_heatmap >= 0.99).sum().item(),
        }
        
        # Count peaks at different thresholds
        for thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
            nms_kernel = 3
            pred_pooled = F.max_pool2d(pred_heatmap, kernel_size=nms_kernel, stride=1, padding=nms_kernel // 2)
            is_peak = (pred_heatmap == pred_pooled) & (pred_heatmap >= thresh)
            stats[f'peaks_at_{thresh}'] = is_peak.sum().item()
        
        # Histogram of prediction values
        pred_np = pred_heatmap.cpu().numpy().flatten()
        hist, bins = np.histogram(pred_np, bins=50, range=(0, 1))
        stats['histogram'] = (hist.tolist(), bins.tolist())
        
        return stats


def visualize_problematic_sample(
    lidar_bev: torch.Tensor,
    pred_heatmap: torch.Tensor,
    target_heatmap: torch.Tensor,
    stats: dict,
    output_path: Path,
):
    """Create detailed visualization of a problematic prediction"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Row 1: LiDAR input
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(lidar_bev[0].cpu().numpy(), cmap='viridis', origin='lower')
    ax1.set_title('LiDAR BEV - Height')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(lidar_bev[1].cpu().numpy(), cmap='hot', origin='lower')
    ax2.set_title('LiDAR BEV - Density')
    
    # Histogram
    ax3 = fig.add_subplot(gs[0, 2])
    hist, bins = stats['histogram']
    ax3.bar(bins[:-1], hist, width=np.diff(bins), align='edge', alpha=0.7)
    ax3.axvline(0.3, color='r', linestyle='--', label='Threshold=0.3')
    ax3.set_xlabel('Prediction Value')
    ax3.set_ylabel('Count')
    ax3.set_title('Prediction Distribution')
    ax3.legend()
    ax3.set_yscale('log')
    
    # Row 2: Heatmaps
    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(target_heatmap[0, 0].cpu().numpy(), cmap='Reds', origin='lower', vmin=0, vmax=1)
    ax4.set_title(f'Target ({stats["target_peaks"]} cars)')
    plt.colorbar(im4, ax=ax4, fraction=0.046)
    
    ax5 = fig.add_subplot(gs[1, 1])
    im5 = ax5.imshow(pred_heatmap[0, 0].cpu().numpy(), cmap='Reds', origin='lower', vmin=0, vmax=1)
    ax5.set_title(f'Prediction (max={stats["pred_max"]:.3f})')
    plt.colorbar(im5, ax=ax5, fraction=0.046)
    
    # Difference map
    ax6 = fig.add_subplot(gs[1, 2])
    diff = (pred_heatmap[0, 0] - target_heatmap[0, 0]).cpu().numpy()
    im6 = ax6.imshow(diff, cmap='RdBu_r', origin='lower', vmin=-1, vmax=1)
    ax6.set_title('Difference (Pred - Target)')
    plt.colorbar(im6, ax=ax6, fraction=0.046)
    
    # Row 3: Peak detection at different thresholds
    thresholds = [0.1, 0.3, 0.5]
    for idx, thresh in enumerate(thresholds):
        ax = fig.add_subplot(gs[2, idx])
        pred_pooled = F.max_pool2d(pred_heatmap, kernel_size=3, stride=1, padding=1)
        is_peak = (pred_heatmap == pred_pooled) & (pred_heatmap >= thresh)
        peak_img = is_peak[0, 0].cpu().numpy().astype(float)
        
        # Overlay on heatmap
        base = pred_heatmap[0, 0].cpu().numpy()
        ax.imshow(base, cmap='gray', origin='lower', alpha=0.5)
        ax.imshow(peak_img, cmap='Reds', origin='lower', alpha=0.5)
        
        num_peaks = stats[f'peaks_at_{thresh}']
        ax.set_title(f'Peaks @ {thresh} ({num_peaks} peaks)')
    
    # Overall statistics text
    stats_text = (
        f"Statistics:\n"
        f"Mean: {stats['pred_mean']:.4f}\n"
        f"Std: {stats['pred_std']:.4f}\n"
        f"Max: {stats['pred_max']:.4f}\n"
        f"Median: {stats['pred_median']:.4f}\n"
    )
    fig.text(0.02, 0.02, stats_text, fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Model Prediction Diagnosis', fontsize=14, fontweight='bold')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Diagnose model prediction behavior")
    parser.add_argument("--config", default="experiments/exp_001_baseline_mini.yaml", type=Path)
    parser.add_argument("--checkpoint", default="outputs/checkpoints/model_latest.pt", type=Path)
    parser.add_argument("--num-samples", default=5, type=int)
    parser.add_argument("--output-dir", default="outputs/diagnostics", type=Path)
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cfg = load_config(args.config)
    device = torch.device(cfg.train.device)
    
    print("Loading dataset...")
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
    
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        print(f"Loading weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print(f"No checkpoint found, using random weights")
    
    model.eval()
    
    x_range = (-50.0, 50.0)
    y_range = (-50.0, 50.0)
    heatmap_size = (100, 100)
    
    print(f"\nAnalyzing {args.num_samples} samples...\n")
    
    all_stats = []
    
    for idx in range(min(args.num_samples, len(dataset))):
        sample = dataset[idx]
        
        lidar_bev = points_to_bev(sample["lidar_points"]).to(device)
        
        gt_heatmap = batch_boxes_to_heatmap(
            [sample["boxes"]],
            target_class="car",
            heatmap_size=heatmap_size,
            x_range=x_range,
            y_range=y_range,
        ).to(device)
        
        with torch.no_grad():
            lidar_bev_batch = lidar_bev.unsqueeze(0)
            dummy_camera_bev = torch.zeros(1, 64, 100, 100, device=device)
            outputs = model(lidar_bev_batch, camera_bev=dummy_camera_bev)
            pred_heatmap = outputs["heatmap"]
        
        stats = analyze_heatmap_statistics(pred_heatmap, gt_heatmap)
        all_stats.append(stats)
        
        # Visualize
        output_path = output_dir / f"diagnosis_{idx:03d}.png"
        visualize_problematic_sample(lidar_bev, pred_heatmap, gt_heatmap, stats, output_path)
        
        print(f"Sample {idx}:")
        print(f"  GT cars: {stats['target_peaks']}")
        print(f"  Pred mean: {stats['pred_mean']:.4f}, max: {stats['pred_max']:.4f}")
        print(f"  Peaks @ 0.1: {stats['peaks_at_0.1']}, @ 0.3: {stats['peaks_at_0.3']}, @ 0.5: {stats['peaks_at_0.5']}")
        print(f"  Saved: {output_path}\n")
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY ACROSS ALL SAMPLES")
    print("="*60)
    
    avg_stats = {
        'pred_mean': np.mean([s['pred_mean'] for s in all_stats]),
        'pred_std': np.mean([s['pred_std'] for s in all_stats]),
        'pred_max': np.mean([s['pred_max'] for s in all_stats]),
        'peaks_at_0.1': np.mean([s['peaks_at_0.1'] for s in all_stats]),
        'peaks_at_0.3': np.mean([s['peaks_at_0.3'] for s in all_stats]),
        'target_peaks': np.mean([s['target_peaks'] for s in all_stats]),
    }
    
    print(f"\nAverage prediction mean: {avg_stats['pred_mean']:.4f}")
    print(f"Average prediction std:  {avg_stats['pred_std']:.4f}")
    print(f"Average prediction max:  {avg_stats['pred_max']:.4f}")
    print(f"\nAverage peaks @ 0.1: {avg_stats['peaks_at_0.1']:.1f}")
    print(f"Average peaks @ 0.3: {avg_stats['peaks_at_0.3']:.1f}")
    print(f"Average GT peaks:    {avg_stats['target_peaks']:.1f}")
    
    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)
    
    if avg_stats['pred_max'] < 0.3:
        print("\n⚠️  PROBLEM: Model predictions are too LOW")
        print("   - Average max prediction is only {:.3f}".format(avg_stats['pred_max']))
        print("   - Model is not confident enough")
        print("   FIXES:")
        print("   1. Train longer (more epochs)")
        print("   2. Use higher learning rate initially")
        print("   3. Check if graph module is suppressing activations")
    
    if avg_stats['peaks_at_0.1'] > avg_stats['target_peaks'] * 5:
        print("\n⚠️  PROBLEM: Too many false peaks at low threshold")
        print(f"   - {avg_stats['peaks_at_0.1']:.0f} peaks @ 0.1 vs {avg_stats['target_peaks']:.0f} GT")
        print("   - Heatmap is too flat/uniform")
        print("   FIXES:")
        print("   1. ✓ Already using threshold=0.3 in training (good!)")
        print("   2. Ensure CenterNet bias init is working (-2.19)")
        print("   3. Train longer to learn sharper peaks")
    
    if avg_stats['pred_std'] < 0.05:
        print("\n⚠️  PROBLEM: Predictions are too uniform (low variance)")
        print(f"   - Std = {avg_stats['pred_std']:.4f}")
        print("   - Model hasn't learned to distinguish object centers")
        print("   FIXES:")
        print("   1. More training iterations")
        print("   2. Check if graph smoothing is over-homogenizing features")
    
    print(f"\nDiagnostic images saved to: {output_dir}/")


if __name__ == "__main__":
    main()