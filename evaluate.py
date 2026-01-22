#!/usr/bin/env python3
"""
Evaluation Script with nuScenes Official Metrics
Computes mAP and NDS for 3D object detection
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List
import yaml
import numpy as np
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.data import NuScenesDataset, collate_fn, CLASS_NAMES
from src.models import build_model


def format_nuscenes_results(
    predictions: List[Dict],
    sample_tokens: List[str],
    config: Dict,
) -> Dict:
    """
    Format predictions for nuScenes evaluation.
    
    Returns dict: {sample_token: [detection_dict, ...]}
    """
    results = {}
    
    for pred, token in zip(predictions, sample_tokens):
        detections = []
        
        boxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        velocities = pred['velocities'].cpu().numpy()
        
        for i in range(len(boxes)):
            det = {
                'sample_token': token,
                'translation': boxes[i, :3].tolist(),
                'size': boxes[i, 3:6].tolist(),
                'rotation': _yaw_to_quaternion(boxes[i, 6]),
                'velocity': velocities[i].tolist(),
                'detection_name': CLASS_NAMES[labels[i]],
                'detection_score': float(scores[i]),
                'attribute_name': '',  # Not predicting attributes
            }
            detections.append(det)
        
        results[token] = detections
    
    return results


def _yaw_to_quaternion(yaw: float) -> List[float]:
    """Convert yaw angle to quaternion [w, x, y, z]."""
    w = np.cos(yaw / 2)
    x = 0.0
    y = 0.0
    z = np.sin(yaw / 2)
    return [w, x, y, z]


def compute_nuscenes_metrics(
    results: Dict,
    nusc,
    eval_set: str = 'val',
    output_dir: str = 'outputs/eval',
) -> Dict:
    """
    Compute official nuScenes metrics.
    
    Returns:
        metrics: Dict with mAP, NDS, and per-class APs
    """
    try:
        from nuscenes.eval.detection.config import config_factory
        from nuscenes.eval.detection.evaluate import DetectionEval
    except ImportError:
        print("nuScenes devkit not available for official eval")
        return {}
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results to JSON
    results_file = os.path.join(output_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump({'meta': {'use_camera': True, 'use_lidar': True}, 'results': results}, f)
    
    # Run official evaluation
    cfg = config_factory('detection_cvpr_2019')
    
    nusc_eval = DetectionEval(
        nusc,
        config=cfg,
        result_path=results_file,
        eval_set=eval_set,
        output_dir=output_dir,
        verbose=True,
    )
    
    metrics = nusc_eval.main(render_curves=False)
    
    return metrics


def simple_metrics(
    predictions: List[Dict],
    annotations: List[Dict],
    distance_thresholds: List[float] = [0.5, 1.0, 2.0, 4.0],
) -> Dict:
    """
    Compute simple evaluation metrics without nuScenes devkit.
    Uses center distance matching.
    """
    per_class_stats = {name: {'tp': [], 'fp': [], 'fn': 0, 'scores': []} 
                       for name in CLASS_NAMES}
    
    for pred, ann in zip(predictions, annotations):
        pred_boxes = pred['boxes'].cpu()
        pred_scores = pred['scores'].cpu()
        pred_labels = pred['labels'].cpu()
        
        gt_boxes = ann['boxes'].cpu()
        gt_labels = ann['labels'].cpu()
        
        # Process each class
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            pred_mask = pred_labels == cls_idx
            gt_mask = gt_labels == cls_idx
            
            cls_pred_boxes = pred_boxes[pred_mask]
            cls_pred_scores = pred_scores[pred_mask]
            cls_gt_boxes = gt_boxes[gt_mask]
            
            if len(cls_gt_boxes) == 0:
                # All predictions are false positives
                per_class_stats[cls_name]['tp'].extend([0] * len(cls_pred_boxes))
                per_class_stats[cls_name]['fp'].extend([1] * len(cls_pred_boxes))
                per_class_stats[cls_name]['scores'].extend(cls_pred_scores.tolist())
                continue
            
            if len(cls_pred_boxes) == 0:
                per_class_stats[cls_name]['fn'] += len(cls_gt_boxes)
                continue
            
            # Match predictions to ground truth
            matched_gt = set()
            
            # Sort by score
            order = cls_pred_scores.argsort(descending=True)
            
            for idx in order:
                p_box = cls_pred_boxes[idx]
                p_score = cls_pred_scores[idx].item()
                
                # Find closest GT
                distances = torch.sqrt(
                    (cls_gt_boxes[:, 0] - p_box[0])**2 +
                    (cls_gt_boxes[:, 1] - p_box[1])**2
                )
                
                min_dist, min_idx = distances.min(), distances.argmin().item()
                
                # Match with 2m threshold
                if min_dist < 2.0 and min_idx not in matched_gt:
                    per_class_stats[cls_name]['tp'].append(1)
                    per_class_stats[cls_name]['fp'].append(0)
                    matched_gt.add(min_idx)
                else:
                    per_class_stats[cls_name]['tp'].append(0)
                    per_class_stats[cls_name]['fp'].append(1)
                
                per_class_stats[cls_name]['scores'].append(p_score)
            
            per_class_stats[cls_name]['fn'] += len(cls_gt_boxes) - len(matched_gt)
    
    # Compute AP per class
    aps = {}
    for cls_name, stats in per_class_stats.items():
        if len(stats['scores']) == 0:
            aps[cls_name] = 0.0
            continue
        
        scores = np.array(stats['scores'])
        tp = np.array(stats['tp'])
        fp = np.array(stats['fp'])
        
        # Sort by score
        order = np.argsort(-scores)
        tp = tp[order]
        fp = fp[order]
        
        # Cumulative sums
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # Total positives
        total_pos = tp_cumsum[-1] + stats['fn'] if len(tp_cumsum) > 0 else stats['fn']
        
        if total_pos == 0:
            aps[cls_name] = 0.0
            continue
        
        # Precision and recall
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        recall = tp_cumsum / (total_pos + 1e-6)
        
        # AP computation (11-point interpolation)
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            mask = recall >= t
            if mask.sum() > 0:
                ap += precision[mask].max() / 11
        
        aps[cls_name] = ap
    
    # Mean AP
    mAP = np.mean(list(aps.values()))
    
    return {
        'mAP': mAP,
        'per_class_AP': aps,
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    config: Dict,
    device: torch.device,
    use_official: bool = False,
) -> Dict:
    """Run evaluation."""
    model.eval()
    
    all_predictions = []
    all_annotations = []
    all_tokens = []
    
    print("Running inference...")
    for batch in tqdm(dataloader):
        points = batch['points'].to(device, non_blocking=True)
        points_mask = batch['points_mask'].to(device, non_blocking=True)
        images = batch['images'].to(device, non_blocking=True)
        cam_intrinsics = batch['cam_intrinsics'].to(device, non_blocking=True)
        cam_extrinsics = batch['cam_extrinsics'].to(device, non_blocking=True)

        amp_ctx = autocast('cuda') if device.type == 'cuda' else nullcontext()
        with amp_ctx:
            predictions = model(
                points, points_mask, images,
                cam_intrinsics, cam_extrinsics
            )
            
            if hasattr(model, 'module'):
                decoded = model.module.predict(
                    predictions,
                    score_thresh=config['eval']['score_thresh'],
                    nms_thresh=config['eval']['nms_thresh'],
                    max_dets=config['eval']['max_dets'],
                )
            else:
                decoded = model.predict(
                    predictions,
                    score_thresh=config['eval']['score_thresh'],
                    nms_thresh=config['eval']['nms_thresh'],
                    max_dets=config['eval']['max_dets'],
                )
        
        all_predictions.extend(decoded)
        all_annotations.extend(batch['annotations'])
        all_tokens.extend(batch['sample_tokens'])
    
    print("Computing metrics...")
    
    # Simple metrics
    metrics = simple_metrics(all_predictions, all_annotations)
    
    print(f"\n{'='*50}")
    print(f"Evaluation Results")
    print(f"{'='*50}")
    print(f"mAP: {metrics['mAP']:.4f}")
    print(f"\nPer-class AP:")
    for cls_name, ap in metrics['per_class_AP'].items():
        print(f"  {cls_name}: {ap:.4f}")
    
    # Official nuScenes metrics (if available)
    if use_official:
        try:
            from nuscenes.nuscenes import NuScenes
            
            nusc = NuScenes(
                version=config['data']['version'],
                dataroot=config['data']['root'],
                verbose=False
            )
            
            results = format_nuscenes_results(all_predictions, all_tokens, config)
            official_metrics = compute_nuscenes_metrics(
                results, nusc, 
                eval_set=config['data']['val_split']
            )
            
            if official_metrics:
                metrics['official'] = official_metrics
                print(f"\nOfficial nuScenes Metrics:")
                print(f"  NDS: {official_metrics.get('nd_score', 0):.4f}")
                print(f"  mAP: {official_metrics.get('mean_ap', 0):.4f}")
                
        except Exception as e:
            print(f"Could not compute official metrics: {e}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/exp1_lidar.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='outputs/eval')
    parser.add_argument('--lite', action='store_true')
    parser.add_argument('--no-graph', action='store_true', help='Disable graph module (ablation)')
    parser.add_argument('--no-camera', action='store_true', help='Disable camera branch (LiDAR-only)')
    parser.add_argument('--official', action='store_true', help='Use official nuScenes eval')
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Override graph/camera settings if flags are set
    if args.no_graph:
        config['model']['use_graph'] = False
    else:
        config['model']['use_graph'] = config['model'].get('use_graph', True)

    if args.no_camera:
        config['model']['use_camera'] = False
    else:
        config['model']['use_camera'] = config['model'].get('use_camera', True)

    # Adjust LiDAR input channels for time-lag feature
    data_cfg = config.get('data', {})
    if data_cfg.get('use_time_lag', False) and data_cfg.get('num_sweeps', 1) > 1:
        lidar_in = config['model']['lidar'].get('in_channels', 5)
        if lidar_in == 5:
            config['model']['lidar']['in_channels'] = 6
    if 'use_camera' not in data_cfg:
        data_cfg['use_camera'] = config['model'].get('use_camera', True)
    if config['model'].get('use_camera', True) != data_cfg['use_camera']:
        config['model']['use_camera'] = data_cfg['use_camera']
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Build model
    print("Building model...")
    model = build_model(
        config,
        lite=args.lite,
        use_graph=not args.no_graph,
        use_camera=config['model'].get('use_camera', True),
    )
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Build dataset
    print("Loading dataset...")
    val_dataset = NuScenesDataset(
        root=config['data']['root'],
        version=config['data']['version'],
        split=config['data']['val_split'],
        img_size=tuple(config['model']['camera']['img_size']),
        point_cloud_range=config['model']['lidar']['point_cloud_range'],
        max_points=config['data']['max_points'],
        num_sweeps=data_cfg.get('num_sweeps', 1),
        sweep_step=data_cfg.get('sweep_step', 1),
        use_time_lag=data_cfg.get('use_time_lag', False),
        use_camera=data_cfg.get('use_camera', True),
        image_norm=data_cfg.get('image_norm', True),
        image_mean=tuple(data_cfg.get('image_mean', [0.485, 0.456, 0.406])),
        image_std=tuple(data_cfg.get('image_std', [0.229, 0.224, 0.225])),
        color_jitter=data_cfg.get('color_jitter', {}),
        augment=False,
    )

    sample_seed = int(data_cfg.get('sample_seed', 0))
    max_val_samples = data_cfg.get('max_val_samples')
    if isinstance(max_val_samples, int) and max_val_samples > 0:
        max_val_samples = min(max_val_samples, len(val_dataset))
        gen = torch.Generator().manual_seed(sample_seed + 1)
        indices = torch.randperm(len(val_dataset), generator=gen)[:max_val_samples].tolist()
        val_dataset = Subset(val_dataset, indices)
    
    train_cfg = config.get('training', {})
    val_num_workers = train_cfg.get('val_num_workers', 0)
    val_prefetch_factor = train_cfg.get('val_prefetch_factor', 2)
    val_persistent_workers = train_cfg.get('val_persistent_workers', False)
    pin_memory = train_cfg.get('pin_memory', True)

    loader_kwargs = {}
    if val_num_workers > 0:
        loader_kwargs['prefetch_factor'] = val_prefetch_factor
        loader_kwargs['persistent_workers'] = val_persistent_workers

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=val_num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        **loader_kwargs,
    )
    
    print(f"Validation samples: {len(val_dataset)}")
    
    # Run evaluation
    os.makedirs(args.output_dir, exist_ok=True)
    
    metrics = evaluate(
        model, val_loader, config, device,
        use_official=args.official
    )
    
    # Save metrics
    metrics_file = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        # Convert numpy to python types
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj
        
        json.dump(convert(metrics), f, indent=2)
    
    print(f"\nMetrics saved to {metrics_file}")


if __name__ == '__main__':
    main()
