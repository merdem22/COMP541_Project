#!/usr/bin/env python3
"""
Distributed Training Script for BEVFusion + Graph Model
Supports multi-GPU training with PyTorch DDP and WandB logging

Usage:
    Single GPU:  python train.py --config configs/exp1_lidar.yaml
    Multi-GPU:   torchrun --nproc_per_node=4 train.py --config configs/exp1_lidar.yaml
"""

import os
import sys
import argparse
import logging
import time
import re
from pathlib import Path
from typing import Dict, Optional
import yaml
from datetime import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler, autocast
import torch.optim as optim

# WandB (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WandB not available, logging disabled")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def _get_rss_gb() -> Optional[float]:
    """Best-effort resident set size (RSS) in GB (Linux only)."""
    try:
        if not os.path.exists("/proc/self/status"):
            return None
        with open("/proc/self/status", "r") as f:
            status = f.read()
        m = re.search(r"VmRSS:\\s+(\\d+)\\s+kB", status)
        if not m:
            return None
        kb = float(m.group(1))
        return kb / 1e6
    except Exception:
        return None

from src.data import NuScenesDataset, collate_fn
from src.models import build_model


def setup_logging(rank: int, output_dir: str) -> logging.Logger:
    """Setup logging for distributed training."""
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    
    if rank == 0:
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File handler
        os.makedirs(output_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(output_dir, 'train.log'))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger


def setup_distributed() -> tuple:
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank, True
    else:
        return 0, 1, 0, False


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def load_config(config_path: str) -> Dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_wandb(
    config: Dict,
    args,
    rank: int,
    model: nn.Module,
) -> Optional[object]:
    """Initialize WandB logging (rank 0 only)."""
    if not WANDB_AVAILABLE or rank != 0:
        return None
    
    # Get wandb config from environment or config file
    wandb_config = config.get('wandb', {})
    
    project = os.environ.get('WANDB_PROJECT', wandb_config.get('project', 'comp541'))
    entity = os.environ.get('WANDB_ENTITY', wandb_config.get('entity', 'merdem22-ko-university'))
    run_name = os.environ.get('WANDB_NAME', wandb_config.get('name', None))
    
    # Auto-generate run name if not provided
    if run_name is None:
        model_type = 'lite' if args.lite else 'full'
        graph_status = 'with_graph' if config['model'].get('use_graph', True) else 'no_graph'
        timestamp = datetime.now().strftime('%m%d_%H%M')
        run_name = f"bevfusion_{model_type}_{graph_status}_{timestamp}"
    
    graph_cfg = config.get('model', {}).get('graph', {}) or {}
    edge_type = str(graph_cfg.get('edge_type', 'dense')).lower()
    k_neighbors = int(graph_cfg.get('k_neighbors', 8))
    edge_topk = graph_cfg.get('edge_topk')
    edge_topk = None if edge_topk is None else int(edge_topk)
    edges_per_node = edge_topk if (edge_type in {'gnn', 'knn', 'message_passing'} and edge_topk is not None) else k_neighbors
    graph_stride = int(graph_cfg.get('stride', 1) or 1)
    bev_res = float(config.get('model', {}).get('bev_x_bound', [0, 0, 0.4])[2])
    bev_w = int((config['model']['bev_x_bound'][1] - config['model']['bev_x_bound'][0]) / bev_res)
    bev_h = int((config['model']['bev_y_bound'][1] - config['model']['bev_y_bound'][0]) / bev_res)
    graph_h = max(1, bev_h // max(1, graph_stride))
    graph_w = max(1, bev_w // max(1, graph_stride))
    graph_num_nodes = graph_h * graph_w
    graph_num_edges = graph_num_nodes * edges_per_node

    # Flatten config for logging
    flat_config = {
        'model_type': 'lite' if args.lite else 'full',
        'use_graph': config['model'].get('use_graph', True),
        'batch_size': config['training']['batch_size'],
        'lr': config['training']['lr'],
        'epochs': config['training']['epochs'],
        'grad_clip': config['training']['grad_clip'],
        'bev_resolution': config['model']['bev_x_bound'][2],
        'num_graph_layers': config['model']['graph'].get('num_layers', 3),
        'graph_edge_type': edge_type,
        'k_neighbors': k_neighbors,
        'edge_topk': edge_topk,
        'edges_per_node': edges_per_node,
        'graph_stride': graph_stride,
        'graph_nodes_h': graph_h,
        'graph_nodes_w': graph_w,
        'graph_num_nodes': graph_num_nodes,
        'graph_num_edges_per_sample': graph_num_edges,
    }
    
    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        config=flat_config,
        resume='allow',
    )
    
    # Log model architecture
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.config.update({'num_params_millions': num_params / 1e6})
    
    # Define custom metrics
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("val/epoch")
    wandb.define_metric("val/*", step_metric="val/epoch")
    wandb.define_metric("system/*", step_metric="train/step")
    
    return run


def build_optimizer(model: nn.Module, config: Dict) -> optim.Optimizer:
    """Build optimizer with parameter groups."""
    train_cfg = config['training']
    
    # Separate backbone and head parameters
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'detection_head' in name or 'graph_module' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
    
    param_groups = [
        {'params': backbone_params, 'lr': train_cfg['lr'] * 0.1},  # Lower LR for pretrained
        {'params': head_params, 'lr': train_cfg['lr']},
    ]
    
    optimizer = optim.AdamW(
        param_groups,
        weight_decay=train_cfg['weight_decay']
    )
    
    return optimizer


def build_scheduler(optimizer: optim.Optimizer, config: Dict, steps_per_epoch: int):
    """Build learning rate scheduler with warmup."""
    train_cfg = config['training']
    
    total_steps = train_cfg['epochs'] * steps_per_epoch
    warmup_steps = train_cfg['warmup_epochs'] * steps_per_epoch
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    epoch: int,
    config: Dict,
    logger: logging.Logger,
    rank: int,
    global_step: int,
    val_loader: Optional[DataLoader] = None,
    val_metrics_batches: Optional[int] = None,
    train_metrics_batches: int = 0,
    metrics_per_epoch: int = 1,
) -> tuple:
    """Train for one epoch with WandB logging."""
    model.train()
    
    grad_clip = config['training']['grad_clip']
    log_interval = config.get('logging', {}).get('train_log_interval', 20)
    
    total_loss = 0.0
    loss_dict_sum = {}
    num_batches = 0
    
    start_time = time.time()
    batch_start = time.time()

    metrics_interval = None
    if metrics_per_epoch > 1:
        metrics_interval = max(1, len(dataloader) // metrics_per_epoch)
    
    for batch_idx, batch in enumerate(dataloader):
        # Move data to GPU
        points = batch['points'].cuda(non_blocking=True)
        points_mask = batch['points_mask'].cuda(non_blocking=True)
        images = batch['images'].cuda(non_blocking=True)
        cam_intrinsics = batch['cam_intrinsics'].cuda(non_blocking=True)
        cam_extrinsics = batch['cam_extrinsics'].cuda(non_blocking=True)
        
        # Move annotations to GPU
        annotations = []
        for ann in batch['annotations']:
            annotations.append({
                'boxes': ann['boxes'].cuda(non_blocking=True),
                'labels': ann['labels'].cuda(non_blocking=True),
                'velocities': ann['velocities'].cuda(non_blocking=True),
            })
        
        optimizer.zero_grad()

        # Forward pass with mixed precision
        with autocast('cuda'):
            predictions = model(
                points, points_mask, images,
                cam_intrinsics, cam_extrinsics
            )
            
            # Compute loss
            if hasattr(model, 'module'):
                loss_dict = model.module.compute_loss(predictions, annotations)
            else:
                loss_dict = model.compute_loss(predictions, annotations)
            
            loss = loss_dict['total']
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        # Optimizer step (only step scheduler if optimizer step ran)
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= scale_before:
            scheduler.step()
        
        # Accumulate losses
        total_loss += loss.item()
        for k, v in loss_dict.items():
            if k not in loss_dict_sum:
                loss_dict_sum[k] = 0.0
            loss_dict_sum[k] += v.item()
        num_batches += 1
        global_step += 1
        
        # Calculate throughput
        batch_time = time.time() - batch_start
        samples_per_sec = points.shape[0] / batch_time
        batch_start = time.time()
        
        # Log to WandB and console at intervals
        if rank == 0 and batch_idx % log_interval == 0:
            lr = optimizer.param_groups[-1]['lr']
            rss = _get_rss_gb()
            rss_s = f'{rss:.1f}GB' if rss is not None else 'n/a'
            
            logger.info(
                f'Epoch [{epoch}] Batch [{batch_idx}/{len(dataloader)}] '
                f'Loss: {loss.item():.4f} LR: {lr:.6f} RSS: {rss_s}'
            )
            
            # WandB logging
            if WANDB_AVAILABLE and wandb.run is not None:
                log_dict = {
                    'train/step': global_step,
                    'train/loss_total': loss.item(),
                    'train/loss_heatmap': loss_dict['heatmap'].item(),
                    'train/loss_reg': loss_dict['reg'].item(),
                    'train/loss_height': loss_dict['height'].item(),
                    'train/loss_dim': loss_dict['dim'].item(),
                    'train/loss_rot': loss_dict['rot'].item(),
                    'train/loss_vel': loss_dict['vel'].item(),
                    'train/learning_rate': lr,
                    'train/grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    'train/epoch': epoch + batch_idx / len(dataloader),
                    'system/samples_per_sec': samples_per_sec,
                    'system/gpu_memory_gb': torch.cuda.max_memory_allocated() / 1e9,
                }
                if rss is not None:
                    log_dict['system/host_rss_gb'] = rss
                wandb.log(log_dict)

        if metrics_interval is not None:
            if (batch_idx + 1) % metrics_interval == 0 and (batch_idx + 1) < len(dataloader):
                epoch_progress = (batch_idx + 1) / len(dataloader)
                if val_loader is not None:
                    run_validation(
                        model,
                        val_loader,
                        config,
                        logger,
                        rank,
                        epoch,
                        epoch_progress,
                        max_batches=val_metrics_batches,
                    )
                if train_metrics_batches > 0:
                    run_train_metrics(
                        model,
                        dataloader,
                        config,
                        logger,
                        rank,
                        epoch,
                        epoch_progress,
                        global_step,
                        train_metrics_batches,
                    )
                model.train()
    
    # Average losses
    avg_loss = total_loss / num_batches
    avg_loss_dict = {k: v / num_batches for k, v in loss_dict_sum.items()}
    
    epoch_time = time.time() - start_time
    
    if rank == 0:
        logger.info(
            f'Epoch [{epoch}] completed in {epoch_time:.1f}s '
            f'Avg Loss: {avg_loss:.4f}'
        )
        for k, v in avg_loss_dict.items():
            logger.info(f'  {k}: {v:.4f}')
        
        # Log epoch summary to WandB
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({
                'train/step': global_step,
                'train/epoch_loss': avg_loss,
                'train/epoch_time_sec': epoch_time,
            })
    
    return avg_loss_dict, global_step


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    config: Dict,
    logger: logging.Logger,
    rank: int,
    epoch: int = 0,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Run validation with comprehensive metrics."""
    model.eval()
    
    total_loss = 0.0
    num_batches = 0

    from src.data import CLASS_NAMES
    per_class_stats, totals = init_metrics_state(CLASS_NAMES)
    distance_thresh = config.get('eval', {}).get('distance_thresh', 2.0)

    for batch_idx, batch in enumerate(dataloader):
        points = batch['points'].cuda(non_blocking=True)
        points_mask = batch['points_mask'].cuda(non_blocking=True)
        images = batch['images'].cuda(non_blocking=True)
        cam_intrinsics = batch['cam_intrinsics'].cuda(non_blocking=True)
        cam_extrinsics = batch['cam_extrinsics'].cuda(non_blocking=True)
        
        annotations = []
        for ann in batch['annotations']:
            annotations.append({
                'boxes': ann['boxes'].cuda(non_blocking=True),
                'labels': ann['labels'].cuda(non_blocking=True),
                'velocities': ann['velocities'].cuda(non_blocking=True),
            })

        with autocast('cuda'):
            predictions = model(
                points, points_mask, images,
                cam_intrinsics, cam_extrinsics
            )
            
            if hasattr(model, 'module'):
                loss_dict = model.module.compute_loss(predictions, annotations)
                decoded = model.module.predict(predictions)
            else:
                loss_dict = model.compute_loss(predictions, annotations)
                decoded = model.predict(predictions)
        
        total_loss += loss_dict['total'].item()
        num_batches += 1

        update_metrics_state(
            decoded,
            annotations,
            CLASS_NAMES,
            per_class_stats,
            totals,
            distance_thresh=distance_thresh,
        )

        # Proactively drop references to large tensors during long validations.
        del predictions, decoded, points, points_mask, images, cam_intrinsics, cam_extrinsics, annotations
        if torch.cuda.is_available() and (batch_idx + 1) % 50 == 0:
            torch.cuda.empty_cache()

        if max_batches is not None and (batch_idx + 1) >= max_batches:
            break
    
    avg_loss = total_loss / num_batches
    
    # Compute comprehensive metrics with per-class breakdown
    metrics = finalize_metrics_state(CLASS_NAMES, per_class_stats, totals)
    
    if rank == 0:
        logger.info(f'Validation Loss: {avg_loss:.4f}')
        logger.info(f'  Precision: {metrics["precision"]:.4f}')
        logger.info(f'  Recall: {metrics["recall"]:.4f}')
        logger.info(f'  F1: {metrics["f1"]:.4f}')
        logger.info(f'  mAP: {metrics["mAP"]:.4f}')
        logger.info(f'Per-class AP:')
        for cls_name, ap in metrics['per_class_AP'].items():
            logger.info(f'    {cls_name}: {ap:.4f}')
    
    return {'val_loss': avg_loss, **metrics}


def run_validation(
    model: nn.Module,
    dataloader: DataLoader,
    config: Dict,
    logger: logging.Logger,
    rank: int,
    epoch: int,
    epoch_progress: float,
    max_batches: Optional[int],
) -> Dict[str, float]:
    """Run validation and log to WandB."""
    if rank == 0:
        total_batches = max_batches if max_batches is not None else len(dataloader)
        rss = _get_rss_gb()
        rss_s = f"{rss:.1f}GB" if rss is not None else "n/a"
        logger.info(f'Validation start ({total_batches} batches, rss={rss_s})')
    val_metrics = validate(
        model,
        dataloader,
        config,
        logger,
        rank,
        epoch=epoch,
        max_batches=max_batches,
    )

    if rank == 0 and WANDB_AVAILABLE and wandb.run is not None:
        val_log = {
            'val/epoch': epoch + epoch_progress,
            'val/loss': val_metrics['val_loss'],
            'val/precision': val_metrics['precision'],
            'val/recall': val_metrics['recall'],
            'val/f1': val_metrics['f1'],
            'val/mAP': val_metrics.get('mAP', 0.0),
        }
        if 'per_class_AP' in val_metrics:
            for cls_name, ap in val_metrics['per_class_AP'].items():
                val_log[f'val/AP_{cls_name}'] = ap
        wandb.log(val_log)

    return val_metrics


def run_train_metrics(
    model: nn.Module,
    dataloader: DataLoader,
    config: Dict,
    logger: logging.Logger,
    rank: int,
    epoch: int,
    epoch_progress: float,
    global_step: int,
    max_batches: int,
) -> Dict[str, float]:
    """Run train metrics on a subset and log to WandB."""
    if rank == 0:
        logger.info(f'Train-metrics start ({max_batches} batches)')
    base_dataset = dataloader.dataset
    if isinstance(base_dataset, Subset):
        base_dataset = base_dataset.dataset
    original_augment = getattr(base_dataset, 'augment', False)
    base_dataset.augment = False

    train_eval_metrics = validate(
        model,
        dataloader,
        config,
        logger,
        rank,
        epoch=epoch,
        max_batches=max_batches,
    )

    base_dataset.augment = original_augment

    if rank == 0:
        logger.info(
            f'Train metrics (subset, {max_batches} batches): '
            f'Precision {train_eval_metrics["precision"]:.4f}, '
            f'Recall {train_eval_metrics["recall"]:.4f}, '
            f'F1 {train_eval_metrics["f1"]:.4f}, '
            f'mAP {train_eval_metrics["mAP"]:.4f}'
        )

        if WANDB_AVAILABLE and wandb.run is not None:
            train_log = {
                'train/step': global_step,
                'train/epoch': epoch + epoch_progress,
                'train/precision': train_eval_metrics['precision'],
                'train/recall': train_eval_metrics['recall'],
                'train/f1': train_eval_metrics['f1'],
                'train/mAP': train_eval_metrics.get('mAP', 0.0),
            }
            if 'per_class_AP' in train_eval_metrics:
                for cls_name, ap in train_eval_metrics['per_class_AP'].items():
                    train_log[f'train/AP_{cls_name}'] = ap
            wandb.log(train_log)

    return train_eval_metrics


def compute_metrics(
    predictions: list,
    annotations: list,
    iou_thresh: float = 0.5,
) -> Dict[str, float]:
    """Compute simple detection metrics (for backward compatibility)."""
    return compute_metrics_with_per_class(predictions, annotations)


def init_metrics_state(class_names: list) -> tuple:
    """Initialize metric accumulation state."""
    from array import array
    per_class_stats = {
        name: {'tp': array('b'), 'fp': array('b'), 'scores': array('f'), 'num_gt': 0}
        for name in class_names
    }
    totals = {'tp': 0, 'fp': 0, 'fn': 0}
    return per_class_stats, totals


def update_metrics_state(
    predictions: list,
    annotations: list,
    class_names: list,
    per_class_stats: Dict[str, Dict],
    totals: Dict[str, int],
    distance_thresh: float = 2.0,
) -> None:
    """Update metric state with a batch of predictions/annotations."""
    for pred, ann in zip(predictions, annotations):
        pred_boxes = pred['boxes'].cpu()
        pred_scores = pred['scores'].cpu()
        pred_labels = pred['labels'].cpu()

        gt_boxes = ann['boxes'].cpu()
        gt_labels = ann['labels'].cpu()

        for cls_idx, cls_name in enumerate(class_names):
            pred_mask = pred_labels == cls_idx
            gt_mask = gt_labels == cls_idx

            cls_pred_boxes = pred_boxes[pred_mask]
            cls_pred_scores = pred_scores[pred_mask]
            cls_gt_boxes = gt_boxes[gt_mask]

            num_gt = len(cls_gt_boxes)
            per_class_stats[cls_name]['num_gt'] += num_gt

            if num_gt == 0:
                for score in cls_pred_scores:
                    per_class_stats[cls_name]['tp'].append(0)
                    per_class_stats[cls_name]['fp'].append(1)
                    per_class_stats[cls_name]['scores'].append(float(score))
                totals['fp'] += len(cls_pred_boxes)
                continue

            if len(cls_pred_boxes) == 0:
                totals['fn'] += num_gt
                continue

            order = cls_pred_scores.argsort(descending=True)
            cls_pred_boxes = cls_pred_boxes[order]
            cls_pred_scores = cls_pred_scores[order]

            matched_gt = set()
            for p_box, p_score in zip(cls_pred_boxes, cls_pred_scores):
                distances = torch.sqrt(
                    (cls_gt_boxes[:, 0] - p_box[0])**2 +
                    (cls_gt_boxes[:, 1] - p_box[1])**2
                )

                min_dist, min_idx = distances.min(), distances.argmin().item()

                if min_dist < distance_thresh and min_idx not in matched_gt:
                    per_class_stats[cls_name]['tp'].append(1)
                    per_class_stats[cls_name]['fp'].append(0)
                    matched_gt.add(min_idx)
                    totals['tp'] += 1
                else:
                    per_class_stats[cls_name]['tp'].append(0)
                    per_class_stats[cls_name]['fp'].append(1)
                    totals['fp'] += 1
                
                per_class_stats[cls_name]['scores'].append(float(p_score))

            totals['fn'] += num_gt - len(matched_gt)


def finalize_metrics_state(
    class_names: list,
    per_class_stats: Dict[str, Dict],
    totals: Dict[str, int],
) -> Dict[str, float]:
    """Finalize metric state into summary metrics."""
    import numpy as np

    aps = {}
    for cls_name in class_names:
        stats = per_class_stats[cls_name]
        if len(stats['scores']) == 0 or stats['num_gt'] == 0:
            aps[cls_name] = 0.0
            continue

        scores = np.frombuffer(stats['scores'], dtype=np.float32)
        tp = np.frombuffer(stats['tp'], dtype=np.int8)
        fp = np.frombuffer(stats['fp'], dtype=np.int8)

        order = np.argsort(-scores)
        tp = tp[order]
        fp = fp[order]

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        recall = tp_cumsum / (stats['num_gt'] + 1e-6)

        ap = 0.0
        for t in np.linspace(0, 1, 11):
            mask = recall >= t
            if mask.sum() > 0:
                ap += precision[mask].max() / 11

        aps[cls_name] = float(ap)

    valid_aps = [ap for cls_name, ap in aps.items()
                 if per_class_stats[cls_name]['num_gt'] > 0]
    mAP = float(np.mean(valid_aps)) if valid_aps else 0.0

    total_tp = totals['tp']
    total_fp = totals['fp']
    total_fn = totals['fn']
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'mAP': mAP,
        'per_class_AP': aps,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
    }


def compute_metrics_with_per_class(
    predictions: list,
    annotations: list,
    distance_thresh: float = 2.0,
) -> Dict[str, float]:
    """
    Compute comprehensive detection metrics including per-class AP.
    Uses center distance matching (2m threshold by default).
    """
    from src.data import CLASS_NAMES
    per_class_stats, totals = init_metrics_state(CLASS_NAMES)
    update_metrics_state(
        predictions,
        annotations,
        CLASS_NAMES,
        per_class_stats,
        totals,
        distance_thresh=distance_thresh,
    )
    return finalize_metrics_state(CLASS_NAMES, per_class_stats, totals)


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: Dict,
    output_dir: str,
    is_best: bool = False,
):
    """Save model checkpoint."""
    state = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics,
    }
    
    checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(state, checkpoint_path)
    
    # Save latest
    latest_path = os.path.join(output_dir, 'checkpoint_latest.pth')
    torch.save(state, latest_path)
    
    if is_best:
        best_path = os.path.join(output_dir, 'checkpoint_best.pth')
        torch.save(state, best_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/exp1_lidar.yaml')
    parser.add_argument('--output-dir', type=str, default='outputs')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--lite', action='store_true', help='Use lightweight model')
    parser.add_argument('--no-graph', action='store_true', help='Disable graph module (ablation)')
    parser.add_argument('--no-camera', action='store_true', help='Disable camera branch (LiDAR-only)')
    parser.add_argument('--eval-only', action='store_true')
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Setup distributed
    rank, world_size, local_rank, distributed = setup_distributed()
    
    # Load config
    config = load_config(args.config)
    
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
    
    # Setup logging
    logger = setup_logging(rank, args.output_dir)
    
    if rank == 0:
        logger.info(f'Config: {args.config}')
        logger.info(f'World size: {world_size}')
        logger.info(f'Using {"lite" if args.lite else "full"} model')
        logger.info(
            f'Graph module: {"enabled" if config["model"].get("use_graph", True) else "disabled"}'
        )
        logger.info(
            f'Camera branch: {"enabled" if config["model"].get("use_camera", True) else "disabled"}'
        )
        if config["model"].get("use_graph", True):
            graph_cfg = config["model"].get("graph", {}) or {}
            edge_type = str(graph_cfg.get("edge_type", "dense")).lower()
            k_neighbors = int(graph_cfg.get("k_neighbors", 8))
            edge_topk = graph_cfg.get("edge_topk")
            edge_topk = None if edge_topk is None else int(edge_topk)
            edges_per_node = edge_topk if (edge_type in {"gnn", "knn", "message_passing"} and edge_topk is not None) else k_neighbors
            graph_stride = int(graph_cfg.get("stride", 1) or 1)
            bev_res = float(config["model"]["bev_x_bound"][2])
            bev_w = int((config["model"]["bev_x_bound"][1] - config["model"]["bev_x_bound"][0]) / bev_res)
            bev_h = int((config["model"]["bev_y_bound"][1] - config["model"]["bev_y_bound"][0]) / bev_res)
            graph_h = max(1, bev_h // max(1, graph_stride))
            graph_w = max(1, bev_w // max(1, graph_stride))
            graph_num_nodes = graph_h * graph_w
            graph_num_edges = graph_num_nodes * edges_per_node
            logger.info(
                f'Graph edges/node: {edges_per_node} (k_neighbors={k_neighbors}, edge_topk={edge_topk}, edge_type={edge_type})'
            )
            logger.info(
                f'Graph nodes: {graph_h}x{graph_w}={graph_num_nodes} (stride={graph_stride}); '
                f'Edges/sample: {graph_num_edges}'
            )
    
    # Build model
    model = build_model(
        config,
        lite=args.lite,
        use_graph=not args.no_graph,
        use_camera=config['model'].get('use_camera', True),
    )
    model = model.cuda()
    
    if distributed:
        model = DDP(model, device_ids=[local_rank])
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        logger.info(f'Model parameters: {num_params / 1e6:.2f}M')
    
    # Setup WandB (after model is built so we can log param count)
    wandb_run = setup_wandb(config, args, rank, model)
    
    # Build datasets
    train_cfg = config['training']
    
    train_dataset = NuScenesDataset(
        root=data_cfg['root'],
        version=data_cfg['version'],
        split=data_cfg['train_split'],
        img_size=tuple(config['model']['camera']['img_size']),
        point_cloud_range=config['model']['lidar']['point_cloud_range'],
        max_points=data_cfg['max_points'],
        num_sweeps=data_cfg.get('num_sweeps', 1),
        sweep_step=data_cfg.get('sweep_step', 1),
        use_time_lag=data_cfg.get('use_time_lag', False),
        use_camera=data_cfg.get('use_camera', True),
        image_norm=data_cfg.get('image_norm', True),
        image_mean=tuple(data_cfg.get('image_mean', [0.485, 0.456, 0.406])),
        image_std=tuple(data_cfg.get('image_std', [0.229, 0.224, 0.225])),
        color_jitter=data_cfg.get('color_jitter', {}),
        augment=True,
    )
    
    val_dataset = NuScenesDataset(
        root=data_cfg['root'],
        version=data_cfg['version'],
        split=data_cfg['val_split'],
        img_size=tuple(config['model']['camera']['img_size']),
        point_cloud_range=config['model']['lidar']['point_cloud_range'],
        max_points=data_cfg['max_points'],
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
    max_train_samples = data_cfg.get('max_train_samples')
    if isinstance(max_train_samples, int) and max_train_samples > 0:
        max_train_samples = min(max_train_samples, len(train_dataset))
        gen = torch.Generator().manual_seed(sample_seed)
        indices = torch.randperm(len(train_dataset), generator=gen)[:max_train_samples].tolist()
        train_dataset = Subset(train_dataset, indices)

    max_val_samples = data_cfg.get('max_val_samples')
    if isinstance(max_val_samples, int) and max_val_samples > 0:
        max_val_samples = min(max_val_samples, len(val_dataset))
        gen = torch.Generator().manual_seed(sample_seed + 1)
        indices = torch.randperm(len(val_dataset), generator=gen)[:max_val_samples].tolist()
        val_dataset = Subset(val_dataset, indices)
    
    if rank == 0:
        logger.info(f'Train samples: {len(train_dataset)}')
        logger.info(f'Val samples: {len(val_dataset)}')
        rss = _get_rss_gb()
        if rss is not None:
            logger.info(f'Host RSS at startup: {rss:.1f}GB')
    
    # Build dataloaders
    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    
    pin_memory = train_cfg.get('pin_memory', True)

    train_num_workers = train_cfg.get('num_workers', 4)
    train_prefetch_factor = train_cfg.get('prefetch_factor', 2)
    train_persistent_workers = train_cfg.get('persistent_workers', True)

    train_loader_kwargs = {}
    if train_num_workers > 0:
        train_loader_kwargs['prefetch_factor'] = train_prefetch_factor
        train_loader_kwargs['persistent_workers'] = train_persistent_workers

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg['batch_size'],
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=train_num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=True,
        **train_loader_kwargs,
    )

    val_num_workers = train_cfg.get('val_num_workers', 0)
    val_prefetch_factor = train_cfg.get('val_prefetch_factor', 2)
    val_persistent_workers = train_cfg.get('val_persistent_workers', False)
    val_loader_kwargs = {}
    if val_num_workers > 0:
        val_loader_kwargs['prefetch_factor'] = val_prefetch_factor
        val_loader_kwargs['persistent_workers'] = val_persistent_workers

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg['batch_size'],
        sampler=val_sampler,
        shuffle=False,
        num_workers=val_num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        **val_loader_kwargs,
    )

    train_metrics_batches = train_cfg.get('train_metrics_batches', 0)
    if not isinstance(train_metrics_batches, int) or train_metrics_batches < 0:
        train_metrics_batches = 0
    val_metrics_batches = train_cfg.get('val_metrics_batches')
    if isinstance(val_metrics_batches, int) and val_metrics_batches <= 0:
        val_metrics_batches = None
    metrics_per_epoch = train_cfg.get('metrics_per_epoch', 1)
    if not isinstance(metrics_per_epoch, int) or metrics_per_epoch < 1:
        metrics_per_epoch = 1
    
    # Build optimizer and scheduler
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, len(train_loader))
    scaler = GradScaler('cuda')
    
    # Resume from checkpoint
    start_epoch = 0
    best_metric = 0.0
    global_step = 0
    
    if args.resume:
        if rank == 0:
            logger.info(f'Resuming from {args.resume}')
        
        checkpoint = torch.load(args.resume, map_location='cpu')
        
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_metric = checkpoint['metrics'].get('f1', 0.0)
        global_step = start_epoch * len(train_loader)
    
    # Eval only mode
    if args.eval_only:
        metrics = validate(model, val_loader, config, logger, rank, epoch=0)
        if rank == 0:
            logger.info('Evaluation complete')
        cleanup_distributed()
        return
    
    # Training loop
    for epoch in range(start_epoch, train_cfg['epochs']):
        if distributed:
            train_sampler.set_epoch(epoch)
        
        # Train
        train_metrics, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            epoch, config, logger, rank, global_step,
            val_loader=val_loader,
            val_metrics_batches=val_metrics_batches,
            train_metrics_batches=train_metrics_batches,
            metrics_per_epoch=metrics_per_epoch,
        )
        
        val_metrics = run_validation(
            model,
            val_loader,
            config,
            logger,
            rank,
            epoch,
            epoch_progress=1.0,
            max_batches=val_metrics_batches,
        )

        # Save checkpoint
        if rank == 0:
            is_best = val_metrics['f1'] > best_metric
            if is_best:
                best_metric = val_metrics['f1']
                if WANDB_AVAILABLE and wandb.run is not None:
                    wandb.run.summary['best_f1'] = best_metric
                    wandb.run.summary['best_epoch'] = epoch

            save_checkpoint(
                model, optimizer, scheduler, epoch,
                val_metrics, args.output_dir, is_best
            )

        if train_metrics_batches > 0:
            run_train_metrics(
                model,
                train_loader,
                config,
                logger,
                rank,
                epoch,
                epoch_progress=1.0,
                global_step=global_step,
                max_batches=train_metrics_batches,
            )
        
        # Sync before next epoch
        if distributed:
            dist.barrier()
    
    if rank == 0:
        logger.info(f'Training complete. Best F1: {best_metric:.4f}')
        
        # Final WandB summary
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.run.summary['final_best_f1'] = best_metric
            wandb.finish()
    
    cleanup_distributed()


if __name__ == '__main__':
    main()
