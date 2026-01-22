# BEVFusion + Graph: Multi-Modal 3D Object Detection

A PyTorch implementation of multi-modal 3D object detection combining LiDAR and camera inputs with a novel **learnable edge graph module** for BEV feature reasoning.

## Architecture Overview

```
┌─────────────────┐     ┌─────────────────┐
│   LiDAR Input   │     │  Camera Input   │
│   (N, 5) pts    │     │  (6, 3, H, W)   │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│  Pillar VFE +   │     │   ResNet +      │
│  2D CNN Encoder │     │   LSS Depth     │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│   LiDAR BEV     │     │   Camera BEV    │
│   (C, H, W)     │     │   (C, H, W)     │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
           ┌─────────────────┐
           │   BEV Fusion    │
           │  (Concatenate)  │
           └────────┬────────┘
                    │
                    ▼
           ┌─────────────────┐
           │  Graph Module   │  ← Key Innovation!
           │ (Learnable Edge │     (can be disabled
           │   GNN + Local   │      for ablation)
           │   Attention)    │
           └────────┬────────┘
                    │
                    ▼
           ┌─────────────────┐
           │ CenterPoint     │
           │ Detection Head  │
           └────────┬────────┘
                    │
                    ▼
           ┌─────────────────┐
           │  3D Detections  │
           │ (boxes, scores) │
           └─────────────────┘
```

## Key Features

- **Multi-Modal Fusion**: Combines LiDAR point clouds and 6 surround-view cameras
- **Learnable Edge Graph Module**: Novel graph reasoning with learned edge features
- **Ablation Support**: Can train with or without graph module for comparison
- **Efficient Design**: Pillar-based LiDAR encoding + LSS-style camera projection
- **Multi-GPU Training**: Full DDP support for 4x V100 GPUs
- **WandB Integration**: Comprehensive logging with per-class metrics
- **nuScenes Compatible**: Works with keyframes-only dataset

## Installation

```bash
# Create environment (or use existing comp541 environment)
conda activate comp541

# Install dependencies
pip install -r requirements.txt

# Dataset is available at /datasets/nuscenes (symlinked to data/nuscenes in the repo)
```

Keep the codebase under `/home/$USER/nuscenes_fusion`. Training and evaluation outputs will be written to `/scratch/$USER/nuscenes_fusion/outputs`.

## Training on Valar HPC

### Experiments

```bash
# Quick smoke test (small subset)
CONFIG=configs/exp0_debug.yaml sbatch scripts/train.sh

# LiDAR-only baseline
CONFIG=configs/exp1_lidar.yaml sbatch scripts/train.sh

# LiDAR + Camera baseline
CONFIG=configs/exp2_lidar_camera.yaml sbatch scripts/train.sh

# LiDAR + Camera + Graph
CONFIG=configs/exp3_lidar_camera_graph.yaml sbatch scripts/train.sh
```

### Monitor on WandB

Results will be logged to: `wandb.ai/merdem22-ko-university/comp541`

Key metrics tracked:
- **Training**: loss (total, heatmap, reg, height, dim, rot, vel), LR, grad norm, throughput
- **Validation**: precision, recall, F1, mAP, per-class AP
- **System**: GPU memory, samples/sec

## Evaluation

```bash
# Evaluate LiDAR + Camera + Graph
CHECKPOINT=/scratch/$USER/nuscenes_fusion/outputs/runs/exp3_lidar_camera_graph_<JOB_ID>/checkpoint_best.pth \
CONFIG=configs/exp3_lidar_camera_graph.yaml sbatch scripts/eval_slurm.sh

# Evaluate LiDAR-only
CHECKPOINT=/scratch/$USER/nuscenes_fusion/outputs/runs/exp1_lidar_<JOB_ID>/checkpoint_best.pth \
CONFIG=configs/exp1_lidar.yaml sbatch scripts/eval_slurm.sh
```

## Configuration

Key hyperparameters in `configs/exp1_lidar.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 2 | Per-GPU batch size |
| `lr` | 2e-4 | Learning rate |
| `epochs` | 10 | Training epochs |
| `bev_x_bound` | [-51.2, 51.2, 0.4] | BEV X range [min, max, res] |
| `camera.img_size` | [256, 704] | Input image size (H, W) |
| `camera.depth_bins` | 64 | Depth bins for LSS projection |
| `graph.num_layers` | 2 | Graph conv layers (exp3) |
| `data.num_sweeps` | 1 | LiDAR sweeps to fuse (set >1 for multi-sweep) |
| `data.use_time_lag` | false | Append time-lag channel for multi-sweep |

## WandB Logging Details

Metrics are logged at different intervals for optimal visualization:

- **Every N batches**: Training loss components, LR, grad norm, throughput (see `logging.train_log_interval`)
- **End of each epoch**: Validation metrics (precision, recall, F1, mAP, per-class AP)
- **Optional mid-epoch**: If `training.metrics_per_epoch > 1`

## Graph Module Details

The graph module combines:

1. **Dense Local Graph Attention**: Efficient local reasoning with learnable relative position encoding
2. **Sparse Global Graph**: k-NN based global context aggregation with learned edge features

Edge features are computed from:
- Relative spatial positions
- Feature difference norms
- Cosine similarity between node features

## Expected Performance

With 4x V100 GPUs and 10 epochs:

| Model | mAP | Training Time |
|-------|-----|---------------|
| LiDAR-only (exp1) | 0.18-0.22 | ~14-18h |
| LiDAR + Camera (exp2) | 0.20-0.25 | ~18-22h |
| LiDAR + Camera + Graph (exp3) | 0.25-0.30 | ~22-26h |

## Project Structure

```
nuscenes_fusion/
├── configs/
│   ├── exp0_debug.yaml        # Quick smoke test (small subset)
│   ├── exp1_lidar.yaml        # LiDAR-only baseline
│   ├── exp2_lidar_camera.yaml # LiDAR + camera baseline
│   └── exp3_lidar_camera_graph.yaml # LiDAR + camera + graph
├── scripts/
│   ├── train.sh               # SLURM: training launcher (set CONFIG env var)
│   └── eval_slurm.sh          # SLURM: evaluation
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py         # nuScenes dataset loader
│   └── models/
│       ├── __init__.py
│       ├── bevfusion_graph.py # Main model (supports --no-graph)
│       ├── lidar_backbone.py  # Pillar-based LiDAR encoder
│       ├── camera_backbone.py # LSS-style camera encoder
│       ├── graph_module.py    # Learnable edge GNN
│       ├── detection_head.py  # CenterPoint head
│       └── dataset_utils.py
├── train.py                   # Training with WandB logging
├── evaluate.py                # Evaluation script
├── requirements.txt
└── README.md
```

## Command Line Options

### train.py
```bash
python train.py --config configs/exp1_lidar.yaml \
                --output-dir outputs/run1 \
                --no-graph           # Disable graph module (ablation)
                --lite               # Use lightweight model
                --resume checkpoint.pth
```

### evaluate.py
```bash
python evaluate.py --checkpoint path/to/checkpoint.pth \
                   --config configs/exp1_lidar.yaml \
                   --no-graph        # For models trained without graph
                   --official        # Use official nuScenes metrics
```

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in config
- Use `--lite` flag for smaller model
- Reduce `max_points` in data config

### WandB Issues
- Check WANDB_API_KEY is set
- Verify entity name: `merdem22-ko-university`
- Check network connectivity on compute nodes

### Poor Convergence
- Increase warmup epochs
- Reduce learning rate
- Enable gradient clipping (default: 35.0)

## Citation

If this code helps your research, please consider citing:

```bibtex
@article{bevfusion2022,
  title={BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation},
  author={Liu, Zhijian and Tang, Haotian and Amini, Alexander and Yang, Xinyu and Mao, Huizi and Rus, Daniela and Han, Song},
  journal={arXiv preprint arXiv:2205.13542},
  year={2022}
}
```

## License

MIT License
