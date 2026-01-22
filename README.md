# BEVFusion + Graph: Multi-Modal 3D Object Detection

PyTorch code for multi-modal 3D object detection on nuScenes using LiDAR-only, LiDAR+Camera fusion, and an optional BEV graph reasoning module.

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
- **Learnable Edge Graph Module**: Optional graph reasoning on fused BEV features (ablation-friendly)
- **Efficient Design**: Pillar-based LiDAR encoding + LSS-style camera projection
- **Visualization**: BEV + per-camera 3D box projections for demos
- **nuScenes Compatible**: Works with `v1.0-trainval` and `v1.0-mini`

## Installation

```bash
pip install -r requirements.txt
```

If you’re on a cluster, keep large artifacts (datasets/outputs/checkpoints) on fast storage (e.g. `/scratch`).

## Data

Download nuScenes and point `--root` to the dataset directory. For quick demos, use `v1.0-mini`.

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

## Evaluation

```bash
# Evaluate LiDAR + Camera + Graph
CHECKPOINT=/scratch/$USER/nuscenes_fusion/outputs/runs/exp3_lidar_camera_graph_<JOB_ID>/checkpoint_best.pth \
CONFIG=configs/exp3_lidar_camera_graph.yaml sbatch scripts/eval_slurm.sh

# Evaluate LiDAR-only
CHECKPOINT=/scratch/$USER/nuscenes_fusion/outputs/runs/exp1_lidar_<JOB_ID>/checkpoint_best.pth \
CONFIG=configs/exp1_lidar.yaml sbatch scripts/eval_slurm.sh
```

## Visualization (BEV + Cameras)

This repository includes a demo-oriented visualization script that:
- runs inference on a random subset of samples
- selects “good” and “questionable” samples using a simple proxy (distance-based matching)
- saves a BEV plot and 6 camera views with projected 3D boxes

Example on `v1.0-mini` (CPU):

```bash
python scripts/visualize_samples.py \
  --config configs/exp1_lidar.yaml \
  --checkpoint /path/to/checkpoint_best.pth \
  --root /path/to/nuscenes \
  --version v1.0-mini \
  --split mini_val \
  --device cpu \
  --save-mode demo \
  --save-cameras \
  --outdir outputs/vis_demo
```

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
                --resume checkpoint.pth
```

### evaluate.py
```bash
python evaluate.py --checkpoint path/to/checkpoint.pth \
                   --config configs/exp1_lidar.yaml \
                   --no-graph        # For models trained without graph
                   --official        # Use official nuScenes metrics
```

## License

MIT License
