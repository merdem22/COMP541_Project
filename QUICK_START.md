# Quick Start Guide - Valar HPC Training

This repo should live in `/home/$USER/nuscenes_fusion`. The dataset is at `/datasets/nuscenes` (symlinked to `data/nuscenes` in the repo), while run outputs go to `/scratch/$USER`.

## Quick Debug Run (Test First!)

Before running real experiments, test that everything works:

```bash
CONFIG=configs/exp0_debug.yaml sbatch scripts/train.sh
```

This runs 1 epoch on 500 samples (~30 min). Check it completes without errors before running full experiments.

## Three Experiments

| Experiment | Config | Command | Target mAP |
|------------|--------|---------|------------|
| 1. LiDAR-only | `configs/exp1_lidar.yaml` | `CONFIG=configs/exp1_lidar.yaml sbatch scripts/train.sh` | 0.18-0.22 |
| 2. LiDAR + Camera | `configs/exp2_lidar_camera.yaml` | `CONFIG=configs/exp2_lidar_camera.yaml sbatch scripts/train.sh` | 0.20-0.25 |
| 3. LiDAR + Camera + Graph | `configs/exp3_lidar_camera_graph.yaml` | `CONFIG=configs/exp3_lidar_camera_graph.yaml sbatch scripts/train.sh` | 0.22-0.27 |

## 1. Submit Training Jobs

```bash
# SSH to Valar
ssh merdem22@valar.ku.edu.tr
cd ~/nuscenes_fusion

# Experiment 1: LiDAR-only baseline
CONFIG=configs/exp1_lidar.yaml sbatch scripts/train.sh

# Experiment 2: LiDAR + Camera baseline
CONFIG=configs/exp2_lidar_camera.yaml sbatch scripts/train.sh

# Experiment 3: LiDAR + Camera + Graph (main model)
CONFIG=configs/exp3_lidar_camera_graph.yaml sbatch scripts/train.sh
```

## 2. Monitor Training

```bash
# Check job status
squeue -u merdem22

# View live logs
tail -f /scratch/$USER/nuscenes_fusion/outputs/runs/*/train.log

# Check WandB dashboard
# https://wandb.ai/merdem22-ko-university/comp541
```

## 3. Configuration Details

All configs use memory-safe settings:
- **BEV Resolution**: 0.4m (256x256 grid) - fits in V100 32GB
- **Batch size**: 2 per GPU (8 total with 4 GPUs)
- **Epochs**: 10
- **Learning rate**: 2e-4

Model parameters:
- LiDAR-only: 3.52M params
- LiDAR + Camera: 25.60M params
- LiDAR + Camera + Graph: 25.90M params

## 4. Output Locations

```
/scratch/$USER/nuscenes_fusion/outputs/
├── runs/
│   └── <experiment>_<JOB_ID>/
│       ├── checkpoint_best.pth
│       ├── checkpoint_latest.pth
│       ├── config.yaml
│       └── train.log
└── wandb/
```

## 5. Post-Training Evaluation

```bash
CHECKPOINT=/scratch/$USER/nuscenes_fusion/outputs/runs/<run_dir>/checkpoint_best.pth \
sbatch scripts/eval_slurm.sh
```

## 6. Troubleshooting

### Out of Memory
Reduce batch_size from 2 to 1 in the config file.

### Job Killed
Check SLURM logs:
```bash
cat /scratch/$USER/slurm_bevfusion_*.err
```

### Import Errors
```bash
conda activate comp541
python -c "from src.models import build_model; print('OK')"
```
