#!/bin/bash
#SBATCH --job-name=bevfusion
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:4
#SBATCH --constraint=tesla_v100
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --chdir=/scratch/%u
#SBATCH --output=/scratch/%u/slurm_%x_%j.out
#SBATCH --error=/scratch/%u/slurm_%x_%j.err

# ============================================================================
# BEVFusion Training Script
# ============================================================================
# Usage:
#   CONFIG=configs/exp0_debug.yaml sbatch scripts/train.sh
#   CONFIG=configs/exp1_lidar.yaml sbatch scripts/train.sh
#   CONFIG=configs/exp2_lidar_camera.yaml sbatch scripts/train.sh
#   CONFIG=configs/exp3_lidar_camera_graph.yaml sbatch scripts/train.sh
# ============================================================================

set -euo pipefail

# ============================================================================
# Environment Setup
# ============================================================================
export PYTHONNOUSERSITE=1
unset PYTHONHOME || true
unset PYTHONPATH || true

ENV_NAME="${ENV_NAME:-comp541}"

module purge
module load conda3/latest

eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

alias python >/dev/null 2>&1 && unalias python || true
hash -r

export PATH="${CONDA_PREFIX}/bin:${PATH}"
PY="${CONDA_PREFIX}/bin/python"

SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${USER}}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/${USER}/nuscenes_fusion}"
WORK_DIR="${WORK_DIR:-${SCRATCH_ROOT}/nuscenes_fusion/work/${SLURM_JOB_ID}}"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

export PYTHONPATH="${PROJECT_ROOT}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [[ -f "${CONDA_PREFIX}/lib/libiomp5.so" ]]; then
    export LD_PRELOAD="${CONDA_PREFIX}/lib/libiomp5.so${LD_PRELOAD:+:$LD_PRELOAD}"
fi

# ============================================================================
# Cache and Performance Settings
# ============================================================================
export TORCH_HOME="${TORCH_HOME:-${SCRATCH_ROOT}/.cache/torch}"
export TORCHVISION_DISABLE_DOWNLOAD_PROGRESS="${TORCHVISION_DISABLE_DOWNLOAD_PROGRESS:-1}"
# NCCL settings for multi-GPU
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# ============================================================================
# Training Configuration
# ============================================================================
CONFIG="${CONFIG:-configs/exp1_lidar.yaml}"
if [[ "${CONFIG}" != /* ]]; then
    CONFIG="${PROJECT_ROOT}/${CONFIG}"
fi

# Extract experiment name from config filename
CONFIG_BASENAME=$(basename "${CONFIG}" .yaml)
NUM_GPUS="${NUM_GPUS:-4}"

# Avoid CPU oversubscription with DDP (one process per GPU)
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-1}"
THREADS_PER_PROC=$(( CPUS_PER_TASK / NUM_GPUS ))
if [[ "${THREADS_PER_PROC}" -lt 1 ]]; then
    THREADS_PER_PROC=1
fi
export OMP_NUM_THREADS="${THREADS_PER_PROC}"
export MKL_NUM_THREADS="${THREADS_PER_PROC}"

# Output directories
RUN_ROOT="${RUN_ROOT:-${SCRATCH_ROOT}/nuscenes_fusion/outputs}"
JOB_ROOT="${RUN_ROOT}/runs/${CONFIG_BASENAME}_${SLURM_JOB_ID}"
mkdir -p "${JOB_ROOT}"

# WandB
export WANDB_PROJECT="comp541"
export WANDB_ENTITY="merdem22-ko-university"
export WANDB_NAME="${CONFIG_BASENAME}_${SLURM_JOB_ID}"
export WANDB_DIR="${RUN_ROOT}/wandb"
mkdir -p "${WANDB_DIR}"

# Copy config for reproducibility
CONFIG_PATH="${JOB_ROOT}/config.yaml"
cp "${CONFIG}" "${CONFIG_PATH}"

# ============================================================================
# Prefetch Model Weights
# ============================================================================
echo "Prefetching ResNet18/34 weights into ${TORCH_HOME}..."
"${PY}" - <<'PY'
from torchvision import models
models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
print("ResNet18/34 weights downloaded successfully")
PY

# ============================================================================
# Training Launch
# ============================================================================
echo "=============================================="
echo "BEVFusion Training"
echo "=============================================="
echo "  Job ID:     ${SLURM_JOB_ID}"
echo "  Config:     ${CONFIG_PATH}"
echo "  Experiment: ${CONFIG_BASENAME}"
echo "  GPUs:       ${NUM_GPUS}"
echo "  Output:     ${JOB_ROOT}"
echo "  WandB:      ${WANDB_PROJECT}/${WANDB_NAME}"
echo "=============================================="

LAUNCH=("${PY}" -u -m torch.distributed.run --standalone --nproc_per_node="${NUM_GPUS}")

srun "${LAUNCH[@]}" "${PROJECT_ROOT}/train.py" \
    --config "${CONFIG_PATH}" \
    --output-dir "${JOB_ROOT}" \
    2>&1 | tee "${JOB_ROOT}/train.log"

echo "=============================================="
echo "Training completed"
echo "Outputs saved to: ${JOB_ROOT}"
echo "=============================================="
