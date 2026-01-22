#!/bin/bash
#SBATCH --job-name=bevfusion_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=tesla_v100
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --chdir=/scratch/%u
#SBATCH --output=/scratch/%u/slurm_%x_%j.out
#SBATCH --error=/scratch/%u/slurm_%x_%j.err

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
RUN_ROOT="${RUN_ROOT:-${SCRATCH_ROOT}/nuscenes_fusion/outputs}"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "${WORK_DIR}"
mkdir -p "${LOG_DIR}"
cd "${WORK_DIR}"

export PYTHONPATH="${PROJECT_ROOT}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export TORCH_HOME="${TORCH_HOME:-${SCRATCH_ROOT}/.cache/torch}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# ============================================================================
# Evaluation Configuration
# ============================================================================
CHECKPOINT="${CHECKPOINT:-${RUN_ROOT}/runs/latest/checkpoint_best.pth}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/exp1_lidar.yaml}"
if [[ "${CONFIG}" != /* ]]; then
    CONFIG="${PROJECT_ROOT}/${CONFIG}"
fi
NO_GRAPH="${NO_GRAPH:-0}"  # Set to 1 if evaluating model trained without graph
USE_OFFICIAL="${USE_OFFICIAL:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NO_CAMERA="${NO_CAMERA:-0}"

OUTPUT_DIR="${RUN_ROOT}/eval/slurm_${SLURM_JOB_ID}"
mkdir -p "${OUTPUT_DIR}"

# ============================================================================
# Run Evaluation
# ============================================================================
echo "=============================================="
echo "BEVFusion + Graph Evaluation"
echo "=============================================="
echo "  Job ID:      ${SLURM_JOB_ID}"
echo "  Checkpoint:  ${CHECKPOINT}"
echo "  Config:      ${CONFIG}"
echo "  Graph:       $([ "${NO_GRAPH}" == "1" ] && echo "DISABLED" || echo "ENABLED")"
echo "  Output:      ${OUTPUT_DIR}"
echo "=============================================="

EVAL_ARGS=(
    "${PROJECT_ROOT}/evaluate.py"
    --config "${CONFIG}"
    --checkpoint "${CHECKPOINT}"
    --output-dir "${OUTPUT_DIR}"
    --batch-size "${BATCH_SIZE}"
)

if [[ "${NO_GRAPH}" == "1" ]]; then
    EVAL_ARGS+=(--no-graph)
fi
if [[ "${NO_CAMERA}" == "1" ]]; then
    EVAL_ARGS+=(--no-camera)
fi

if [[ "${USE_OFFICIAL}" == "1" ]]; then
    EVAL_ARGS+=(--official)
fi

"${PY}" "${EVAL_ARGS[@]}" 2>&1 | tee "${OUTPUT_DIR}/eval.log"

echo "=============================================="
echo "Evaluation completed"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=============================================="
