#!/bin/bash
#SBATCH --job-name=bevfusion_vis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --chdir=/scratch/merdem22
#SBATCH --output=/scratch/merdem22/slurm_%x_%j.out
#SBATCH --error=/scratch/merdem22/slurm_%x_%j.err

set -euo pipefail

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

PROJECT_ROOT="${PROJECT_ROOT:-/home/merdem22/nuscenes_fusion}"
WORK_DIR="${WORK_DIR:-/scratch/merdem22/nuscenes_fusion/work/${SLURM_JOB_ID}}"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"
export PYTHONPATH="${PROJECT_ROOT}"

CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/exp1_lidar.yaml}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT=/scratch/.../checkpoint_best.pth}"
DATA_ROOT="${DATA_ROOT:-/datasets/nuscenes}"
DATA_VERSION="${DATA_VERSION:-v1.0-trainval}"
DATA_SPLIT="${DATA_SPLIT:-val}"

NUM_CANDIDATES="${NUM_CANDIDATES:-400}"
OUTDIR="${OUTDIR:-/scratch/merdem22/nuscenes_fusion/outputs/vis_${SLURM_JOB_ID}}"
SAVE_CAMERAS="${SAVE_CAMERAS:-1}"
CAM_SCALE="${CAM_SCALE:-1.0}"
CAM_SCORE_THRESH="${CAM_SCORE_THRESH:-0.2}"
CAM_MAX_PRED="${CAM_MAX_PRED:-30}"

SCORE_THRESH="${SCORE_THRESH:-0.05}"
EVAL_SCORE_THRESH="${EVAL_SCORE_THRESH:-0.18}"
NMS_THRESH="${NMS_THRESH:-0.2}"
MAX_DETS="${MAX_DETS:-500}"

DEVICE="${DEVICE:-cuda}"

SAVE_MODE="${SAVE_MODE:-demo}"
SAVE_TOPK="${SAVE_TOPK:-3}"
MIN_F1="${MIN_F1:-0.08}"
MIN_PRECISION="${MIN_PRECISION:-0.08}"
MIN_RECALL="${MIN_RECALL:-0.20}"
PRED_MODE="${PRED_MODE:-tp}"
GT_MODE="${GT_MODE:-all}"
REQUIRE_CLASS="${REQUIRE_CLASS:-}"
GOOD_PRED_MODE="${GOOD_PRED_MODE:-tp}"
QUESTIONABLE_PRED_MODE="${QUESTIONABLE_PRED_MODE:-all}"
QUESTIONABLE_TARGET="${QUESTIONABLE_TARGET:-low_precision}"

mkdir -p "${OUTDIR}"

ARGS=(
  "${PROJECT_ROOT}/scripts/visualize_samples.py"
  --config "${CONFIG}"
  --checkpoint "${CHECKPOINT}"
  --root "${DATA_ROOT}"
  --version "${DATA_VERSION}"
  --split "${DATA_SPLIT}"
  --num-candidates "${NUM_CANDIDATES}"
  --outdir "${OUTDIR}"
  --device "${DEVICE}"
  --score-thresh "${SCORE_THRESH}"
  --eval-score-thresh "${EVAL_SCORE_THRESH}"
  --nms-thresh "${NMS_THRESH}"
  --max-dets "${MAX_DETS}"
  --cam-score-thresh "${CAM_SCORE_THRESH}"
  --cam-max-pred "${CAM_MAX_PRED}"
  --cam-scale "${CAM_SCALE}"
  --save-mode "${SAVE_MODE}"
  --save-topk "${SAVE_TOPK}"
  --min-f1 "${MIN_F1}"
  --min-precision "${MIN_PRECISION}"
  --min-recall "${MIN_RECALL}"
  --pred-mode "${PRED_MODE}"
  --gt-mode "${GT_MODE}"
  --good-pred-mode "${GOOD_PRED_MODE}"
  --questionable-pred-mode "${QUESTIONABLE_PRED_MODE}"
  --questionable-target "${QUESTIONABLE_TARGET}"
)

if [[ -n "${REQUIRE_CLASS}" ]]; then
  ARGS+=(--require-class "${REQUIRE_CLASS}")
fi

if [[ "${SAVE_CAMERAS}" == "1" ]]; then
  ARGS+=(--save-cameras)
fi

echo "=============================================="
echo "BEVFusion Visualization"
echo "=============================================="
echo "  Job ID:     ${SLURM_JOB_ID}"
echo "  Config:     ${CONFIG}"
echo "  Checkpoint: ${CHECKPOINT}"
echo "  Data:       ${DATA_ROOT} (${DATA_VERSION}/${DATA_SPLIT})"
echo "  Candidates: ${NUM_CANDIDATES}"
echo "  Output:     ${OUTDIR}"
echo "=============================================="

"${PY}" "${ARGS[@]}" 2>&1 | tee "${OUTDIR}/vis.log"

echo "Saved visualizations to: ${OUTDIR}"
