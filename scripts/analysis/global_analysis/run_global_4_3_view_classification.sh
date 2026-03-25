#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${CONDA_ENV:-dinov3_stack}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-1}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

IMAGE_ROOT="${IMAGE_ROOT:-input/Stent-Contrast-unique-view}"
DCM_ROOT="${DCM_ROOT:-input/stent_split_dcm_unique_view}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/global_4_3_view_classification_unique_view}"
DEVICE="${DEVICE:-cuda}"
MAX_IMAGES_PER_SPLIT="${MAX_IMAGES_PER_SPLIT:-}"
SEED="${SEED:-42}"

MODEL_NAME="${MODEL_NAME:-dinov3_vits16}"
REPO_DIR="${REPO_DIR:-dinov3}"
IMAGENET_CKPT="${IMAGENET_CKPT:-dinov3/output/a6000/1_pretrain/dinov3_vits16/2_imagenet1k/3_stage3_high_res_adapt/eval/training_29999/teacher_checkpoint.pth}"
CAG_CKPT="${CAG_CKPT:-dinov3/output/a6000/1_pretrain/dinov3_vits16/3_cagcontfm3m/3_stage3_high_res_adapt/eval/training_29999/teacher_checkpoint.pth}"

RESIZE_SIZE="${RESIZE_SIZE:-480}"
CENTER_CROP_SIZE="${CENTER_CROP_SIZE:-448}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-128}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PROBE_SEEDS_STR="${PROBE_SEEDS_STR:-11 22 33}"
PROBE_LR_GRID_STR="${PROBE_LR_GRID_STR:-1e-2 3e-3 1e-3}"
PROBE_MAX_EPOCH="${PROBE_MAX_EPOCH:-200}"
PROBE_PATIENCE="${PROBE_PATIENCE:-20}"
PROBE_MIN_DELTA="${PROBE_MIN_DELTA:-0.0}"
PROBE_WEIGHT_DECAY="${PROBE_WEIGHT_DECAY:-1e-4}"
STRICT_DETERMINISTIC="${STRICT_DETERMINISTIC:-0}"
CACHE_FEATURES="${CACHE_FEATURES:-1}"
LOG_FILE="${LOG_FILE:-run_gpu${CUDA_VISIBLE_DEVICES}.log}"

read -r -a PROBE_SEEDS <<< "${PROBE_SEEDS_STR}"
read -r -a PROBE_LR_GRID <<< "${PROBE_LR_GRID_STR}"

CMD=(
  conda run --no-capture-output -n "${CONDA_ENV}"
  env
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
  PYTHONUNBUFFERED="${PYTHONUNBUFFERED}"
  python scripts/analysis/global_4_3_view_classification.py
  --image-root "${IMAGE_ROOT}"
  --dcm-root "${DCM_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --log-file "${LOG_FILE}"
  --imagenet-ckpt "${IMAGENET_CKPT}"
  --cag-ckpt "${CAG_CKPT}"
  --model-name "${MODEL_NAME}"
  --repo-dir "${REPO_DIR}"
  --device "${DEVICE}"
  --resize-size "${RESIZE_SIZE}"
  --center-crop-size "${CENTER_CROP_SIZE}"
  --feature-batch-size "${FEATURE_BATCH_SIZE}"
  --probe-batch-size "${PROBE_BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --probe-max-epoch "${PROBE_MAX_EPOCH}"
  --probe-patience "${PROBE_PATIENCE}"
  --probe-min-delta "${PROBE_MIN_DELTA}"
  --probe-weight-decay "${PROBE_WEIGHT_DECAY}"
  --seed "${SEED}"
  --probe-seeds
)
CMD+=("${PROBE_SEEDS[@]}")
CMD+=(--probe-lr-grid)
CMD+=("${PROBE_LR_GRID[@]}")

if [[ -n "${MAX_IMAGES_PER_SPLIT}" ]]; then
  CMD+=(--max-images-per-split "${MAX_IMAGES_PER_SPLIT}")
fi
if [[ "${STRICT_DETERMINISTIC}" == "1" ]]; then
  CMD+=(--strict-deterministic)
fi
if [[ "${CACHE_FEATURES}" == "1" ]]; then
  CMD+=(--cache-features)
fi
CMD+=("$@")

echo "[run_global_4_3_view_classification] ROOT_DIR=${ROOT_DIR}"
echo "[run_global_4_3_view_classification] OUTPUT_ROOT=${OUTPUT_ROOT}"
printf '[run_global_4_3_view_classification] CMD:'
printf ' %q' "${CMD[@]}"
printf '\n'
exec "${CMD[@]}"
