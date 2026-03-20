#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${CONDA_ENV:-dinov3_stack}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-1}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

BASE_IMAGE_ROOT="${BASE_IMAGE_ROOT:-input/Stent-Contrast-unique-view}"
DCM_ROOT="${DCM_ROOT:-input/stent_split_dcm_unique_view}"
LABEL_JSON="${LABEL_JSON:-input/frames_prediction.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/global_4_2_same_dicom_retrieval_unique_view}"
DEVICE="${DEVICE:-cuda}"
MAX_IMAGES_PER_SPLIT="${MAX_IMAGES_PER_SPLIT:-}"
SEED="${SEED:-42}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

DEFAULT_DATASET_ROOT="input/Stent-Contrast-same-dicom-unique-view"
if [[ -z "${DATASET_ROOT:-}" ]]; then
  if [[ -n "${MAX_IMAGES_PER_SPLIT}" ]]; then
    DATASET_ROOT="${OUTPUT_ROOT}/derived_dataset"
  else
    DATASET_ROOT="${DEFAULT_DATASET_ROOT}"
  fi
fi
DATASET_MANIFEST="${DATASET_MANIFEST:-${DATASET_ROOT}/manifest_same_dicom_master.csv}"
BUILDER_LOG_FILE="${BUILDER_LOG_FILE:-build_same_dicom_dataset.log}"

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
SUPCON_TEMPERATURE="${SUPCON_TEMPERATURE:-0.07}"
STRICT_DETERMINISTIC="${STRICT_DETERMINISTIC:-0}"
CACHE_FEATURES="${CACHE_FEATURES:-1}"
LOG_FILE="${LOG_FILE:-run_gpu${CUDA_VISIBLE_DEVICES}.log}"

read -r -a PROBE_SEEDS <<< "${PROBE_SEEDS_STR}"
read -r -a PROBE_LR_GRID <<< "${PROBE_LR_GRID_STR}"

BUILDER_CMD=(
  conda run --no-capture-output -n "${CONDA_ENV}"
  env
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
  PYTHONUNBUFFERED="${PYTHONUNBUFFERED}"
  python scripts/data/build_stent_contrast_same_dicom_unique_view.py
  --base-image-root "${BASE_IMAGE_ROOT}"
  --dcm-root "${DCM_ROOT}"
  --label-json "${LABEL_JSON}"
  --output-root "${DATASET_ROOT}"
  --log-file "${BUILDER_LOG_FILE}"
  --seed "${SEED}"
)

if [[ -n "${MAX_IMAGES_PER_SPLIT}" ]]; then
  BUILDER_CMD+=(--max-images-per-split "${MAX_IMAGES_PER_SPLIT}")
fi

if [[ "${SKIP_EXISTING}" == "1" ]]; then
  BUILDER_CMD+=(--skip-existing)
fi

ANALYSIS_CMD=(
  conda run --no-capture-output -n "${CONDA_ENV}"
  env
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
  PYTHONUNBUFFERED="${PYTHONUNBUFFERED}"
  python scripts/analysis/global_4_2_same_dicom_retrieval.py
  --image-root "${DATASET_ROOT}"
  --dataset-manifest "${DATASET_MANIFEST}"
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
  --supcon-temperature "${SUPCON_TEMPERATURE}"
  --seed "${SEED}"
  --probe-seeds
)

ANALYSIS_CMD+=("${PROBE_SEEDS[@]}")
ANALYSIS_CMD+=(--probe-lr-grid)
ANALYSIS_CMD+=("${PROBE_LR_GRID[@]}")

if [[ "${STRICT_DETERMINISTIC}" == "1" ]]; then
  ANALYSIS_CMD+=(--strict-deterministic)
fi

if [[ "${CACHE_FEATURES}" == "1" ]]; then
  ANALYSIS_CMD+=(--cache-features)
fi

ANALYSIS_CMD+=("$@")

echo "[run_global_4_2_same_dicom_retrieval] ROOT_DIR=${ROOT_DIR}"
echo "[run_global_4_2_same_dicom_retrieval] CONDA_ENV=${CONDA_ENV} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[run_global_4_2_same_dicom_retrieval] DATASET_ROOT=${DATASET_ROOT}"
echo "[run_global_4_2_same_dicom_retrieval] OUTPUT_ROOT=${OUTPUT_ROOT}"

printf '[run_global_4_2_same_dicom_retrieval] BUILDER_CMD:'
printf ' %q' "${BUILDER_CMD[@]}"
printf '\n'
"${BUILDER_CMD[@]}"

printf '[run_global_4_2_same_dicom_retrieval] ANALYSIS_CMD:'
printf ' %q' "${ANALYSIS_CMD[@]}"
printf '\n'
exec "${ANALYSIS_CMD[@]}"
