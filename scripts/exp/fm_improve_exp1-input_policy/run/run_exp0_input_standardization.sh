#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${CONDA_ENV:-dinov3_stack}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

UNIQUE_VIEW_ROOT="${UNIQUE_VIEW_ROOT:-input/Stent-Contrast-unique-view}"
SAME_DICOM_ROOT="${SAME_DICOM_ROOT:-input/Stent-Contrast-same-dicom-unique-view}"
DCM_ROOT="${DCM_ROOT:-input/stent_split_dcm_unique_view}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/exp1_fm_improve/ablation/exp0_input_standardization}"
DEVICE="${DEVICE:-cuda}"
VARIANTS_STR="${VARIANTS_STR:-norm_v1 norm_v2 norm_v3}"
MAX_IMAGES_PER_SPLIT="${MAX_IMAGES_PER_SPLIT:-}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-128}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
RESIZE_SIZE="${RESIZE_SIZE:-480}"
CENTER_CROP_SIZE="${CENTER_CROP_SIZE:-448}"
PROBE_SEEDS_STR="${PROBE_SEEDS_STR:-11 22 33}"
PROBE_LR_GRID_STR="${PROBE_LR_GRID_STR:-1e-2 3e-3 1e-3}"
PROBE_MAX_EPOCH="${PROBE_MAX_EPOCH:-200}"
PROBE_PATIENCE="${PROBE_PATIENCE:-20}"
PROBE_MIN_DELTA="${PROBE_MIN_DELTA:-0.0}"
PROBE_WEIGHT_DECAY="${PROBE_WEIGHT_DECAY:-1e-4}"
SUPCON_TEMPERATURE="${SUPCON_TEMPERATURE:-0.07}"
SEED="${SEED:-42}"
STRICT_DETERMINISTIC="${STRICT_DETERMINISTIC:-0}"
CACHE_FEATURES="${CACHE_FEATURES:-1}"
OVERWRITE="${OVERWRITE:-1}"
SKIP_CACHE="${SKIP_CACHE:-0}"
SKIP_BENCHMARK="${SKIP_BENCHMARK:-0}"
SKIP_ANCHORING="${SKIP_ANCHORING:-0}"
SKIP_SUMMARY="${SKIP_SUMMARY:-0}"

read -r -a VARIANTS <<< "${VARIANTS_STR}"
read -r -a PROBE_SEEDS <<< "${PROBE_SEEDS_STR}"
read -r -a PROBE_LR_GRID <<< "${PROBE_LR_GRID_STR}"

CMD=(
  conda run --no-capture-output -n "${CONDA_ENV}"
  env
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
  PYTHONUNBUFFERED="${PYTHONUNBUFFERED}"
  python scripts/exp/exp1_fm_improve/analysis/run_exp0_input_standardization.py
  --variants
)
CMD+=("${VARIANTS[@]}")
CMD+=(
  --unique-view-root "${UNIQUE_VIEW_ROOT}"
  --same-dicom-root "${SAME_DICOM_ROOT}"
  --dcm-root "${DCM_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --device "${DEVICE}"
  --feature-batch-size "${FEATURE_BATCH_SIZE}"
  --probe-batch-size "${PROBE_BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --resize-size "${RESIZE_SIZE}"
  --center-crop-size "${CENTER_CROP_SIZE}"
  --probe-seeds
)
CMD+=("${PROBE_SEEDS[@]}")
CMD+=(--probe-lr-grid)
CMD+=("${PROBE_LR_GRID[@]}")
CMD+=(
  --probe-max-epoch "${PROBE_MAX_EPOCH}"
  --probe-patience "${PROBE_PATIENCE}"
  --probe-min-delta "${PROBE_MIN_DELTA}"
  --probe-weight-decay "${PROBE_WEIGHT_DECAY}"
  --supcon-temperature "${SUPCON_TEMPERATURE}"
  --seed "${SEED}"
)

if [[ -n "${MAX_IMAGES_PER_SPLIT}" ]]; then
  CMD+=(--max-images-per-split "${MAX_IMAGES_PER_SPLIT}")
fi
if [[ "${STRICT_DETERMINISTIC}" == "1" ]]; then
  CMD+=(--strict-deterministic)
fi
if [[ "${CACHE_FEATURES}" == "1" ]]; then
  CMD+=(--cache-features)
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  CMD+=(--overwrite)
fi
if [[ "${SKIP_CACHE}" == "1" ]]; then
  CMD+=(--skip-cache)
fi
if [[ "${SKIP_BENCHMARK}" == "1" ]]; then
  CMD+=(--skip-benchmark)
fi
if [[ "${SKIP_ANCHORING}" == "1" ]]; then
  CMD+=(--skip-anchoring)
fi
if [[ "${SKIP_SUMMARY}" == "1" ]]; then
  CMD+=(--skip-summary)
fi
CMD+=("$@")

echo "[run_exp0_input_standardization] ROOT_DIR=${ROOT_DIR}"
echo "[run_exp0_input_standardization] OUTPUT_ROOT=${OUTPUT_ROOT}"
printf '[run_exp0_input_standardization] CMD:'
printf ' %q' "${CMD[@]}"
printf '\n'
exec "${CMD[@]}"
