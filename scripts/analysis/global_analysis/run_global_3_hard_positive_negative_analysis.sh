#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${CONDA_ENV:-dinov3_stack}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

GLOBAL2_ROOT="${GLOBAL2_ROOT:-outputs/global_2_study_patient_retrieval_unique_view}"
IMAGE_ROOT="${IMAGE_ROOT:-input/Stent-Contrast-unique-view}"
DCM_ROOT="${DCM_ROOT:-input/stent_split_dcm_unique_view}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/global_3_hard_positive_negative_analysis_unique_view}"
DEVICE="${DEVICE:-cuda}"

PROBE_SEEDS_STR="${PROBE_SEEDS_STR:-11 22 33}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-256}"
VIEW_BINS_STR="${VIEW_BINS_STR:-5 20}"
NUM_EXAMPLES_PER_SUBTYPE="${NUM_EXAMPLES_PER_SUBTYPE:-50}"
MAX_IMAGES="${MAX_IMAGES:-}"
SEED="${SEED:-42}"
LOG_FILE="${LOG_FILE:-run_gpu${CUDA_VISIBLE_DEVICES}.log}"

read -r -a PROBE_SEEDS <<< "${PROBE_SEEDS_STR}"
read -r -a VIEW_BINS <<< "${VIEW_BINS_STR}"

echo "[run_global_3_hard_positive_negative_analysis] ROOT_DIR=${ROOT_DIR}"
echo "[run_global_3_hard_positive_negative_analysis] CONDA_ENV=${CONDA_ENV} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[run_global_3_hard_positive_negative_analysis] GLOBAL2_ROOT=${GLOBAL2_ROOT}"
echo "[run_global_3_hard_positive_negative_analysis] OUTPUT_ROOT=${OUTPUT_ROOT}"

CMD=(
  conda run --no-capture-output -n "${CONDA_ENV}"
  env
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
  PYTHONUNBUFFERED="${PYTHONUNBUFFERED}"
  python scripts/analysis/global_3_hard_positive_negative_analysis.py
  --global2-root "${GLOBAL2_ROOT}"
  --image-root "${IMAGE_ROOT}"
  --dcm-root "${DCM_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --log-file "${LOG_FILE}"
  --device "${DEVICE}"
  --probe-batch-size "${PROBE_BATCH_SIZE}"
  --num-examples-per-subtype "${NUM_EXAMPLES_PER_SUBTYPE}"
  --seed "${SEED}"
  --probe-seeds
)

CMD+=("${PROBE_SEEDS[@]}")
CMD+=(--view-bins)
CMD+=("${VIEW_BINS[@]}")

if [[ -n "${MAX_IMAGES}" ]]; then
  CMD+=(--max-images "${MAX_IMAGES}")
fi

CMD+=("$@")

printf '[run_global_3_hard_positive_negative_analysis] CMD:'
printf ' %q' "${CMD[@]}"
printf '\n'

exec "${CMD[@]}"
