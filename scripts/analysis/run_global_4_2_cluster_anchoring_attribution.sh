#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${CONDA_ENV:-dinov3_stack}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-1}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

GLOBAL2_ROOT="${GLOBAL2_ROOT:-outputs/global_4_2_same_dicom_retrieval_unique_view}"
IMAGE_ROOT="${IMAGE_ROOT:-input/Stent-Contrast-same-dicom-unique-view}"
DCM_ROOT="${DCM_ROOT:-input/stent_split_dcm_unique_view}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/global_4_2_same_dicom_retrieval_unique_view}"
DEVICE="${DEVICE:-cuda}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-256}"
PROBE_SEEDS_STR="${PROBE_SEEDS_STR:-11 22 33}"
KNN_K_STR="${KNN_K_STR:-5 10 20}"
CLUSTER_K_STR="${CLUSTER_K_STR:-8 12 16}"
MAX_IMAGES="${MAX_IMAGES:-}"
SEED="${SEED:-42}"
LOG_FILE="${LOG_FILE:-run_global_4_2_cluster_anchoring.log}"

read -r -a PROBE_SEEDS <<< "${PROBE_SEEDS_STR}"
read -r -a KNN_K <<< "${KNN_K_STR}"
read -r -a CLUSTER_K <<< "${CLUSTER_K_STR}"

CMD=(
  conda run --no-capture-output -n "${CONDA_ENV}"
  env
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
  PYTHONUNBUFFERED="${PYTHONUNBUFFERED}"
  python scripts/analysis/global_4_2_cluster_anchoring_attribution.py
  --global2-root "${GLOBAL2_ROOT}"
  --image-root "${IMAGE_ROOT}"
  --dcm-root "${DCM_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --log-file "${LOG_FILE}"
  --device "${DEVICE}"
  --probe-batch-size "${PROBE_BATCH_SIZE}"
  --seed "${SEED}"
  --probe-seeds
)
CMD+=("${PROBE_SEEDS[@]}")
CMD+=(--targets same_dicom --knn-k)
CMD+=("${KNN_K[@]}")
CMD+=(--cluster-k)
CMD+=("${CLUSTER_K[@]}")
if [[ -n "${MAX_IMAGES}" ]]; then
  CMD+=(--max-images "${MAX_IMAGES}")
fi
CMD+=("$@")

echo "[run_global_4_2_cluster_anchoring_attribution] ROOT_DIR=${ROOT_DIR}"
echo "[run_global_4_2_cluster_anchoring_attribution] OUTPUT_ROOT=${OUTPUT_ROOT}"
printf '[run_global_4_2_cluster_anchoring_attribution] CMD:'
printf ' %q' "${CMD[@]}"
printf '\n'
exec "${CMD[@]}"
