#!/usr/bin/env bash
set -euo pipefail

IMAGE_ROOT="${IMAGE_ROOT:-input/Stent-Contrast-unique-view}"
DCM_ROOT="${DCM_ROOT:-input/stent_split_dcm_unique_view}"
GLOBAL2_ROOT="${GLOBAL2_ROOT:-outputs/global_2_study_patient_retrieval_unique_view}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/global_4_1_cluster_anchoring_attribution_unique_view}"
LOG_FILE="${LOG_FILE:-run_gpu1.log}"
DEVICE="${DEVICE:-cuda}"
PROBE_SEEDS_STR="${PROBE_SEEDS_STR:-11 22 33}"
TARGETS_STR="${TARGETS_STR:-patient study}"
KNN_K_STR="${KNN_K_STR:-5 10 20}"
CLUSTER_K_STR="${CLUSTER_K_STR:-8 12 16}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-256}"
SEED="${SEED:-42}"
MAX_IMAGES_ARG=()

if [[ -n "${MAX_IMAGES:-}" ]]; then
  MAX_IMAGES_ARG=(--max-images "${MAX_IMAGES}")
fi

read -r -a PROBE_SEEDS <<< "${PROBE_SEEDS_STR}"
read -r -a TARGETS <<< "${TARGETS_STR}"
read -r -a KNN_K <<< "${KNN_K_STR}"
read -r -a CLUSTER_K <<< "${CLUSTER_K_STR}"

conda run --no-capture-output -n dinov3_stack env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
  python scripts/analysis/global_4_1_cluster_anchoring_attribution.py \
  --global2-root "${GLOBAL2_ROOT}" \
  --image-root "${IMAGE_ROOT}" \
  --dcm-root "${DCM_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --log-file "${LOG_FILE}" \
  --device "${DEVICE}" \
  --probe-seeds "${PROBE_SEEDS[@]}" \
  --targets "${TARGETS[@]}" \
  --knn-k "${KNN_K[@]}" \
  --cluster-k "${CLUSTER_K[@]}" \
  --probe-batch-size "${PROBE_BATCH_SIZE}" \
  --seed "${SEED}" \
  "${MAX_IMAGES_ARG[@]}"
