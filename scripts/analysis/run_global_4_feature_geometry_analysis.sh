#!/usr/bin/env bash
set -euo pipefail

IMAGE_ROOT="${IMAGE_ROOT:-input/Stent-Contrast-unique-view}"
DCM_ROOT="${DCM_ROOT:-input/stent_split_dcm_unique_view}"
GLOBAL2_ROOT="${GLOBAL2_ROOT:-outputs/global_2_study_patient_retrieval_unique_view}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/global_4_feature_geometry_analysis_unique_view}"
LOG_FILE="${LOG_FILE:-run_gpu1.log}"
DEVICE="${DEVICE:-cuda}"
PROBE_SEEDS_STR="${PROBE_SEEDS_STR:-11 22 33}"
NEAR_VIEW_THRESHOLD_DEG="${NEAR_VIEW_THRESHOLD_DEG:-5}"
MID_VIEW_THRESHOLD_DEG="${MID_VIEW_THRESHOLD_DEG:-20}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-256}"
SEED="${SEED:-42}"
MAX_IMAGES_ARG=()
SKIP_UMAP_ARG=()

if [[ -n "${MAX_IMAGES:-}" ]]; then
  MAX_IMAGES_ARG=(--max-images "${MAX_IMAGES}")
fi
if [[ "${SKIP_UMAP:-0}" == "1" ]]; then
  SKIP_UMAP_ARG=(--skip-umap)
fi

read -r -a PROBE_SEEDS <<< "${PROBE_SEEDS_STR}"

conda run --no-capture-output -n dinov3_stack env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
  python scripts/analysis/global_4_feature_geometry_analysis.py \
  --global2-root "${GLOBAL2_ROOT}" \
  --image-root "${IMAGE_ROOT}" \
  --dcm-root "${DCM_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --log-file "${LOG_FILE}" \
  --device "${DEVICE}" \
  --probe-seeds "${PROBE_SEEDS[@]}" \
  --near-view-threshold-deg "${NEAR_VIEW_THRESHOLD_DEG}" \
  --mid-view-threshold-deg "${MID_VIEW_THRESHOLD_DEG}" \
  --probe-batch-size "${PROBE_BATCH_SIZE}" \
  --seed "${SEED}" \
  "${MAX_IMAGES_ARG[@]}" \
  "${SKIP_UMAP_ARG[@]}"
