#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-dinov3_stack}"
IMAGE_ROOT="${IMAGE_ROOT:-input/Stent-Contrast-unique-view}"
DICOM_ROOT="${DICOM_ROOT:-input/stent_split_dcm_unique_view}"
OUTPUT_ROOT="${OUTPUT_ROOT:-input/global_analysis_5_2_integris_allura_flat_detector_unique_view_subset}"
TARGET_MODEL_NAME="${TARGET_MODEL_NAME:-INTEGRIS Allura Flat Detector}"
ANALYSIS_TITLE="${ANALYSIS_TITLE:-Global Analysis 5-2 Single-Device Subset}"
SUMMARY_PREFIX="${SUMMARY_PREFIX:-summary_global_5_2}"
MARKDOWN_NAME="${MARKDOWN_NAME:-analysis_global_5_2_integris_allura_flat_detector_unique_view_subset.md}"
LOG_PREFIX="${LOG_PREFIX:-build_global_5_2_integris_allura_flat_detector_unique_view_subset}"
LOG_EVERY="${LOG_EVERY:-200}"
OVERWRITE="${OVERWRITE:-0}"

CMD=(
  conda run --no-capture-output -n "${CONDA_ENV}" python -u
  scripts/analysis/global_analysis/build_global_5_single_device_unique_view_subset.py
  --image-root "${IMAGE_ROOT}"
  --dicom-root "${DICOM_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --target-model-name "${TARGET_MODEL_NAME}"
  --analysis-title "${ANALYSIS_TITLE}"
  --summary-prefix "${SUMMARY_PREFIX}"
  --markdown-name "${MARKDOWN_NAME}"
  --log-prefix "${LOG_PREFIX}"
  --log-every "${LOG_EVERY}"
)

if [[ "${OVERWRITE}" == "1" ]]; then
  CMD+=(--overwrite)
fi

echo "ROOT_DIR=${ROOT_DIR}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
printf 'CMD=%q ' "${CMD[@]}"
printf '\n'

export PYTHONUNBUFFERED=1
"${CMD[@]}"
