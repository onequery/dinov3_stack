#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${CONDA_ENV:-dinov3_stack}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/analysis2_rep_analysis/global_5_per_device_patient_retrieval}"
INPUT_FRAME_ROOT="${INPUT_FRAME_ROOT:-input/global_analysis_5_per_device_patient_retrieval}"

CMD=(
  conda run --no-capture-output -n "${CONDA_ENV}"
  env
  PYTHONUNBUFFERED="${PYTHONUNBUFFERED}"
  python -u scripts/analysis/global_analysis/summarize_global_5_per_device_patient_retrieval.py
  --output-root "${OUTPUT_ROOT}"
  --input-frame-root "${INPUT_FRAME_ROOT}"
)
CMD+=("$@")

echo "[run_global_5_per_device_patient_retrieval_summary] ROOT_DIR=${ROOT_DIR}"
echo "[run_global_5_per_device_patient_retrieval_summary] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[run_global_5_per_device_patient_retrieval_summary] INPUT_FRAME_ROOT=${INPUT_FRAME_ROOT}"
printf '[run_global_5_per_device_patient_retrieval_summary] CMD:'
printf ' %q' "${CMD[@]}"
printf '\n'

exec "${CMD[@]}"
