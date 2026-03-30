#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-dinov3_stack}"
IMAGE_ROOT="${IMAGE_ROOT:-/mnt/nas/snubhcvc/project/cag_fm/pretrain/datasets/images}"
DICOM_ROOT="${DICOM_ROOT:-/mnt/nas/snubhcvc/raw/cpacs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/cag_pretrain_metadata_stats}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LOG_EVERY="${LOG_EVERY:-200}"
MAX_IMAGES_PER_SUBTASK="${MAX_IMAGES_PER_SUBTASK:-0}"
REFRESH_ENUMERATION_CACHE="${REFRESH_ENUMERATION_CACHE:-0}"

CMD=(
  conda run --no-capture-output -n "${CONDA_ENV}" python -u
  scripts/analysis/pretrain_metadata/analyze_cag_pretrain_metadata.py
  --image-root "${IMAGE_ROOT}"
  --dicom-root "${DICOM_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --splits train val test
  --num-workers "${NUM_WORKERS}"
  --log-every "${LOG_EVERY}"
  --resume
  --max-images-per-subtask "${MAX_IMAGES_PER_SUBTASK}"
)

if [[ "${REFRESH_ENUMERATION_CACHE}" == "1" ]]; then
  CMD+=(--refresh-enumeration-cache)
fi

echo "ROOT_DIR=${ROOT_DIR}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
printf 'CMD=%q ' "${CMD[@]}"
printf '\n'

export PYTHONUNBUFFERED=1
"${CMD[@]}"
