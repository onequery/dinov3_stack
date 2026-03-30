#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

IMAGE_ROOT="${IMAGE_ROOT:-input/Stent-Contrast-unique-view}"
DCM_ROOT="${DCM_ROOT:-input/stent_split_dcm_unique_view}"
OUTPUT_ROOT="${OUTPUT_ROOT:-input/contrast_benchmark/1_global/1_view_classification}"
OVERWRITE="${OVERWRITE:-1}"
LOG_EVERY="${LOG_EVERY:-500}"

CMD=(conda run --no-capture-output -n dinov3_stack python scripts/data/contrast_benchmark/build_view_classification_dataset.py
  --image-root "$IMAGE_ROOT"
  --dcm-root "$DCM_ROOT"
  --output-root "$OUTPUT_ROOT"
  --log-every "$LOG_EVERY"
)
if [[ "$OVERWRITE" == "1" ]]; then
  CMD+=(--overwrite)
fi

echo "ROOT_DIR=$ROOT_DIR"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
printf 'CMD='
printf '%q ' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
