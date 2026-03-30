#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-dinov3_stack}"
STATS_ROOT="${STATS_ROOT:-outputs/cag_pretrain_metadata_stats}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STATS_ROOT}/plots}"
TOP_CATEGORICAL_VALUES="${TOP_CATEGORICAL_VALUES:-40}"
ALL_CATEGORICAL_THRESHOLD="${ALL_CATEGORICAL_THRESHOLD:-50}"
CATEGORICAL_LOG_EVERY_ROWS="${CATEGORICAL_LOG_EVERY_ROWS:-100000}"
SPLITS_STR="${SPLITS_STR:-}"

CMD=(
  conda run --no-capture-output -n "${CONDA_ENV}" python -u
  scripts/analysis/pretrain_metadata/plot_cag_pretrain_metadata_distributions.py
  --stats-root "${STATS_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --top-categorical-values "${TOP_CATEGORICAL_VALUES}"
  --all-categorical-threshold "${ALL_CATEGORICAL_THRESHOLD}"
  --categorical-log-every-rows "${CATEGORICAL_LOG_EVERY_ROWS}"
)

if [[ -n "${SPLITS_STR}" ]]; then
  read -r -a SPLITS_ARR <<< "${SPLITS_STR}"
  CMD+=(--splits "${SPLITS_ARR[@]}")
fi

echo "ROOT_DIR=${ROOT_DIR}"
echo "STATS_ROOT=${STATS_ROOT}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
printf 'CMD=%q ' "${CMD[@]}"
printf '\n'

export PYTHONUNBUFFERED=1
"${CMD[@]}"
