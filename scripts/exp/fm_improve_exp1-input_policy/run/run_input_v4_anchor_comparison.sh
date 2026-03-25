#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-dinov3_stack}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED

EXP_ROOT="${EXP_ROOT:-outputs/fm_improve_exp1-input_policy/input_v4_border_suppression/downstream_only}"
DCM_ROOT="${DCM_ROOT:-input/stent_split_dcm_unique_view}"
TASKS_STR="${TASKS_STR:-ga4_1}"
DEVICE="${DEVICE:-cuda}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-256}"
PROBE_SEEDS_STR="${PROBE_SEEDS_STR:-11 22 33}"
MAX_IMAGES="${MAX_IMAGES:-0}"
SEED="${SEED:-42}"
SKIP_SUMMARY="${SKIP_SUMMARY:-0}"
OVERWRITE="${OVERWRITE:-0}"

CMD=(
  conda run --no-capture-output -n "${CONDA_ENV}" python -u
  scripts/exp/fm_improve_exp1-input_policy/analysis/run_input_v4_anchor_comparison.py
  --exp-root "${EXP_ROOT}"
  --dcm-root "${DCM_ROOT}"
  --tasks ${TASKS_STR}
  --device "${DEVICE}"
  --probe-batch-size "${PROBE_BATCH_SIZE}"
  --probe-seeds-str "${PROBE_SEEDS_STR}"
  --seed "${SEED}"
)

if [[ "${MAX_IMAGES}" != "0" ]]; then
  CMD+=(--max-images "${MAX_IMAGES}")
fi
if [[ "${SKIP_SUMMARY}" == "1" ]]; then
  CMD+=(--skip-summary)
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  CMD+=(--overwrite)
fi

echo "ROOT_DIR=${ROOT_DIR}"
echo "EXP_ROOT=${EXP_ROOT}"
printf 'CMD=%q ' "${CMD[@]}"
printf '
'

"${CMD[@]}"
