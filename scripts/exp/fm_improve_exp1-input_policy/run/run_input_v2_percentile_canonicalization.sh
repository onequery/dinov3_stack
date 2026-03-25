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

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/fm_improve_exp1-input_policy/input_v2_percentile_canonicalization/downstream_only}"
UNIQUE_VIEW_ROOT="${UNIQUE_VIEW_ROOT:-input/Stent-Contrast-unique-view}"
SAME_DICOM_ROOT="${SAME_DICOM_ROOT:-input/Stent-Contrast-same-dicom-unique-view}"
SAME_DICOM_MANIFEST="${SAME_DICOM_MANIFEST:-input/Stent-Contrast-same-dicom-unique-view/manifest_same_dicom_master.csv}"
DEVICE="${DEVICE:-cuda}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-128}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-256}"
PROBE_MAX_EPOCH="${PROBE_MAX_EPOCH:-200}"
PROBE_PATIENCE="${PROBE_PATIENCE:-20}"
PROBE_SEEDS_STR="${PROBE_SEEDS_STR:-11 22 33}"
PROBE_LR_GRID_STR="${PROBE_LR_GRID_STR:-0.001 0.003 0.01}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_IMAGES_PER_SPLIT="${MAX_IMAGES_PER_SPLIT:-0}"
CACHE_FEATURES="${CACHE_FEATURES:-1}"
OVERWRITE="${OVERWRITE:-0}"
STRICT_DETERMINISTIC="${STRICT_DETERMINISTIC:-0}"
SKIP_CACHE="${SKIP_CACHE:-0}"
SKIP_BENCHMARK="${SKIP_BENCHMARK:-0}"
SKIP_SUMMARY="${SKIP_SUMMARY:-0}"
SKIP_PLOT="${SKIP_PLOT:-0}"
PASS_OVERWRITE="${PASS_OVERWRITE:-0}"

CMD=(
  conda run --no-capture-output -n "${CONDA_ENV}" python -u
  scripts/exp/fm_improve_exp1-input_policy/analysis/run_input_v2_percentile_canonicalization.py
  --output-root "${OUTPUT_ROOT}"
  --unique-view-root "${UNIQUE_VIEW_ROOT}"
  --same-dicom-root "${SAME_DICOM_ROOT}"
  --same-dicom-manifest "${SAME_DICOM_MANIFEST}"
  --device "${DEVICE}"
  --feature-batch-size "${FEATURE_BATCH_SIZE}"
  --probe-batch-size "${PROBE_BATCH_SIZE}"
  --probe-max-epoch "${PROBE_MAX_EPOCH}"
  --probe-patience "${PROBE_PATIENCE}"
  --probe-seeds-str "${PROBE_SEEDS_STR}"
  --probe-lr-grid-str "${PROBE_LR_GRID_STR}"
  --num-workers "${NUM_WORKERS}"
  --max-images-per-split "${MAX_IMAGES_PER_SPLIT}"
)

if [[ "${CACHE_FEATURES}" == "1" ]]; then
  CMD+=(--cache-features)
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  CMD+=(--overwrite)
fi
if [[ "${STRICT_DETERMINISTIC}" == "1" ]]; then
  CMD+=(--strict-deterministic)
fi
if [[ "${SKIP_CACHE}" == "1" ]]; then
  CMD+=(--skip-cache)
fi
if [[ "${SKIP_BENCHMARK}" == "1" ]]; then
  CMD+=(--skip-benchmark)
fi
if [[ "${SKIP_SUMMARY}" == "1" ]]; then
  CMD+=(--skip-summary)
fi
if [[ "${SKIP_PLOT}" == "1" ]]; then
  CMD+=(--skip-plot)
fi
if [[ "${PASS_OVERWRITE}" == "1" ]]; then
  CMD+=(--pass-overwrite)
fi

printf 'ROOT_DIR=%s\n' "${ROOT_DIR}"
printf 'OUTPUT_ROOT=%s\n' "${OUTPUT_ROOT}"
printf 'CMD=%q ' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
