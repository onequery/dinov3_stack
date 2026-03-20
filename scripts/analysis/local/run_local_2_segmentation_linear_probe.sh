#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${CONDA_ENV:-dinov3_stack}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

TRAIN_IMAGES="${TRAIN_IMAGES:-input/MPXA-Seg/train_images}"
TRAIN_MASKS="${TRAIN_MASKS:-input/MPXA-Seg/train_labels}"
VALID_IMAGES="${VALID_IMAGES:-input/MPXA-Seg/valid_images}"
VALID_MASKS="${VALID_MASKS:-input/MPXA-Seg/valid_labels}"
TEST_IMAGES="${TEST_IMAGES:-input/MPXA-Seg/test_images}"
TEST_MASKS="${TEST_MASKS:-input/MPXA-Seg/test_labels}"
SEG_CONFIG="${SEG_CONFIG:-configs_segmentation/mpxa-seg.yaml}"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/local_2_segmentation_linear_probe}"
DEVICE="${DEVICE:-cuda}"
MODEL_NAME="${MODEL_NAME:-dinov3_vits16}"
REPO_DIR="${REPO_DIR:-dinov3}"
IMAGENET_CKPT="${IMAGENET_CKPT:-dinov3/output/a6000/1_pretrain/dinov3_vits16/2_imagenet1k/3_stage3_high_res_adapt/eval/training_29999/teacher_checkpoint.pth}"
CAG_CKPT="${CAG_CKPT:-dinov3/output/a6000/1_pretrain/dinov3_vits16/3_cagcontfm3m/3_stage3_high_res_adapt/eval/training_29999/teacher_checkpoint.pth}"

IMG_W="${IMG_W:-640}"
IMG_H="${IMG_H:-640}"
PATCH_SIZE="${PATCH_SIZE:-16}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-16}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-32}"
MAX_EPOCH="${MAX_EPOCH:-200}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-20}"
EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-0.0}"
LR_GRID_STR="${LR_GRID_STR:-1e-2 3e-3 1e-3}"
read -r -a LR_GRID <<<"${LR_GRID_STR}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-42}"
NUM_EXAMPLE_IMAGES="${NUM_EXAMPLE_IMAGES:-8}"
MAX_IMAGES_PER_SPLIT="${MAX_IMAGES_PER_SPLIT:-}"
LOG_FILE="${LOG_FILE:-run_gpu${CUDA_VISIBLE_DEVICES}.log}"

echo "[run_local_2_segmentation_linear_probe] ROOT_DIR=${ROOT_DIR}"
echo "[run_local_2_segmentation_linear_probe] CONDA_ENV=${CONDA_ENV} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[run_local_2_segmentation_linear_probe] OUTPUT_ROOT=${OUTPUT_ROOT}"

CMD=(
  conda run --no-capture-output -n "${CONDA_ENV}"
  env
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
  PYTHONUNBUFFERED="${PYTHONUNBUFFERED}"
  python scripts/analysis/local_2_segmentation_linear_probe.py
  --train-images "${TRAIN_IMAGES}"
  --train-masks "${TRAIN_MASKS}"
  --valid-images "${VALID_IMAGES}"
  --valid-masks "${VALID_MASKS}"
  --test-images "${TEST_IMAGES}"
  --test-masks "${TEST_MASKS}"
  --seg-config "${SEG_CONFIG}"
  --imagenet-ckpt "${IMAGENET_CKPT}"
  --cag-ckpt "${CAG_CKPT}"
  --output-root "${OUTPUT_ROOT}"
  --log-file "${LOG_FILE}"
  --img-size "${IMG_W}" "${IMG_H}"
  --patch-size "${PATCH_SIZE}"
  --feature-batch-size "${FEATURE_BATCH_SIZE}"
  --probe-batch-size "${PROBE_BATCH_SIZE}"
  --max-epoch "${MAX_EPOCH}"
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE}"
  --early-stopping-min-delta "${EARLY_STOPPING_MIN_DELTA}"
  --device "${DEVICE}"
  --num-workers "${NUM_WORKERS}"
  --seed "${SEED}"
  --num-example-images "${NUM_EXAMPLE_IMAGES}"
  --model-name "${MODEL_NAME}"
  --repo-dir "${REPO_DIR}"
  --cache-features
  --lr-grid "${LR_GRID[@]}"
)

if [[ -n "${MAX_IMAGES_PER_SPLIT}" ]]; then
  CMD+=(--max-images-per-split "${MAX_IMAGES_PER_SPLIT}")
fi

CMD+=("$@")

printf '[run_local_2_segmentation_linear_probe] CMD:'
printf ' %q' "${CMD[@]}"
printf '\n'

exec "${CMD[@]}"
