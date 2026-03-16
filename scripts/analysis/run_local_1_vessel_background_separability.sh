#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
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

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/local_1_vessel_background_separability}"
DEVICE="${DEVICE:-cuda}"
MODEL_NAME="${MODEL_NAME:-dinov3_vits16}"
REPO_DIR="${REPO_DIR:-dinov3}"
IMAGENET_CKPT="${IMAGENET_CKPT:-dinov3/output/a6000/1_pretrain/dinov3_vits16/2_imagenet1k/3_stage3_high_res_adapt/eval/training_29999/teacher_checkpoint.pth}"
CAG_CKPT="${CAG_CKPT:-dinov3/output/a6000/1_pretrain/dinov3_vits16/3_cagcontfm3m/3_stage3_high_res_adapt/eval/training_29999/teacher_checkpoint.pth}"

IMG_W="${IMG_W:-448}"
IMG_H="${IMG_H:-448}"
PATCH_SIZE="${PATCH_SIZE:-16}"
VESSEL_FRAC_POS_THR="${VESSEL_FRAC_POS_THR:-0.10}"
BACKGROUND_FRAC_NEG_THR="${BACKGROUND_FRAC_NEG_THR:-0.01}"
TRAIN_MAX_POS_PER_IMAGE="${TRAIN_MAX_POS_PER_IMAGE:-32}"
TRAIN_MAX_NEG_PER_IMAGE="${TRAIN_MAX_NEG_PER_IMAGE:-32}"
VALID_MAX_POS_PER_IMAGE="${VALID_MAX_POS_PER_IMAGE:-32}"
VALID_MAX_NEG_PER_IMAGE="${VALID_MAX_NEG_PER_IMAGE:-32}"
VIZ_SAMPLES_PER_CLASS="${VIZ_SAMPLES_PER_CLASS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-42}"
MAX_IMAGES_PER_SPLIT="${MAX_IMAGES_PER_SPLIT:-}"
LOG_FILE="${LOG_FILE:-run_gpu${CUDA_VISIBLE_DEVICES}.log}"

echo "[run_local_1_vessel_background_separability] ROOT_DIR=${ROOT_DIR}"
echo "[run_local_1_vessel_background_separability] CONDA_ENV=${CONDA_ENV} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[run_local_1_vessel_background_separability] OUTPUT_ROOT=${OUTPUT_ROOT}"

CMD=(
  conda run --no-capture-output -n "${CONDA_ENV}"
  env
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
  PYTHONUNBUFFERED="${PYTHONUNBUFFERED}"
  python scripts/analysis/local_1_vessel_background_separability.py
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
  --model-name "${MODEL_NAME}"
  --repo-dir "${REPO_DIR}"
  --img-size "${IMG_W}" "${IMG_H}"
  --patch-size "${PATCH_SIZE}"
  --vessel-frac-pos-thr "${VESSEL_FRAC_POS_THR}"
  --background-frac-neg-thr "${BACKGROUND_FRAC_NEG_THR}"
  --train-max-pos-per-image "${TRAIN_MAX_POS_PER_IMAGE}"
  --train-max-neg-per-image "${TRAIN_MAX_NEG_PER_IMAGE}"
  --valid-max-pos-per-image "${VALID_MAX_POS_PER_IMAGE}"
  --valid-max-neg-per-image "${VALID_MAX_NEG_PER_IMAGE}"
  --viz-samples-per-class "${VIZ_SAMPLES_PER_CLASS}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --device "${DEVICE}"
  --seed "${SEED}"
  --cache-features
  --log-file "${LOG_FILE}"
)

if [[ -n "${MAX_IMAGES_PER_SPLIT}" ]]; then
  CMD+=(--max-images-per-split "${MAX_IMAGES_PER_SPLIT}")
fi

CMD+=("$@")

printf '[run_local_1_vessel_background_separability] CMD:'
printf ' %q' "${CMD[@]}"
printf '\n'

exec "${CMD[@]}"
