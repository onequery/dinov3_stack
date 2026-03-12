#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${CONDA_ENV:-dinov3_stack}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

IMAGE_ROOT="${IMAGE_ROOT:-input/Stent-Contrast}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/global_1_anatomy_identity_proxy}"
DEVICE="${DEVICE:-cuda}"

MODEL_NAME="${MODEL_NAME:-dinov3_vits16}"
REPO_DIR="${REPO_DIR:-dinov3}"
IMAGENET_CKPT="${IMAGENET_CKPT:-dinov3/output/a6000/1_pretrain/dinov3_vits16/2_imagenet1k/3_stage3_high_res_adapt/eval/training_29999/teacher_checkpoint.pth}"
CAG_CKPT="${CAG_CKPT:-dinov3/output/a6000/1_pretrain/dinov3_vits16/3_cagcontfm3m/3_stage3_high_res_adapt/eval/training_29999/teacher_checkpoint.pth}"

BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-42}"
NEGATIVE_RATIO="${NEGATIVE_RATIO:-1.0}"
RESIZE_SIZE="${RESIZE_SIZE:-480}"
CENTER_CROP_SIZE="${CENTER_CROP_SIZE:-448}"
MAX_IMAGES="${MAX_IMAGES:-}"
LOG_FILE="${LOG_FILE:-run_gpu${CUDA_VISIBLE_DEVICES}.log}"

echo "[run_global_1_anatomy_identity_proxy] ROOT_DIR=${ROOT_DIR}"
echo "[run_global_1_anatomy_identity_proxy] CONDA_ENV=${CONDA_ENV} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[run_global_1_anatomy_identity_proxy] OUTPUT_ROOT=${OUTPUT_ROOT}"

CMD=(
  conda run --no-capture-output -n "${CONDA_ENV}"
  env
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
  PYTHONUNBUFFERED="${PYTHONUNBUFFERED}"
  python scripts/analysis/global_1_anatomy_identity_proxy.py
  --image-root "${IMAGE_ROOT}"
  --imagenet-ckpt "${IMAGENET_CKPT}"
  --cag-ckpt "${CAG_CKPT}"
  --output-root "${OUTPUT_ROOT}"
  --model-name "${MODEL_NAME}"
  --repo-dir "${REPO_DIR}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --seed "${SEED}"
  --negative-ratio "${NEGATIVE_RATIO}"
  --device "${DEVICE}"
  --resize-size "${RESIZE_SIZE}"
  --center-crop-size "${CENTER_CROP_SIZE}"
  --log-file "${LOG_FILE}"
)

if [[ -n "${MAX_IMAGES}" ]]; then
  CMD+=(--max-images "${MAX_IMAGES}")
fi

CMD+=("$@")

printf '[run_global_1_anatomy_identity_proxy] CMD:'
printf ' %q' "${CMD[@]}"
printf '\n'

exec "${CMD[@]}"
