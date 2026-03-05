#!/bin/bash
set -e
set -o pipefail

export CUDA_VISIBLE_DEVICES=1

MODEL_NAME=dinov3_vits16
EVAL_IMAGES_DIR=input/MPXA-Seg/test_images
EVAL_MASKS_DIR=input/MPXA-Seg/test_labels
SEG_CONFIG=configs_segmentation/mpxa-seg.yaml

IMG_WIDTH=640
IMG_HEIGHT=640
BATCH_SIZE=64
NUM_WORKERS=32
OUTPUT_ROOT=outputs/3_eval/3_seg/${MODEL_NAME}

# Segmentation selection metric: best IoU checkpoint is generally more aligned
# with downstream segmentation quality than best loss checkpoint.
BEST_MODEL_CHECKPOINT=best_model_iou.pth

mkdir -p "$OUTPUT_ROOT"

run_seg_eval() {
  local stage_name="$1"
  local pretrain_name="$2"
  local trained_out_dir="$3"
  local eval_out_dir="$4"
  local weights_filepath="${trained_out_dir}/${BEST_MODEL_CHECKPOINT}"

  if [[ ! -f "$weights_filepath" ]]; then
    echo "[ERROR] Missing checkpoint: $weights_filepath"
    return 1
  fi

  mkdir -p "$eval_out_dir"

  echo "========================================" | tee -a "$eval_out_dir/eval.log"
  echo "==== START Segmentation Eval (${stage_name}, ${pretrain_name}): $(date) ====" | tee -a "$eval_out_dir/eval.log"
  echo "========================================" | tee -a "$eval_out_dir/eval.log"
  echo "CHECKPOINT=$weights_filepath BATCH_SIZE=$BATCH_SIZE NUM_WORKERS=$NUM_WORKERS IMG_SIZE=${IMG_WIDTH}x${IMG_HEIGHT}" | tee -a "$eval_out_dir/eval.log"

  stdbuf -oL -eL python eval_segmentation.py \
    --eval-images "$EVAL_IMAGES_DIR" \
    --eval-masks "$EVAL_MASKS_DIR" \
    --config "$SEG_CONFIG" \
    --weights "$weights_filepath" \
    --out-dir "$eval_out_dir" \
    --imgsz "$IMG_WIDTH" "$IMG_HEIGHT" \
    --batch "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --model-name "$MODEL_NAME" \
    2>&1 | tee -a "$eval_out_dir/eval.log"

  echo "==== END Segmentation Eval (${stage_name}, ${pretrain_name}): $(date) ====" | tee -a "$eval_out_dir/eval.log"
  echo "" | tee -a "$eval_out_dir/eval.log"
}

# ===================================================================
# 1. Segmentation evaluation with MPXA-Seg test set - Full fine-tuning
# ===================================================================
run_seg_eval \
  "Full" \
  "LVD-1689M" \
  "outputs/2_finetune/3_seg/${MODEL_NAME}/1_lvd1689m/2_cor_seg_full_finetune" \
  "${OUTPUT_ROOT}/1_lvd1689m/2_cor_seg_full_finetune"

run_seg_eval \
  "Full" \
  "ImageNet-1K" \
  "outputs/2_finetune/3_seg/${MODEL_NAME}/2_imagenet1k/2_cor_seg_full_finetune" \
  "${OUTPUT_ROOT}/2_imagenet1k/2_cor_seg_full_finetune"

run_seg_eval \
  "Full" \
  "CAG-Contrast-FM-3M" \
  "outputs/2_finetune/3_seg/${MODEL_NAME}/3_cagcontfm3m/2_cor_seg_full_finetune" \
  "${OUTPUT_ROOT}/3_cagcontfm3m/2_cor_seg_full_finetune"

# ===================================================================
# 2. Segmentation evaluation with MPXA-Seg test set - Head fine-tuning
# ===================================================================
run_seg_eval \
  "Head" \
  "LVD-1689M" \
  "outputs/2_finetune/3_seg/${MODEL_NAME}/1_lvd1689m/1_cor_seg_head_finetune" \
  "${OUTPUT_ROOT}/1_lvd1689m/1_cor_seg_head_finetune"

run_seg_eval \
  "Head" \
  "ImageNet-1K" \
  "outputs/2_finetune/3_seg/${MODEL_NAME}/2_imagenet1k/1_cor_seg_head_finetune" \
  "${OUTPUT_ROOT}/2_imagenet1k/1_cor_seg_head_finetune"

run_seg_eval \
  "Head" \
  "CAG-Contrast-FM-3M" \
  "outputs/2_finetune/3_seg/${MODEL_NAME}/3_cagcontfm3m/1_cor_seg_head_finetune" \
  "${OUTPUT_ROOT}/3_cagcontfm3m/1_cor_seg_head_finetune"
