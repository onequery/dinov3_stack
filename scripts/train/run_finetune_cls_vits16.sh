#!/bin/bash
set -e
set -o pipefail

export CUDA_VISIBLE_DEVICES=1

MODEL_NAME=dinov3_vits16
TRAIN_DIR=input/Stent-First-Frame/train/
VALID_DIR=input/Stent-First-Frame/valid/
CLS_CONFIG=configs_classification/stent.yaml

BATCH_SIZE=32
MAX_EPOCHS=1000
NUM_WORKERS=24

EARLY_STOPPING_PATIENCE=30
EARLY_STOPPING_MIN_DELTA=0.0
EARLY_STOPPING_MONITOR=val_loss

HEAD_LR=0.001
FULL_HEAD_LR=0.0001
FULL_BACKBONE_LR=0.00001

run_cls_experiment() {
  local stage_name="$1"
  local pretrain_name="$2"
  local weights_filepath="$3"
  local out_dir="$4"
  local fine_tune="$5"
  local lr="$6"
  local backbone_lr="${7:-}"
  local fine_tune_args=()
  local backbone_lr_args=()

  if [[ "$fine_tune" == "true" ]]; then
    fine_tune_args+=(--fine-tune)
  fi
  if [[ -n "$backbone_lr" ]]; then
    backbone_lr_args+=(--backbone-lr "$backbone_lr")
  fi

  mkdir -p "$out_dir"

  echo "========================================" | tee -a "$out_dir/train.log"
  echo "==== START Stent Classification ${stage_name} Fine-tuning (${pretrain_name}): $(date) ====" | tee -a "$out_dir/train.log"
  echo "========================================" | tee -a "$out_dir/train.log"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] BATCH_SIZE=$BATCH_SIZE NUM_WORKERS=$NUM_WORKERS MAX_EPOCHS=$MAX_EPOCHS HEAD_LR=$lr BACKBONE_LR=${backbone_lr:-none} EARLY_STOPPING_MONITOR=$EARLY_STOPPING_MONITOR PATIENCE=$EARLY_STOPPING_PATIENCE" | tee -a "$out_dir/train.log"

  stdbuf -oL -eL python train_classifier.py \
    --train-dir "$TRAIN_DIR" \
    --valid-dir "$VALID_DIR" \
    --weights "$weights_filepath" \
    --repo-dir dinov3 \
    --model-name "$MODEL_NAME" \
    --max-epochs "$MAX_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --early-stopping \
    --early-stopping-patience "$EARLY_STOPPING_PATIENCE" \
    --early-stopping-min-delta "$EARLY_STOPPING_MIN_DELTA" \
    --early-stopping-monitor "$EARLY_STOPPING_MONITOR" \
    --out-dir "$out_dir" \
    --config "$CLS_CONFIG" \
    -lr "$lr" \
    "${backbone_lr_args[@]}" \
    "${fine_tune_args[@]}" \
    2>&1 | tee -a "$out_dir/train.log"

  echo "==== END Stent Classification ${stage_name} Fine-tuning (${pretrain_name}): $(date) ====" | tee -a "$out_dir/train.log"
  echo "" | tee -a "$out_dir/train.log"
}

# =====================================================
# 1. Stent classification full fine-tuning
# =====================================================
run_cls_experiment \
  "Full" \
  "LVD-1689M" \
  "weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth" \
  "outputs/2_finetune/1_cls/${MODEL_NAME}/1_lvd1689m/2_stent_full_finetune" \
  "true" \
  "$FULL_HEAD_LR" \
  "$FULL_BACKBONE_LR"

run_cls_experiment \
  "Full" \
  "ImageNet-1K" \
  "dinov3/output/a6000/1_pretrain/dinov3_vits16/2_imagenet1k/3_stage3_high_res_adapt/eval/training_29999/teacher_checkpoint.pth" \
  "outputs/2_finetune/1_cls/${MODEL_NAME}/2_imagenet1k/2_stent_full_finetune" \
  "true" \
  "$FULL_HEAD_LR" \
  "$FULL_BACKBONE_LR"

run_cls_experiment \
  "Full" \
  "CAG-Contrast-FM-3M" \
  "dinov3/output/a6000/1_pretrain/dinov3_vits16/3_cagcontfm3m/3_stage3_high_res_adapt/eval/training_29999/teacher_checkpoint.pth" \
  "outputs/2_finetune/1_cls/${MODEL_NAME}/3_cagcontfm3m/2_stent_full_finetune" \
  "true" \
  "$FULL_HEAD_LR" \
  "$FULL_BACKBONE_LR"

# =====================================================
# 2. Stent classification head fine-tuning
# =====================================================
run_cls_experiment \
  "Head" \
  "LVD-1689M" \
  "weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth" \
  "outputs/2_finetune/1_cls/${MODEL_NAME}/1_lvd1689m/1_stent_head_finetune" \
  "false" \
  "$HEAD_LR"

run_cls_experiment \
  "Head" \
  "ImageNet-1K" \
  "dinov3/output/a6000/1_pretrain/dinov3_vits16/2_imagenet1k/3_stage3_high_res_adapt/eval/training_29999/teacher_checkpoint.pth" \
  "outputs/2_finetune/1_cls/${MODEL_NAME}/2_imagenet1k/1_stent_head_finetune" \
  "false" \
  "$HEAD_LR"

run_cls_experiment \
  "Head" \
  "CAG-Contrast-FM-3M" \
  "dinov3/output/a6000/1_pretrain/dinov3_vits16/3_cagcontfm3m/3_stage3_high_res_adapt/eval/training_29999/teacher_checkpoint.pth" \
  "outputs/2_finetune/1_cls/${MODEL_NAME}/3_cagcontfm3m/1_stent_head_finetune" \
  "false" \
  "$HEAD_LR"
