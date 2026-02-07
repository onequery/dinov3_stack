#!/bin/bash
set -e
set -o pipefail

# WEIGHTS_FILENAME=dinov3_vits16_pretrain_lvd1689m-08c60483.pth
WEIGHTS_FILEPATH=dinov3/output/a6000/1_pretrain/dinov3_vits16/2_imagenet1k/3_stage3_high_res_adapt/eval/training_29999/teacher_checkpoint.pth
# WEIGHTS_FILEPATH=weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
MODEL_NAME=dinov3_vits16

# Dataset (ImageFolder-style)
TRAIN_DIR=input/Stent-First-Frame/train/
VALID_DIR=input/Stent-First-Frame/valid/

# NOTE: ViT-S/16 with CENTER_CROP_SIZE=448 is memory heavy.
# Start small (e.g., 2~8) and increase if you have headroom.
BATCH_SIZE=32

# # -----------------------------
# # Card classification head fine-tuning
# # -----------------------------
# OUT1=outputs/train/vits16/1_card_cls_head_fine_tune
# mkdir -p "$OUT1"

# # 실행 시작 시각 로그
# echo "========================================" | tee -a "$OUT1/train.log"
# echo "==== START Card Classification Head Fine-tuning: $(date) ====" | tee -a "$OUT1/train.log"
# echo "========================================" | tee -a "$OUT1/train.log"
# stdbuf -oL -eL python train_classifier.py \
#   --train-dir input/archive/train/ \
#   --valid-dir input/archive/valid/ \
#   --weights "$WEIGHTS_FILENAME" \
#   --repo-dir dinov3 \
#   --model-name "$MODEL_NAME" \
#   --epochs 80 \
#   --out-dir "$OUT1" \
#   --config classification_configs/cards.yaml \
#   -lr 0.005 \
#   2>&1 | tee -a "$OUT1/train.log"
# # 실행 종료 시각 로그
# echo "==== END Card Classification Head Fine-tuning: $(date) ====" | tee -a "$OUT1/train.log"
# echo "" | tee -a "$OUT1/train.log"
# echo 

# # -----------------------------
# # Card classification full fine-tuning
# # -----------------------------
# OUT1=outputs/train/vits16/2_card_cls_full_fine_tune
# mkdir -p "$OUT1"

# # 실행 시작 시각 로그
# echo "========================================" | tee -a "$OUT1/train.log"
# echo "==== START Card Classification Full Fine-tuning: $(date) ====" | tee -a "$OUT1/train.log"
# echo "========================================" | tee -a "$OUT1/train.log"

# stdbuf -oL -eL python train_classifier.py \
#   --train-dir input/archive/train/ \
#   --valid-dir input/archive/valid/ \
#   --weights "$WEIGHTS_FILENAME" \
#   --repo-dir dinov3 \
#   --model-name "$MODEL_NAME" \
#   --epochs 20 \
#   --fine-tune \
#   --out-dir "$OUT1" \
#   --config classification_configs/cards.yaml \
#   2>&1 | tee -a "$OUT1/train.log"

# # 실행 종료 시각 로그
# echo "==== END Card Classification Full Fine-tuning: $(date) ====" | tee -a "$OUT1/train.log"
# echo "" | tee -a "$OUT1/train.log"

# -----------------------------
# 1. Stent classification head fine-tuning
# -----------------------------
# OUT2=outputs/2_finetune/${MODEL_NAME}/1_lvd1689m/1_cls/1_stent_head_fine_tune
# OUT2=outputs/2_finetune/${MODEL_NAME}/2_imagenet1k/1_cls/1_stent_head_fine_tune
OUT2=outputs/2_finetune/1_cls/${MODEL_NAME}/2_imagenet1k/1_stent_head_finetune
mkdir -p "$OUT2"

# 실행 시작 시각 로그
echo "========================================" | tee -a "$OUT2/train.log"
echo "==== START Stent Classification Head Fine-tuning: $(date) ====" | tee -a "$OUT2/train.log"
echo "========================================" | tee -a "$OUT2/train.log"

stdbuf -oL -eL python train_classifier.py \
  --train-dir "$TRAIN_DIR" \
  --valid-dir "$VALID_DIR" \
  --weights "$WEIGHTS_FILEPATH" \
  --repo-dir dinov3 \
  --model-name "$MODEL_NAME" \
  --max-epochs 1000 \
  --batch-size "$BATCH_SIZE" \
  --early-stopping \
  --early-stopping-patience 15 \
  --out-dir "$OUT2" \
  --config configs_classification/stent.yaml \
  -lr 0.005 \
  2>&1 | tee -a "$OUT2/train.log"

# 실행 종료 시각 로그
echo "==== END Stent Classification Head Fine-tuning: $(date) ====" | tee -a "$OUT2/train.log"
echo "" | tee -a "$OUT2/train.log"

# -----------------------------
# 2. Stent classification full fine-tuning
# -----------------------------
# OUT2=outputs/2_finetune/${MODEL_NAME}/1_lvd1689m/1_cls/2_stent_full_fine_tune
# OUT2=outputs/2_finetune/${MODEL_NAME}/2_imagenet1k/1_cls/2_stent_full_fine_tune
# OUT2=outputs/train/${MODEL_NAME}/3_cagimgs/1_cls/2_stent_full_fine_tune
OUT2=outputs/2_finetune/1_cls/${MODEL_NAME}/2_imagenet1k/2_stent_full_finetune

mkdir -p "$OUT2"

# 실행 시작 시각 로그
echo "========================================" | tee -a "$OUT2/train.log"
echo "==== START Stent Classification Full Fine-tuning: $(date) ====" | tee -a "$OUT2/train.log"
echo "========================================" | tee -a "$OUT2/train.log"

stdbuf -oL -eL python train_classifier.py \
  --train-dir "$TRAIN_DIR" \
  --valid-dir "$VALID_DIR" \
  --weights "$WEIGHTS_FILEPATH" \
  --repo-dir dinov3 \
  --model-name "$MODEL_NAME" \
  --max-epochs 1000 \
  --batch-size "$BATCH_SIZE" \
  --early-stopping \
  --early-stopping-patience 15 \
  --fine-tune \
  --out-dir "$OUT2" \
  --config configs_classification/stent.yaml \
  -lr 0.0001 \
  2>&1 | tee -a "$OUT2/train.log"

# 실행 시작 시각 로그
echo "==== END Stent Classification Full Fine-tuning: $(date) ====" | tee -a "$OUT2/train.log"
echo "" | tee -a "$OUT2/train.log"
