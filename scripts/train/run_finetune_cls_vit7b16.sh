#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
set -e
set -o pipefail

WEIGHTS_FILENAME=dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth
MODEL_NAME=dinov3_vit7b16
BATCH_SIZE_HEAD_FINETUNE_A6000=32
# BATCH_SIZE_FULL_FINETUNE_A6000=1  # Not capable

# Test with small model
# WEIGHTS_FILENAME=dinov3_vits16_pretrain_lvd1689m-08c60483.pth  
# MODEL_NAME=dinov3_vits16  

# # -----------------------------
# # 1. Card classification head fine-tuning
# # -----------------------------
# OUT1=outputs/train/${MODEL_NAME}/1_card_cls_head_fine_tune
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
#   --batch-size "$BATCH_SIZE_HEAD_FINETUNE_A6000"
#   2>&1 | tee -a "$OUT1/train.log"
# # 실행 종료 시각 로그
# echo "==== END Card Classification Head Fine-tuning: $(date) ====" | tee -a "$OUT1/train.log"
# echo "" | tee -a "$OUT1/train.log"
# echo 

# # -----------------------------
# # 2. Card classification full fine-tuning
# # -----------------------------
# OUT1=outputs/train/${MODEL_NAME}/2_card_cls_full_fine_tune
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
#   --epochs 40 \
#   --fine-tune \
#   --out-dir "$OUT1" \
#   --config classification_configs/cards.yaml \
#   2>&1 | tee -a "$OUT1/train.log"

# # 실행 종료 시각 로그
# echo "==== END Card Classification Full Fine-tuning: $(date) ====" | tee -a "$OUT1/train.log"
# echo "" | tee -a "$OUT1/train.log"

# -----------------------------
# 3. Stent classification head fine-tuning
# -----------------------------
OUT2=outputs/train/${MODEL_NAME}/3_stent_cls_head_fine_tune
mkdir -p "$OUT2"

# 실행 시작 시각 로그
echo "========================================" | tee -a "$OUT2/train.log"
echo "==== START Stent Classification Head Fine-tuning: $(date) ====" | tee -a "$OUT2/train.log"
echo "========================================" | tee -a "$OUT2/train.log"

stdbuf -oL -eL python train_classifier.py \
  --train-dir input/stent_split_img/train/ \
  --valid-dir input/stent_split_img/valid/ \
  --weights "$WEIGHTS_FILENAME" \
  --repo-dir dinov3 \
  --model-name "$MODEL_NAME" \
  --epochs 80 \
  --out-dir "$OUT2" \
  --config classification_configs/stent.yaml \
  -lr 0.005 \
  --batch-size "$BATCH_SIZE_HEAD_FINETUNE_A6000" \
  2>&1 | tee -a "$OUT2/train.log"

# 실행 종료 시각 로그
echo "==== END Stent Classification Head Fine-tuning: $(date) ====" | tee -a "$OUT2/train.log"
echo "" | tee -a "$OUT2/train.log"

# # -----------------------------
# # 4. Stent classification full fine-tuning
# # -----------------------------
# OUT2=outputs/train/${MODEL_NAME}/4_stent_cls_full_fine_tune
# mkdir -p "$OUT2"

# # 실행 시작 시각 로그
# echo "========================================" | tee -a "$OUT2/train.log"
# echo "==== START Stent Classification Full Fine-tuning: $(date) ====" | tee -a "$OUT2/train.log"
# echo "========================================" | tee -a "$OUT2/train.log"

# stdbuf -oL -eL python train_classifier.py \
#   --train-dir input/stent_split_img/train/ \
#   --valid-dir input/stent_split_img/valid/ \
#   --weights "$WEIGHTS_FILENAME" \
#   --repo-dir dinov3 \
#   --model-name "$MODEL_NAME" \
#   --epochs 40 \
#   --fine-tune \
#   --out-dir "$OUT2" \
#   --config classification_configs/stent.yaml \
#   2>&1 | tee -a "$OUT2/train.log"

# # 실행 종료 시각 로그
# echo "==== END Stent Classification Full Fine-tuning: $(date) ====" | tee -a "$OUT2/train.log"
# echo "" | tee -a "$OUT2/train.log"
