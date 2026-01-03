#!/bin/bash
set -e

# Cards classification head fine-tuning
# python train_classifier.py \
# --train-dir input/archive/train/ \
# --valid-dir input/archive/valid/ \
# --weights weights/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth \
# --repo-dir dinov3 \
# --model-name dinov3_convnext_tiny \
# --epochs 80 \
# --out-dir transfer_learn \
# -lr 0.005

# -----------------------------
# Card classification
# -----------------------------
OUT1=train/card_cls_full_fine_tune
mkdir -p "$OUT1"

# 실행 시작 시각 로그
echo "========================================" | tee -a "$OUT1/train.log"
echo "==== START Card Classification: $(date) ====" | tee -a "$OUT1/train.log"
echo "========================================" | tee -a "$OUT1/train.log"

stdbuf -oL -eL python train_classifier.py \
  --train-dir input/archive/train/ \
  --valid-dir input/archive/valid/ \
  --weights weights/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth \
  --repo-dir dinov3 \
  --model-name dinov3_convnext_tiny \
  --epochs 20 \
  --fine-tune \
  --out-dir "$OUT1" \
  2>&1 | tee "$OUT1/train.log"

# 실행 종료 시각 로그
echo "==== END Card Classification: $(date) ====" | tee -a "$OUT1/train.log"
echo "" | tee -a "$OUT1/train.log"

# # Stent classification head fine-tuning
# python train_classifier.py \
# --train-dir input/stent_split_img/train/ \
# --valid-dir input/stent_split_img/valid/ \
# --weights weights/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth \
# --repo-dir dinov3 \
# --model-name dinov3_convnext_tiny \
# --epochs 80 \
# --out-dir stent_cls \
# -lr 0.005

# -----------------------------
# Stent classification
# -----------------------------
OUT2=train/stent_cls_full_fine_tune
mkdir -p "$OUT2"

# 실행 시작 시각 로그
echo "========================================" | tee -a "$OUT2/train.log"
echo "==== START Stent Classification: $(date) ====" | tee -a "$OUT2/train.log"
echo "========================================" | tee -a "$OUT2/train.log"

stdbuf -oL -eL python train_classifier.py \
  --train-dir input/stent_split_img/train/ \
  --valid-dir input/stent_split_img/valid/ \
  --weights weights/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth \
  --repo-dir dinov3 \
  --model-name dinov3_convnext_tiny \
  --epochs 20 \
  --fine-tune \
  --out-dir "$OUT2" \
  2>&1 | tee "$OUT2/train.log"

# 실행 시작 시각 로그
echo "========================================" | tee -a "$OUT2/train.log"
echo "==== START Stent Classification: $(date) ====" | tee -a "$OUT2/train.log"
echo "========================================" | tee -a "$OUT2/train.log"