#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
set -e
set -o pipefail

WEIGHTS_FILENAME=dinov3_vits16_pretrain_lvd1689m-08c60483.pth
MODEL_NAME=dinov3_vits16
# BATCH_SIZE_HEAD_FINETUNE_A6000=32
# BATCH_SIZE_FULL_FINETUNE_A6000=1  # Not capable

# -----------------------------
# 1. Patient retrieval head fine-tuning
# -----------------------------
OUT=outputs/train/${MODEL_NAME}/1_patient_ret_head_fine_tune
mkdir -p "$OUT"

# 실행 시작 시각 로그
echo "========================================" | tee -a "$OUT/train.log"
echo "==== START Patient Retrieval Head Fine-tuning: $(date) ====" | tee -a "$OUT/train.log"
echo "========================================" | tee -a "$OUT/train.log"

stdbuf -oL -eL python train_retrieval.py \
  --train-dir input/stent_split_img/train/ \
  --valid-dir input/stent_split_img/valid/ \
  --weights "$WEIGHTS_FILENAME" \
  --model-name "$MODEL_NAME" \
  --epochs 10 \
  --out-dir "$OUT" \
  --config configs_retrieval/patients.yaml \
  2>&1 | tee -a "$OUT/train.log"

# 실행 종료 시각 로그
echo "==== END Patient Retrieval Head Fine-tuning: $(date) ====" | tee -a "$OUT/train.log"
echo "" | tee -a "$OUT/train.log"

# # -----------------------------
# # 4. Stent classification full fine-tuning
# # -----------------------------
# OUT=outputs/train/${MODEL_NAME}/4_stent_cls_full_fine_tune
# mkdir -p "$OUT"

# # 실행 시작 시각 로그
# echo "========================================" | tee -a "$OUT/train.log"
# echo "==== START Stent Classification Full Fine-tuning: $(date) ====" | tee -a "$OUT/train.log"
# echo "========================================" | tee -a "$OUT/train.log"

# stdbuf -oL -eL python train_classifier.py \
#   --train-dir input/stent_split_img/train/ \
#   --valid-dir input/stent_split_img/valid/ \
#   --weights "$WEIGHTS_FILENAME" \
#   --repo-dir dinov3 \
#   --model-name "$MODEL_NAME" \
#   --epochs 40 \
#   --fine-tune \
#   --out-dir "$OUT" \
#   --config classification_configs/stent.yaml \
#   2>&1 | tee -a "$OUT/train.log"

# # 실행 종료 시각 로그
# echo "==== END Stent Classification Full Fine-tuning: $(date) ====" | tee -a "$OUT/train.log"
# echo "" | tee -a "$OUT/train.log"
