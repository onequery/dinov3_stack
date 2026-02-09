#!/bin/bash

set -e
set -o pipefail

export CUDA_VISIBLE_DEVICES=0

MODEL_NAME=dinov3_vits16
NUM_WORKERS=32
BATCH_SIZE=128  # Need to fix through out experiments
TRAIN_DIR=input/Stent-Contrast/train/
VALID_DIR=input/Stent-Contrast/valid/
WEIGHTS_FILEPATH=weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
EARLY_STOPPING_PATIENCE=15
EARLY_STOPPING_MIN_DELTA=0.0
EARLY_STOPPING_MONITOR=r1

# =====================================================
# 1. Patient retrieval head fine-tuning
# =====================================================
# -----------------------------------------------
# 1.1 Using LV-D1689M pre-trained weights
# -----------------------------------------------
OUT=outputs/2_finetune/2_ret/${MODEL_NAME}/1_lvd1689m/1_patient_ret_head_fine_tune
mkdir -p "$OUT"

# # 실행 시작 시각 로그
echo "========================================" | tee -a "$OUT/train.log"
echo "==== START Patient Retrieval Head Fine-tuning: $(date) ====" | tee -a "$OUT/train.log"
echo "========================================" | tee -a "$OUT/train.log"
echo "BATCH_SIZE=$BATCH_SIZE NUM_WORKERS=$NUM_WORKERS EARLY_STOPPING_MONITOR=$EARLY_STOPPING_MONITOR PATIENCE=$EARLY_STOPPING_PATIENCE" | tee -a "$OUT/train.log"

stdbuf -oL -eL python train_retrieval.py \
  --train-dir "$TRAIN_DIR" \
  --valid-dir "$VALID_DIR" \
  --weights "$WEIGHTS_FILEPATH" \
  --repo-dir dinov3 \
  --model-name "$MODEL_NAME" \
  --max-epochs 1000 \
  --out-dir "$OUT" \
  --config configs_retrieval/patients.yaml \
  --num-workers "$NUM_WORKERS" \
  --batch-size "$BATCH_SIZE" \
  --early-stopping \
  --early-stopping-patience "$EARLY_STOPPING_PATIENCE" \
  --early-stopping-min-delta "$EARLY_STOPPING_MIN_DELTA" \
  --early-stopping-monitor "$EARLY_STOPPING_MONITOR" \
  2>&1 | tee -a "$OUT/train.log"

# 실행 종료 시각 로그
echo "==== END Patient Retrieval Head Fine-tuning: $(date) ====" | tee -a "$OUT/train.log"
echo "" | tee -a "$OUT/train.log"

# =====================================================
# 2. Patient retrieval full fine-tuning
# =====================================================
# -----------------------------------------------
# 2.1 Using LV-D1689M pre-trained weights
# -----------------------------------------------
OUT=outputs/2_finetune/2_ret/${MODEL_NAME}/1_lvd1689m/2_patient_ret_full_fine_tune
mkdir -p "$OUT"

# 실행 시작 시각 로그
echo "========================================" | tee -a "$OUT/train.log"
echo "==== START Patient Retrieval Full Fine-tuning: $(date) ====" | tee -a "$OUT/train.log"
echo "========================================" | tee -a "$OUT/train.log"
echo "BATCH_SIZE=$BATCH_SIZE NUM_WORKERS=$NUM_WORKERS EARLY_STOPPING_MONITOR=$EARLY_STOPPING_MONITOR PATIENCE=$EARLY_STOPPING_PATIENCE" | tee -a "$OUT/train.log"

stdbuf -oL -eL python train_retrieval.py \
  --train-dir "$TRAIN_DIR" \
  --valid-dir "$VALID_DIR" \
  --weights "$WEIGHTS_FILEPATH" \
  --repo-dir dinov3 \
  --model-name "$MODEL_NAME" \
  --max-epochs 1000 \
  --fine-tune \
  --out-dir "$OUT" \
  --config configs_retrieval/patients.yaml \
  --num-workers "$NUM_WORKERS" \
  --batch-size "$BATCH_SIZE" \
  --early-stopping \
  --early-stopping-patience "$EARLY_STOPPING_PATIENCE" \
  --early-stopping-min-delta "$EARLY_STOPPING_MIN_DELTA" \
  --early-stopping-monitor "$EARLY_STOPPING_MONITOR" \
  2>&1 | tee -a "$OUT/train.log"

# 실행 종료 시각 로그
echo "==== END Patient Retrieval Full Fine-tuning: $(date) ====" | tee -a "$OUT/train.log"
echo "" | tee -a "$OUT/train.log"
