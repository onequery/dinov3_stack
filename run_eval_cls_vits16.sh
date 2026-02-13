#!/bin/bash
set -e
set -o pipefail

export CUDA_VISIBLE_DEVICES=1

MODEL_NAME=dinov3_vits16
INPUT_DIR=input/Stent-First-Frame/test
CLS_CONFIG=configs_classification/stent.yaml
OUTPUT_ROOT=outputs/3_eval/1_cls/${MODEL_NAME}

mkdir -p "$OUTPUT_ROOT"

run_cls_eval() {
  local stage_name="$1"
  local pretrain_name="$2"
  local weights_filepath="$3"
  local eval_out_dir="$4"

  if [[ ! -f "$weights_filepath" ]]; then
    echo "[ERROR] Missing checkpoint: $weights_filepath"
    return 1
  fi

  mkdir -p "$eval_out_dir"

  echo "========================================" | tee -a "$eval_out_dir/eval.log"
  echo "==== START Classification Eval (${stage_name}, ${pretrain_name}): $(date) ====" | tee -a "$eval_out_dir/eval.log"
  echo "========================================" | tee -a "$eval_out_dir/eval.log"
  echo "CHECKPOINT=$weights_filepath INPUT=$INPUT_DIR" | tee -a "$eval_out_dir/eval.log"

  stdbuf -oL -eL python eval_classifier.py \
    --weights "$weights_filepath" \
    --input "$INPUT_DIR" \
    --config "$CLS_CONFIG" \
    --model-name "$MODEL_NAME" \
    --out-dir "$eval_out_dir" \
    2>&1 | tee -a "$eval_out_dir/eval.log"

  echo "==== END Classification Eval (${stage_name}, ${pretrain_name}): $(date) ====" | tee -a "$eval_out_dir/eval.log"
  echo "" | tee -a "$eval_out_dir/eval.log"
}

# ======================================================
# 1. Stent classification full fine-tuning evaluation
# ======================================================
run_cls_eval \
  "Full" \
  "LVD-1689M" \
  "outputs/2_finetune/1_cls/${MODEL_NAME}/1_lvd1689m/2_stent_full_finetune/best_model.pth" \
  "${OUTPUT_ROOT}/1_lvd1689m/2_stent_full_finetune"

run_cls_eval \
  "Full" \
  "ImageNet-1K" \
  "outputs/2_finetune/1_cls/${MODEL_NAME}/2_imagenet1k/2_stent_full_finetune/best_model.pth" \
  "${OUTPUT_ROOT}/2_imagenet1k/2_stent_full_finetune"

run_cls_eval \
  "Full" \
  "CAG-Contrast-FM-3M" \
  "outputs/2_finetune/1_cls/${MODEL_NAME}/3_cagcontfm3m/2_stent_full_finetune/best_model.pth" \
  "${OUTPUT_ROOT}/3_cagcontfm3m/2_stent_full_finetune"

# ======================================================
# 2. Stent classification head fine-tuning evaluation
# ======================================================
run_cls_eval \
  "Head" \
  "LVD-1689M" \
  "outputs/2_finetune/1_cls/${MODEL_NAME}/1_lvd1689m/1_stent_head_finetune/best_model.pth" \
  "${OUTPUT_ROOT}/1_lvd1689m/1_stent_head_finetune"

run_cls_eval \
  "Head" \
  "ImageNet-1K" \
  "outputs/2_finetune/1_cls/${MODEL_NAME}/2_imagenet1k/1_stent_head_finetune/best_model.pth" \
  "${OUTPUT_ROOT}/2_imagenet1k/1_stent_head_finetune"

run_cls_eval \
  "Head" \
  "CAG-Contrast-FM-3M" \
  "outputs/2_finetune/1_cls/${MODEL_NAME}/3_cagcontfm3m/1_stent_head_finetune/best_model.pth" \
  "${OUTPUT_ROOT}/3_cagcontfm3m/1_stent_head_finetune"
