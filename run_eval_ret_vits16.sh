#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
set -e
set -o pipefail

MODEL_NAME=dinov3_vits16

# ======================================================
# 1. Patient retrieval head fine-tuning evaluation
# ======================================================
BACKBONE_WEIGHTS=outputs/2_finetune/2_ret/${MODEL_NAME}/1_lvd1689m/1_patient_ret_head_fine_tune/best_model.pth

python eval_retrieval.py \
    --input input/Stent-Contrast/test/ \
    --config configs_retrieval/patients.yaml \
    --model-name ${MODEL_NAME} \
    --repo-dir dinov3 \
    --backbone-weights ${BACKBONE_WEIGHTS} \
    --out-dir outputs/3_eval/2_ret/${MODEL_NAME}/1_lvd1689m/1_patient_ret_head_fine_tune

# ======================================================
# 2. Patient retrieval full fine-tuning evaluation
# ======================================================
BACKBONE_WEIGHTS=outputs/2_finetune/2_ret/${MODEL_NAME}/1_lvd1689m/2_patient_ret_full_fine_tune/best_model.pth

python eval_retrieval.py \
    --input input/stent_split_img/test/ \
    --config configs_retrieval/patients.yaml \
    --model-name ${MODEL_NAME} \
    --repo-dir dinov3 \
    --backbone-weights ${BACKBONE_WEIGHTS} \
    --out-dir outputs/3_eval/2_ret/${MODEL_NAME}/1_lvd1689m/2_patient_ret_full_fine_tune
