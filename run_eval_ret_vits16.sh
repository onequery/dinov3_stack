#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
set -e
set -o pipefail

MODEL_NAME=dinov3_vits16
PROJ_DIM=128

# ======================================================
# 1. Patient retrieval head fine-tuning evaluation
# ======================================================
# -----------------------------------------------
# 1.1 Backbone pretraining dataset: LVD-1689M
# -----------------------------------------------
WEIGHTS_FILEPATH=outputs/2_finetune/2_ret/${MODEL_NAME}/1_lvd1689m/1_patient_ret_head_fine_tune/best_model.pth

python eval_retrieval.py \
    --input input/Stent-Contrast/test/ \
    --config configs_retrieval/patients.yaml \
    --model-name ${MODEL_NAME} \
    --repo-dir dinov3 \
    --weights ${WEIGHTS_FILEPATH} \
    --proj-dim ${PROJ_DIM} \
    --out-dir outputs/3_eval/2_ret/${MODEL_NAME}/1_lvd1689m/1_patient_ret_head_fine_tune

# -----------------------------------------------
# 1.2 Backbone pretraining dataset: ImageNet-1K
# -----------------------------------------------
WEIGHTS_FILEPATH=outputs/2_finetune/2_ret/${MODEL_NAME}/2_imagenet1k/1_patient_ret_head_fine_tune/best_model.pth

python eval_retrieval.py \
    --input input/Stent-Contrast/test/ \
    --config configs_retrieval/patients.yaml \
    --model-name ${MODEL_NAME} \
    --repo-dir dinov3 \
    --weights ${WEIGHTS_FILEPATH} \
    --proj-dim ${PROJ_DIM} \
    --out-dir outputs/3_eval/2_ret/${MODEL_NAME}/2_imagenet1k/1_patient_ret_head_fine_tune

# -----------------------------------------------
# 1.3 Backbone pretraining dataset: CAG-Contrast-FM-3M
# -----------------------------------------------
WEIGHTS_FILEPATH=outputs/2_finetune/2_ret/${MODEL_NAME}/3_cagcontfm3m/1_patient_ret_head_fine_tune/best_model.pth

python eval_retrieval.py \
    --input input/Stent-Contrast/test/ \
    --config configs_retrieval/patients.yaml \
    --model-name ${MODEL_NAME} \
    --repo-dir dinov3 \
    --weights ${WEIGHTS_FILEPATH} \
    --proj-dim ${PROJ_DIM} \
    --out-dir outputs/3_eval/2_ret/${MODEL_NAME}/3_cagcontfm3m/1_patient_ret_head_fine_tune

# ======================================================
# 2. Patient retrieval full fine-tuning evaluation
# ======================================================
# -----------------------------------------------
# 1.1 Backbone pretraining dataset: LVD-1689M
# -----------------------------------------------
WEIGHTS_FILEPATH=outputs/2_finetune/2_ret/${MODEL_NAME}/1_lvd1689m/2_patient_ret_full_fine_tune/best_model.pth

python eval_retrieval.py \
    --input input/Stent-Contrast/test/ \
    --config configs_retrieval/patients.yaml \
    --model-name ${MODEL_NAME} \
    --repo-dir dinov3 \
    --weights ${WEIGHTS_FILEPATH} \
    --proj-dim ${PROJ_DIM} \
    --out-dir outputs/3_eval/2_ret/${MODEL_NAME}/1_lvd1689m/2_patient_ret_full_fine_tune

# -----------------------------------------------
# 1.2 Backbone pretraining dataset: ImageNet-1K
# -----------------------------------------------
WEIGHTS_FILEPATH=outputs/2_finetune/2_ret/${MODEL_NAME}/2_imagenet1k/2_patient_ret_full_fine_tune/best_model.pth

python eval_retrieval.py \
    --input input/Stent-Contrast/test/ \
    --config configs_retrieval/patients.yaml \
    --model-name ${MODEL_NAME} \
    --repo-dir dinov3 \
    --weights ${WEIGHTS_FILEPATH} \
    --proj-dim ${PROJ_DIM} \
    --out-dir outputs/3_eval/2_ret/${MODEL_NAME}/2_imagenet1k/2_patient_ret_full_fine_tune

# -----------------------------------------------
# 1.3 Backbone pretraining dataset: CAG-Contrast-FM-3M
# -----------------------------------------------
WEIGHTS_FILEPATH=outputs/2_finetune/2_ret/${MODEL_NAME}/3_cagcontfm3m/2_patient_ret_full_fine_tune/best_model.pth

python eval_retrieval.py \
    --input input/Stent-Contrast/test/ \
    --config configs_retrieval/patients.yaml \
    --model-name ${MODEL_NAME} \
    --repo-dir dinov3 \
    --weights ${WEIGHTS_FILEPATH} \
    --proj-dim ${PROJ_DIM} \
    --out-dir outputs/3_eval/2_ret/${MODEL_NAME}/3_cagcontfm3m/2_patient_ret_full_fine_tune