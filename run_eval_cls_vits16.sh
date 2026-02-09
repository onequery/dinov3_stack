#!/bin/bash

# python eval_classifier.py \
# --weights outputs/transfer_learn/best_model.pth \
# --input input/archive/test \
# --config classification_configs/cards.yaml \
# --model-name dinov3_convnext_tiny \

# python eval_classifier.py \
# --weights outputs/train/stent_cls/best_model.pth \
# --input input/stent_split_img/test \
# --config classification_configs/stent.yaml \
# --model-name dinov3_convnext_tiny \
# --out-dir outputs/eval/stent_cls

# ======================================================
# Setup
# ======================================================
MODEL_NAME=dinov3_vits16
OUTPUT_ROOT=outputs/3_eval/1_cls/${MODEL_NAME}
mkdir -p ${OUTPUT_ROOT}

# ======================================================
# 1. Card classification head fine-tuning evaluation
# ======================================================
# python eval_classifier.py \
# --weights outputs/train/vits16/1_card_cls_head_fine_tune/best_model.pth \
# --input input/archive/test \
# --config classification_configs/cards.yaml \
# --model-name ${MODEL_NAME} \
# --out-dir ${OUTPUT_ROOT}/1_card_cls_head_fine_tune

# ======================================================
# 2. Card classification full fine-tuning evaluation
# ======================================================
# python eval_classifier.py \
# --weights outputs/train/vits16/2_card_cls_full_fine_tune/best_model.pth \
# --input input/archive/test/ \
# --config classification_configs/cards.yaml \
# --model-name ${MODEL_NAME} \
# --out-dir ${OUTPUT_ROOT}/2_card_cls_full_fine_tune

# ======================================================
# 3. Stent classification head fine-tuning evaluation
# ======================================================
# -----------------------------------------------
# 3.1 Using LV-D1689M pre-trained weights
# -----------------------------------------------
# python eval_classifier.py \
# --weights outputs/2_finetune/dinov3_vits16/1_lvd1689m/1_cls/1_stent_head_fine_tune/best_model.pth \
# --input input/Stent-First-Frame/test \
# --config configs_classification/stent.yaml \
# --model-name ${MODEL_NAME} \
# --out-dir ${OUTPUT_ROOT}/1_lvd1689m/1_cls/1_stent_head_fine_tune

# -----------------------------------------------
# 3.2 Using ImageNet-1K pre-trained weights
# -----------------------------------------------
# python eval_classifier.py \
# --weights outputs/2_finetune/1_cls/dinov3_vits16/2_imagenet1k/1_stent_head_finetune/best_model.pth \
# --input input/Stent-First-Frame/test \
# --config configs_classification/stent.yaml \
# --model-name ${MODEL_NAME} \
# --out-dir ${OUTPUT_ROOT}/2_imagenet1k/1_stent_head_finetune

# -----------------------------------------------
# 3.3 Using CAGCont-FM3M pre-trained weights
# -----------------------------------------------
python eval_classifier.py \
--weights outputs/2_finetune/1_cls/dinov3_vits16/3_cagcontfm3m/1_stent_head_finetune/best_model.pth \
--input input/Stent-First-Frame/test \
--config configs_classification/stent.yaml \
--model-name ${MODEL_NAME} \
--out-dir ${OUTPUT_ROOT}/3_cagcontfm3m/1_stent_head_finetune

# ======================================================
# 4. Stent classification full fine-tuning evaluation
# ======================================================
# -----------------------------------------------
# 4.1 Using LV-D1689M pre-trained weights
# -----------------------------------------------
# python eval_classifier.py \
# --weights outputs/2_finetune/dinov3_vits16/1_lvd1689m/1_cls/2_stent_full_fine_tune/best_model.pth \
# --input input/Stent-First-Frame/test \
# --config configs_classification/stent.yaml \
# --model-name ${MODEL_NAME} \
# --out-dir ${OUTPUT_ROOT}/1_lvd1689m/1_cls/2_stent_full_fine_tune

# -----------------------------------------------
# 4.2 Using ImageNet-1K pre-trained weights
# -----------------------------------------------
# python eval_classifier.py \
# --weights outputs/2_finetune/1_cls/dinov3_vits16/2_imagenet1k/2_stent_full_finetune/best_model.pth \
# --input input/Stent-First-Frame/test \
# --config configs_classification/stent.yaml \
# --model-name ${MODEL_NAME} \
# --out-dir ${OUTPUT_ROOT}/2_imagenet1k/2_stent_full_finetune

# -----------------------------------------------
# 4.3 Using CAGCont-FM3M pre-trained weights
# -----------------------------------------------
python eval_classifier.py \
--weights outputs/2_finetune/1_cls/dinov3_vits16/3_cagcontfm3m/2_stent_full_finetune/best_model.pth \
--input input/Stent-First-Frame/test \
--config configs_classification/stent.yaml \
--model-name ${MODEL_NAME} \
--out-dir ${OUTPUT_ROOT}/3_cagcontfm3m/2_stent_full_finetune