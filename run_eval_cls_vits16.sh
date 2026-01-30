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

# -----------------------------
# Evaluation
# -----------------------------
MODEL_NAME=dinov3_vits16
OUTPUT_ROOT=outputs/eval/${MODEL_NAME}
mkdir -p ${OUTPUT_ROOT}

# # -----------------------------
# # 1. Card classification head fine-tuning evaluation
# # -----------------------------
# python eval_classifier.py \
# --weights outputs/train/vits16/1_card_cls_head_fine_tune/best_model.pth \
# --input input/archive/test \
# --config classification_configs/cards.yaml \
# --model-name ${MODEL_NAME} \
# --out-dir ${OUTPUT_ROOT}/1_card_cls_head_fine_tune

# # -----------------------------
# # 2. Card classification full fine-tuning evaluation
# # -----------------------------
# python eval_classifier.py \
# --weights outputs/train/vits16/2_card_cls_full_fine_tune/best_model.pth \
# --input input/archive/test/ \
# --config classification_configs/cards.yaml \
# --model-name ${MODEL_NAME} \
# --out-dir ${OUTPUT_ROOT}/2_card_cls_full_fine_tune

# -----------------------------
# 3. Stent classification head fine-tuning evaluation
# -----------------------------
python eval_classifier.py \
--weights outputs/train/dinov3_vits16/1_lvd1689m/1_cls/1_stent_head_fine_tune/best_model.pth \
--input input/stent_split_img_first_frame/test \
--config configs_classification/stent.yaml \
--model-name ${MODEL_NAME} \
--out-dir ${OUTPUT_ROOT}/1_lvd1689m/1_cls/1_stent_head_fine_tune

# -----------------------------
# 4. Stent classification full fine-tuning evaluation
# -----------------------------
python eval_classifier.py \
--weights outputs/train/dinov3_vits16/1_lvd1689m/1_cls/2_stent_full_fine_tune/best_model.pth \
--input input/stent_split_img_first_frame/test \
--config configs_classification/stent.yaml \
--model-name ${MODEL_NAME} \
--out-dir ${OUTPUT_ROOT}/1_lvd1689m/1_cls/2_stent_full_fine_tune