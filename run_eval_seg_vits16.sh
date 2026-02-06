#!/bin/bash

# ================================
# 1. Head fine-tuned
# ================================
python eval_segmentation.py \
    --eval-images input/pascal_voc_seg/voc_2012_segmentation_data/valid_images \
    --eval-masks input/pascal_voc_seg/voc_2012_segmentation_data/valid_labels \
    --config configs_segmentation/voc.yaml \
    --weights outputs/2_finetune/3_seg/dinov3_vits16/1_lvd1689m/1_voc2012seg/1_voc_head_finetune/best_model_loss.pth \
    --out-dir outputs/3_eval/3_seg/dinov3_vits16/1_lvd1689m/1_voc2012seg/1_voc_head_finetune \
    --imgsz 640 640 \
    --batch 64 \
    --num-workers 16

# ================================
# 2. Full fine-tuned
# ================================
python eval_segmentation.py \
    --eval-images input/pascal_voc_seg/voc_2012_segmentation_data/valid_images \
    --eval-masks input/pascal_voc_seg/voc_2012_segmentation_data/valid_labels \
    --config configs_segmentation/voc.yaml \
    --weights outputs/2_finetune/3_seg/dinov3_vits16/1_lvd1689m/1_voc2012seg/2_voc_full_finetune/best_model_loss.pth \
    --out-dir outputs/3_eval/3_seg/dinov3_vits16/1_lvd1689m/1_voc2012seg/2_voc_full_finetune \
    --imgsz 640 640 \
    --batch 64 \
    --num-workers 16