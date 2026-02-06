#!/bin/bash
set -e
set -o pipefail

# =========================================
# 1. Head fine tuning
# =========================================
python train_segmentation.py \
  --train-images input/pascal_voc_seg/voc_2012_segmentation_data/train_images \
  --train-masks input/pascal_voc_seg/voc_2012_segmentation_data/train_labels \
  --valid-images input/pascal_voc_seg/voc_2012_segmentation_data/valid_images \
  --valid-masks input/pascal_voc_seg/voc_2012_segmentation_data/valid_labels \
  --config configs_segmentation/voc.yaml \
  --weights dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
  --model-name dinov3_vits16 \
  --epochs 50 \
  --out-dir outputs/2_finetune/3_seg/dinov3_vits16/1_lvd1689m/1_voc2012seg/1_voc_head_finetune \
  --imgsz 640 640 \
  --batch 12

# =========================================
# 2. Full fine tuning
# =========================================
python train_segmentation.py \
  --train-images input/pascal_voc_seg/voc_2012_segmentation_data/train_images \
  --train-masks input/pascal_voc_seg/voc_2012_segmentation_data/train_labels \
  --valid-images input/pascal_voc_seg/voc_2012_segmentation_data/valid_images \
  --valid-masks input/pascal_voc_seg/voc_2012_segmentation_data/valid_labels \
  --config configs_segmentation/voc.yaml \
  --weights dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
  --model-name dinov3_vits16 \
  --epochs 50 \
  --out-dir outputs/2_finetune/3_seg/dinov3_vits16/1_lvd1689m/1_voc2012seg/2_voc_full_finetune \
  --imgsz 640 640 \
  --batch 12 \
  --fine-tune
