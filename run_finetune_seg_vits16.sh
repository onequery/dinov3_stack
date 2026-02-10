#!/bin/bash
set -e
set -o pipefail

# ======================================================================
# 0. Segementation fine-tuning test with VOC 2012 segmentation dataset
# ======================================================================
# -----------------------------------
# 0.1 Head fine tuning
# -----------------------------------
# python train_segmentation.py \
#   --train-images input/pascal_voc_seg/voc_2012_segmentation_data/train_images \
#   --train-masks input/pascal_voc_seg/voc_2012_segmentation_data/train_labels \
#   --valid-images input/pascal_voc_seg/voc_2012_segmentation_data/valid_images \
#   --valid-masks input/pascal_voc_seg/voc_2012_segmentation_data/valid_labels \
#   --config configs_segmentation/voc.yaml \
#   --weights dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
#   --model-name dinov3_vits16 \
#   --epochs 50 \
#   --out-dir outputs/2_finetune/3_seg/dinov3_vits16/1_lvd1689m/1_voc2012seg/1_voc_head_finetune \
#   --imgsz 640 640 \
#   --batch 12

# -----------------------------------
# 0.2 Full fine tuning
# -----------------------------------
# python train_segmentation.py \
#   --train-images input/pascal_voc_seg/voc_2012_segmentation_data/train_images \
#   --train-masks input/pascal_voc_seg/voc_2012_segmentation_data/train_labels \
#   --valid-images input/pascal_voc_seg/voc_2012_segmentation_data/valid_images \
#   --valid-masks input/pascal_voc_seg/voc_2012_segmentation_data/valid_labels \
#   --config configs_segmentation/voc.yaml \
#   --weights dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
#   --model-name dinov3_vits16 \
#   --epochs 50 \
#   --out-dir outputs/2_finetune/3_seg/dinov3_vits16/1_lvd1689m/1_voc2012seg/2_voc_full_finetune \
#   --imgsz 640 640 \
#   --batch 12 \
#   --fine-tune


# ===================================================================
# 1. Segmentation fine-tuning with MPXA-Seg dataset - Head fine-tuning
# ===================================================================
# -----------------------------------------------------------------
# 1.1 Pre-trained on LV-1689M dataset
# -----------------------------------------------------------------
python train_segmentation.py \
  --train-images input/MPXA-Seg/train_images \
  --train-masks input/MPXA-Seg/train_labels \
  --valid-images input/MPXA-Seg/valid_images \
  --valid-masks input/MPXA-Seg/valid_labels \
  --config configs_segmentation/mpxa-seg.yaml \
  --weights weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
  --model-name dinov3_vits16 \
  --max-epochs 1000 \
  --out-dir outputs/2_finetune/3_seg/dinov3_vits16/1_lvd1689m/1_cor_seg_head_finetune \
  --imgsz 640 640 \
  --batch 12
# -----------------------------------------------------------------
# 1.2 Pre-trained on ImageNet-1K dataset
# -----------------------------------------------------------------

# -----------------------------------------------------------------
# 1.3 Pre-trained on CAG-Contrast-FM-3M dataset
# -----------------------------------------------------------------

# ===================================================================
# 2. Segmentation fine-tuning with MPXA-Seg dataset - Full fine-tuning
# ===================================================================
# -----------------------------------------------------------------
# 1.1 Pre-trained on LV-1689M dataset
# -----------------------------------------------------------------
# 
# -----------------------------------------------------------------
# 1.2 Pre-trained on ImageNet-1K dataset
# -----------------------------------------------------------------

# -----------------------------------------------------------------
# 1.3 Pre-trained on CAG-Contrast-FM-3M dataset
# -----------------------------------------------------------------

