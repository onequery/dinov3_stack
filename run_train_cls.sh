#!bin/bash

# python train_classifier.py \
# --train-dir input/archive/train/ \
# --valid-dir input/archive/valid/ \
# --weights weights/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth \
# --repo-dir dinov3 \
# --model-name dinov3_convnext_tiny \
# --epochs 80 \
# --out-dir transfer_learn \
# -lr 0.005

python train_classifier.py \
--train-dir input/stent_split/train/ \
--valid-dir input/stent_split/valid/ \
--weights weights/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth \
--repo-dir dinov3 \
--model-name dinov3_convnext_tiny \
--epochs 80 \
--out-dir stent_cls \
-lr 0.005