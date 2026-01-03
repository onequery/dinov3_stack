#!bin/bash

# python eval_classifier.py \
# --weights outputs/transfer_learn/best_model.pth \
# --input input/archive/test \
# --config classification_configs/cards.yaml \
# --model-name dinov3_convnext_tiny \

python eval_classifier.py \
--weights outputs/train/stent_cls/best_model.pth \
--input input/stent_split_img/test \
--config classification_configs/stent.yaml \
--model-name dinov3_convnext_tiny \
--out-dir outputs/eval/stent_cls