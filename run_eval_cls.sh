#!bin/bash

python eval_classifier.py \
--weights outputs/transfer_learn/best_model.pth \
--input input/archive/test \
--config classification_configs/cards.yaml \
--model-name dinov3_convnext_tiny \