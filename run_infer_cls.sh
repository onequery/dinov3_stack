#!bin/bash

python infer_classifier.py \
--weights outputs/transfer_learn/best_model.pth \
--config classification_configs/cards.yaml \
--model-name dinov3_convnext_tiny \
--repo-dir dinov3 \
--input input/inference_data/ 
