#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
set -e
set -o pipefail

python eval_retrieval.py \
    --input input/stent_split_img_contrast/test/ \
    --config configs_retrieval/patients.yaml \
    --model-name dinov3_vits16 \
    --out-dir outputs/eval/dinov3_vits16/ret/1_cls_token