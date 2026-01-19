#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
set -e
set -o pipefail

python eval_retrieval.py \
    --input input/stent_split_img/test/ \
    --config configs_retrieval/patients.yaml \
    --model-name dinov3_vits16 \
    --weights outputs/train/dinov3_vits16/ret/1_patient_ret_head_fine_tune/best_model.pth \
    --out-dir outputs/eval/dinov3_vits16/ret/1_patient_ret_head_fine_tune/ \
