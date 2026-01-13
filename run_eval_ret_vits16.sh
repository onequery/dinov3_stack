#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
set -e
set -o pipefail

python eval_retrieval.py \
    --input input/stent_split_img/test/ \
    --config configs_retrieval/patients.yaml \
    --model-name dinov3_vits16 \
    --out-dir outputs/eval/pat_ret/