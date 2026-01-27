#!/bin/bash

python export_dicom_frame.py \
  --dcm-root input/stent_split_dcm \
  --img-root input/stent_split_img_contrast \
  --label-json input/frames_prediction.json \
  --frame-index-base 0