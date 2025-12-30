#!/bin/bash

rsync -avh --progress \
  --ignore-missing-args \
  --files-from=/home/heesu/workspace/dinov3_stack/stent_filenames.txt \
  /mnt/nas/snubhcvc/raw/cpacs/ \
  /home/heesu/workspace/dinov3_stack/input/stent/ \
  || true
