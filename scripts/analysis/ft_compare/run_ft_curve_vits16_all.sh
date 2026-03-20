#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODEL_NAME="${MODEL_NAME:-dinov3_vits16}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-outputs/4_ft_curve}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SMOKE_TEST="${SMOKE_TEST:-0}"

UNFREEZE_BLOCKS_STR="${UNFREEZE_BLOCKS_STR:-0 1 2 4 8 12}"
read -r -a UNFREEZE_BLOCKS <<<"$UNFREEZE_BLOCKS_STR"

PRETRAIN_KEYS=("1_lvd1689m" "2_imagenet1k" "3_cagcontfm3m")
PRETRAIN_LABELS=("LVD-1689M" "ImageNet-1K" "CAG-Contrast-FM-3M")
PRETRAIN_WEIGHTS=(
  "weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
  "dinov3/output/a6000/1_pretrain/dinov3_vits16/2_imagenet1k/3_stage3_high_res_adapt/eval/training_29999/teacher_checkpoint.pth"
  "dinov3/output/a6000/1_pretrain/dinov3_vits16/3_cagcontfm3m/3_stage3_high_res_adapt/eval/training_29999/teacher_checkpoint.pth"
)
PRETRAIN_INDEXES_STR="${PRETRAIN_INDEXES_STR:-0 1 2}"
read -r -a PRETRAIN_INDEXES <<<"$PRETRAIN_INDEXES_STR"

# Classification settings
CLS_TRAIN_DIR="${CLS_TRAIN_DIR:-input/Stent-First-Frame/train/}"
CLS_VALID_DIR="${CLS_VALID_DIR:-input/Stent-First-Frame/valid/}"
CLS_TEST_DIR="${CLS_TEST_DIR:-input/Stent-First-Frame/test/}"
CLS_CONFIG="${CLS_CONFIG:-configs_classification/stent.yaml}"
CLS_BATCH_SIZE="${CLS_BATCH_SIZE:-32}"
CLS_MAX_EPOCHS="${CLS_MAX_EPOCHS:-1000}"
CLS_NUM_WORKERS="${CLS_NUM_WORKERS:-24}"
CLS_EARLY_STOP_PATIENCE="${CLS_EARLY_STOP_PATIENCE:-30}"
CLS_EARLY_STOP_MIN_DELTA="${CLS_EARLY_STOP_MIN_DELTA:-0.0}"
CLS_EARLY_STOP_MONITOR="${CLS_EARLY_STOP_MONITOR:-val_loss}"
CLS_HEAD_LR="${CLS_HEAD_LR:-0.001}"
CLS_FT_HEAD_LR="${CLS_FT_HEAD_LR:-0.0001}"
CLS_FT_BACKBONE_LR="${CLS_FT_BACKBONE_LR:-0.00001}"

# Retrieval settings
RET_TRAIN_DIR="${RET_TRAIN_DIR:-input/Stent-Contrast/train/}"
RET_VALID_DIR="${RET_VALID_DIR:-input/Stent-Contrast/valid/}"
RET_TEST_DIR="${RET_TEST_DIR:-input/Stent-Contrast/test/}"
RET_CONFIG="${RET_CONFIG:-configs_retrieval/patients.yaml}"
RET_BATCH_SIZE="${RET_BATCH_SIZE:-128}"
RET_MAX_EPOCHS="${RET_MAX_EPOCHS:-1000}"
RET_NUM_WORKERS="${RET_NUM_WORKERS:-32}"
RET_PROJ_DIM="${RET_PROJ_DIM:-128}"
RET_EARLY_STOP_PATIENCE="${RET_EARLY_STOP_PATIENCE:-30}"
RET_EARLY_STOP_MIN_DELTA="${RET_EARLY_STOP_MIN_DELTA:-0.0}"
RET_EARLY_STOP_MONITOR="${RET_EARLY_STOP_MONITOR:-r1}"
RET_HEAD_LR="${RET_HEAD_LR:-0.0003}"
RET_FT_HEAD_LR="${RET_FT_HEAD_LR:-0.0001}"
RET_FT_BACKBONE_LR="${RET_FT_BACKBONE_LR:-0.00001}"

# Segmentation settings
SEG_TRAIN_IMAGES="${SEG_TRAIN_IMAGES:-input/MPXA-Seg/train_images}"
SEG_TRAIN_MASKS="${SEG_TRAIN_MASKS:-input/MPXA-Seg/train_labels}"
SEG_VALID_IMAGES="${SEG_VALID_IMAGES:-input/MPXA-Seg/valid_images}"
SEG_VALID_MASKS="${SEG_VALID_MASKS:-input/MPXA-Seg/valid_labels}"
SEG_TEST_IMAGES="${SEG_TEST_IMAGES:-input/MPXA-Seg/test_images}"
SEG_TEST_MASKS="${SEG_TEST_MASKS:-input/MPXA-Seg/test_labels}"
SEG_CONFIG="${SEG_CONFIG:-configs_segmentation/mpxa-seg.yaml}"
SEG_IMG_W="${SEG_IMG_W:-640}"
SEG_IMG_H="${SEG_IMG_H:-640}"
SEG_BATCH_SIZE="${SEG_BATCH_SIZE:-48}"
SEG_EVAL_BATCH_SIZE="${SEG_EVAL_BATCH_SIZE:-64}"
SEG_MAX_EPOCHS="${SEG_MAX_EPOCHS:-1000}"
SEG_NUM_WORKERS="${SEG_NUM_WORKERS:-32}"
SEG_EARLY_STOP_PATIENCE="${SEG_EARLY_STOP_PATIENCE:-30}"
SEG_EARLY_STOP_MIN_DELTA="${SEG_EARLY_STOP_MIN_DELTA:-0.0}"
SEG_EARLY_STOP_MONITOR="${SEG_EARLY_STOP_MONITOR:-valid_miou}"
SEG_HEAD_LR="${SEG_HEAD_LR:-0.0001}"
SEG_FT_HEAD_LR="${SEG_FT_HEAD_LR:-0.0001}"
SEG_FT_BACKBONE_LR="${SEG_FT_BACKBONE_LR:-0.00001}"
SEG_BEST_CHECKPOINT="${SEG_BEST_CHECKPOINT:-best_model_iou.pth}"

TRAIN_ROOT="${EXPERIMENT_ROOT}/train"
EVAL_ROOT="${EXPERIMENT_ROOT}/eval"
SUMMARY_ROOT="${EXPERIMENT_ROOT}/summary/${MODEL_NAME}"

mkdir -p "$TRAIN_ROOT" "$EVAL_ROOT" "$SUMMARY_ROOT"

LOG_PREFIX=""
if [[ "$SMOKE_TEST" == "1" ]]; then
  LOG_PREFIX="[SMOKE] "
  UNFREEZE_BLOCKS=(0)
  PRETRAIN_INDEXES=(0)
  CLS_MAX_EPOCHS=1
  RET_MAX_EPOCHS=1
  SEG_MAX_EPOCHS=1
  CLS_EARLY_STOP_PATIENCE=1
  RET_EARLY_STOP_PATIENCE=1
  SEG_EARLY_STOP_PATIENCE=1
fi

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${LOG_PREFIX}$*"
}

run_logged() {
  local log_file="$1"
  shift
  mkdir -p "$(dirname "$log_file")"
  stdbuf -oL -eL "$@" 2>&1 | tee -a "$log_file"
}

has_glob_match() {
  local pattern="$1"
  compgen -G "$pattern" >/dev/null
}

is_cls_train_complete() {
  local train_dir="$1"
  [[ -f "${train_dir}/model.pth" ]]
}

is_ret_train_complete() {
  local train_dir="$1"
  [[ -f "${train_dir}/model.pth" ]]
}

is_seg_train_complete() {
  local train_dir="$1"
  [[ -f "${train_dir}/final_model.pth" ]]
}

is_cls_eval_complete() {
  local eval_dir="$1"
  has_glob_match "${eval_dir}/per_image_predictions_*.csv"
}

is_ret_eval_complete() {
  local eval_dir="$1"
  has_glob_match "${eval_dir}/retrieval_result_*.txt"
}

is_seg_eval_complete() {
  local eval_dir="$1"
  [[ -f "${eval_dir}/metrics.json" ]]
}

select_cls_ckpt() {
  local train_dir="$1"
  if [[ -f "${train_dir}/best_model.pth" ]]; then
    echo "${train_dir}/best_model.pth"
  else
    echo "${train_dir}/model.pth"
  fi
}

select_ret_ckpt() {
  local train_dir="$1"
  if [[ -f "${train_dir}/best_model.pth" ]]; then
    echo "${train_dir}/best_model.pth"
  else
    echo "${train_dir}/model.pth"
  fi
}

select_seg_ckpt() {
  local train_dir="$1"
  if [[ -f "${train_dir}/${SEG_BEST_CHECKPOINT}" ]]; then
    echo "${train_dir}/${SEG_BEST_CHECKPOINT}"
  else
    echo "${train_dir}/final_model.pth"
  fi
}

assert_file_exists() {
  local file_path="$1"
  if [[ ! -f "$file_path" ]]; then
    log "[ERROR] Required file not found: $file_path"
    exit 1
  fi
}

assert_dir_exists() {
  local dir_path="$1"
  if [[ ! -d "$dir_path" ]]; then
    log "[ERROR] Required directory not found: $dir_path"
    exit 1
  fi
}

run_cls_experiment() {
  local pretrain_key="$1"
  local pretrain_label="$2"
  local weights_path="$3"
  local unfreeze_blocks="$4"

  local run_tag
  run_tag=$(printf "u%02d" "$unfreeze_blocks")
  local train_dir="${TRAIN_ROOT}/1_cls/${MODEL_NAME}/${pretrain_key}/${run_tag}"
  local eval_dir="${EVAL_ROOT}/1_cls/${MODEL_NAME}/${pretrain_key}/${run_tag}"
  local train_log="${train_dir}/train.log"
  local eval_log="${eval_dir}/eval.log"
  local done_marker="${eval_dir}/.done"

  if [[ "$SKIP_EXISTING" == "1" && -f "$done_marker" ]]; then
    log "[SKIP][CLS] ${pretrain_label} unfreeze=${unfreeze_blocks} (already done)"
    return
  fi

  if [[ "$SKIP_EXISTING" == "1" ]] && is_cls_eval_complete "$eval_dir"; then
    log "[SKIP][CLS] ${pretrain_label} unfreeze=${unfreeze_blocks} (eval outputs already exist)"
    touch "$done_marker"
    return
  fi

  mkdir -p "$train_dir" "$eval_dir"

  local train_args=(
    python scripts/train/train_classifier.py
    --train-dir "$CLS_TRAIN_DIR"
    --valid-dir "$CLS_VALID_DIR"
    --weights "$weights_path"
    --repo-dir dinov3
    --model-name "$MODEL_NAME"
    --max-epochs "$CLS_MAX_EPOCHS"
    --batch-size "$CLS_BATCH_SIZE"
    --num-workers "$CLS_NUM_WORKERS"
    --early-stopping
    --early-stopping-patience "$CLS_EARLY_STOP_PATIENCE"
    --early-stopping-min-delta "$CLS_EARLY_STOP_MIN_DELTA"
    --early-stopping-monitor "$CLS_EARLY_STOP_MONITOR"
    --out-dir "$train_dir"
    --config "$CLS_CONFIG"
    --unfreeze-blocks "$unfreeze_blocks"
  )

  if (( unfreeze_blocks == 0 )); then
    train_args+=(-lr "$CLS_HEAD_LR")
  else
    train_args+=(--fine-tune -lr "$CLS_FT_HEAD_LR" --backbone-lr "$CLS_FT_BACKBONE_LR")
  fi

  if [[ "$SKIP_EXISTING" == "1" ]] && is_cls_train_complete "$train_dir"; then
    log "[SKIP][CLS] ${pretrain_label} unfreeze=${unfreeze_blocks} (train checkpoint exists)"
  else
    log "[RUN][CLS] ${pretrain_label} unfreeze=${unfreeze_blocks}"
    run_logged "$train_log" "${train_args[@]}"
  fi

  local ckpt_path
  ckpt_path="$(select_cls_ckpt "$train_dir")"
  assert_file_exists "$ckpt_path"

  if [[ "$SKIP_EXISTING" == "1" ]] && is_cls_eval_complete "$eval_dir"; then
    log "[SKIP][CLS] ${pretrain_label} unfreeze=${unfreeze_blocks} (eval outputs already exist)"
  else
    run_logged "$eval_log" \
      python scripts/eval/eval_classifier.py \
      --weights "$ckpt_path" \
      --input "$CLS_TEST_DIR" \
      --config "$CLS_CONFIG" \
      --model-name "$MODEL_NAME" \
      --out-dir "$eval_dir"
  fi

  touch "$done_marker"
}

run_ret_experiment() {
  local pretrain_key="$1"
  local pretrain_label="$2"
  local weights_path="$3"
  local unfreeze_blocks="$4"

  local run_tag
  run_tag=$(printf "u%02d" "$unfreeze_blocks")
  local train_dir="${TRAIN_ROOT}/2_ret/${MODEL_NAME}/${pretrain_key}/${run_tag}"
  local eval_dir="${EVAL_ROOT}/2_ret/${MODEL_NAME}/${pretrain_key}/${run_tag}"
  local train_log="${train_dir}/train.log"
  local eval_log="${eval_dir}/eval.log"
  local done_marker="${eval_dir}/.done"

  if [[ "$SKIP_EXISTING" == "1" && -f "$done_marker" ]]; then
    log "[SKIP][RET] ${pretrain_label} unfreeze=${unfreeze_blocks} (already done)"
    return
  fi

  if [[ "$SKIP_EXISTING" == "1" ]] && is_ret_eval_complete "$eval_dir"; then
    log "[SKIP][RET] ${pretrain_label} unfreeze=${unfreeze_blocks} (eval outputs already exist)"
    touch "$done_marker"
    return
  fi

  mkdir -p "$train_dir" "$eval_dir"

  local train_args=(
    python scripts/train/train_retrieval.py
    --train-dir "$RET_TRAIN_DIR"
    --valid-dir "$RET_VALID_DIR"
    --weights "$weights_path"
    --repo-dir dinov3
    --model-name "$MODEL_NAME"
    --max-epochs "$RET_MAX_EPOCHS"
    --out-dir "$train_dir"
    --config "$RET_CONFIG"
    --num-workers "$RET_NUM_WORKERS"
    --batch-size "$RET_BATCH_SIZE"
    --proj-dim "$RET_PROJ_DIM"
    --early-stopping
    --early-stopping-patience "$RET_EARLY_STOP_PATIENCE"
    --early-stopping-min-delta "$RET_EARLY_STOP_MIN_DELTA"
    --early-stopping-monitor "$RET_EARLY_STOP_MONITOR"
    --unfreeze-blocks "$unfreeze_blocks"
  )

  if (( unfreeze_blocks == 0 )); then
    train_args+=(--lr "$RET_HEAD_LR")
  else
    train_args+=(--fine-tune --lr "$RET_FT_HEAD_LR" --backbone-lr "$RET_FT_BACKBONE_LR")
  fi

  if [[ "$SKIP_EXISTING" == "1" ]] && is_ret_train_complete "$train_dir"; then
    log "[SKIP][RET] ${pretrain_label} unfreeze=${unfreeze_blocks} (train checkpoint exists)"
  else
    log "[RUN][RET] ${pretrain_label} unfreeze=${unfreeze_blocks}"
    run_logged "$train_log" "${train_args[@]}"
  fi

  local ckpt_path
  ckpt_path="$(select_ret_ckpt "$train_dir")"
  assert_file_exists "$ckpt_path"

  if [[ "$SKIP_EXISTING" == "1" ]] && is_ret_eval_complete "$eval_dir"; then
    log "[SKIP][RET] ${pretrain_label} unfreeze=${unfreeze_blocks} (eval outputs already exist)"
  else
    run_logged "$eval_log" \
      python scripts/eval/eval_retrieval.py \
      --input "$RET_TEST_DIR" \
      --config "$RET_CONFIG" \
      --model-name "$MODEL_NAME" \
      --repo-dir dinov3 \
      --weights "$ckpt_path" \
      --proj-dim "$RET_PROJ_DIM" \
      --out-dir "$eval_dir"
  fi

  touch "$done_marker"
}

run_seg_experiment() {
  local pretrain_key="$1"
  local pretrain_label="$2"
  local weights_path="$3"
  local unfreeze_blocks="$4"

  local run_tag
  run_tag=$(printf "u%02d" "$unfreeze_blocks")
  local train_dir="${TRAIN_ROOT}/3_seg/${MODEL_NAME}/${pretrain_key}/${run_tag}"
  local eval_dir="${EVAL_ROOT}/3_seg/${MODEL_NAME}/${pretrain_key}/${run_tag}"
  local train_log="${train_dir}/train.log"
  local eval_log="${eval_dir}/eval.log"
  local done_marker="${eval_dir}/.done"

  if [[ "$SKIP_EXISTING" == "1" && -f "$done_marker" ]]; then
    log "[SKIP][SEG] ${pretrain_label} unfreeze=${unfreeze_blocks} (already done)"
    return
  fi

  if [[ "$SKIP_EXISTING" == "1" ]] && is_seg_eval_complete "$eval_dir"; then
    log "[SKIP][SEG] ${pretrain_label} unfreeze=${unfreeze_blocks} (eval outputs already exist)"
    touch "$done_marker"
    return
  fi

  mkdir -p "$train_dir" "$eval_dir"

  local train_args=(
    python scripts/train/train_segmentation.py
    --train-images "$SEG_TRAIN_IMAGES"
    --train-masks "$SEG_TRAIN_MASKS"
    --valid-images "$SEG_VALID_IMAGES"
    --valid-masks "$SEG_VALID_MASKS"
    --config "$SEG_CONFIG"
    --weights "$weights_path"
    --repo-dir dinov3
    --model-name "$MODEL_NAME"
    --max-epoch "$SEG_MAX_EPOCHS"
    --out-dir "$train_dir"
    --imgsz "$SEG_IMG_W" "$SEG_IMG_H"
    --batch "$SEG_BATCH_SIZE"
    --num-workers "$SEG_NUM_WORKERS"
    --early-stopping
    --early-stopping-patience "$SEG_EARLY_STOP_PATIENCE"
    --early-stopping-min-delta "$SEG_EARLY_STOP_MIN_DELTA"
    --early-stopping-monitor "$SEG_EARLY_STOP_MONITOR"
    --unfreeze-blocks "$unfreeze_blocks"
  )

  if (( unfreeze_blocks == 0 )); then
    train_args+=(--lr "$SEG_HEAD_LR")
  else
    train_args+=(--fine-tune --lr "$SEG_FT_HEAD_LR" --backbone-lr "$SEG_FT_BACKBONE_LR")
  fi

  if [[ "$SKIP_EXISTING" == "1" ]] && is_seg_train_complete "$train_dir"; then
    log "[SKIP][SEG] ${pretrain_label} unfreeze=${unfreeze_blocks} (train checkpoint exists)"
  else
    log "[RUN][SEG] ${pretrain_label} unfreeze=${unfreeze_blocks}"
    run_logged "$train_log" "${train_args[@]}"
  fi

  local ckpt_path
  ckpt_path="$(select_seg_ckpt "$train_dir")"
  assert_file_exists "$ckpt_path"

  if [[ "$SKIP_EXISTING" == "1" ]] && is_seg_eval_complete "$eval_dir"; then
    log "[SKIP][SEG] ${pretrain_label} unfreeze=${unfreeze_blocks} (eval outputs already exist)"
  else
    run_logged "$eval_log" \
      python scripts/eval/eval_segmentation.py \
      --eval-images "$SEG_TEST_IMAGES" \
      --eval-masks "$SEG_TEST_MASKS" \
      --config "$SEG_CONFIG" \
      --weights "$ckpt_path" \
      --out-dir "$eval_dir" \
      --imgsz "$SEG_IMG_W" "$SEG_IMG_H" \
      --batch "$SEG_EVAL_BATCH_SIZE" \
      --num-workers "$SEG_NUM_WORKERS" \
      --model-name "$MODEL_NAME"
  fi

  touch "$done_marker"
}

log "===== FT Curve Batch Start ====="
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
log "MODEL_NAME=${MODEL_NAME}"
log "EXPERIMENT_ROOT=${EXPERIMENT_ROOT}"
log "SKIP_EXISTING=${SKIP_EXISTING}"

for idx in "${PRETRAIN_INDEXES[@]}"; do
  if (( idx < 0 || idx >= ${#PRETRAIN_KEYS[@]} )); then
    log "[ERROR] Invalid PRETRAIN_INDEXES entry: ${idx}"
    exit 1
  fi
  assert_file_exists "${PRETRAIN_WEIGHTS[$idx]}"
done

assert_dir_exists "$CLS_TRAIN_DIR"
assert_dir_exists "$CLS_VALID_DIR"
assert_dir_exists "$CLS_TEST_DIR"
assert_dir_exists "$RET_TRAIN_DIR"
assert_dir_exists "$RET_VALID_DIR"
assert_dir_exists "$RET_TEST_DIR"
assert_dir_exists "$SEG_TRAIN_IMAGES"
assert_dir_exists "$SEG_TRAIN_MASKS"
assert_dir_exists "$SEG_VALID_IMAGES"
assert_dir_exists "$SEG_VALID_MASKS"
assert_dir_exists "$SEG_TEST_IMAGES"
assert_dir_exists "$SEG_TEST_MASKS"

for idx in "${PRETRAIN_INDEXES[@]}"; do
  pretrain_key="${PRETRAIN_KEYS[$idx]}"
  pretrain_label="${PRETRAIN_LABELS[$idx]}"
  pretrain_weights="${PRETRAIN_WEIGHTS[$idx]}"

  for unfreeze_blocks in "${UNFREEZE_BLOCKS[@]}"; do
    run_cls_experiment "$pretrain_key" "$pretrain_label" "$pretrain_weights" "$unfreeze_blocks"
  done
done

for idx in "${PRETRAIN_INDEXES[@]}"; do
  pretrain_key="${PRETRAIN_KEYS[$idx]}"
  pretrain_label="${PRETRAIN_LABELS[$idx]}"
  pretrain_weights="${PRETRAIN_WEIGHTS[$idx]}"

  for unfreeze_blocks in "${UNFREEZE_BLOCKS[@]}"; do
    run_ret_experiment "$pretrain_key" "$pretrain_label" "$pretrain_weights" "$unfreeze_blocks"
  done
done

for idx in "${PRETRAIN_INDEXES[@]}"; do
  pretrain_key="${PRETRAIN_KEYS[$idx]}"
  pretrain_label="${PRETRAIN_LABELS[$idx]}"
  pretrain_weights="${PRETRAIN_WEIGHTS[$idx]}"

  for unfreeze_blocks in "${UNFREEZE_BLOCKS[@]}"; do
    run_seg_experiment "$pretrain_key" "$pretrain_label" "$pretrain_weights" "$unfreeze_blocks"
  done
done

log "Generating summary curves and crossover table"
python scripts/analysis/ft_compare/summarize_ft_curves.py \
  --root "$EXPERIMENT_ROOT" \
  --model-name "$MODEL_NAME" \
  --out-dir "$SUMMARY_ROOT" \
  --unfreeze-blocks "${UNFREEZE_BLOCKS[@]}"

log "===== FT Curve Batch Done ====="
log "Summary directory: ${SUMMARY_ROOT}"
