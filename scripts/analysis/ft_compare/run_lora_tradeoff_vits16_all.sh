#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PYTHONPATH="${ROOT_DIR}/scripts/train:${PYTHONPATH}"

MODEL_NAME="${MODEL_NAME:-dinov3_vits16}"
REPO_DIR="${REPO_DIR:-dinov3}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-outputs/5_lora_tradeoff}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SMOKE_TEST="${SMOKE_TEST:-0}"
ETA_HEARTBEAT_SEC="${ETA_HEARTBEAT_SEC:-60}"
ETA_MIN_EXEC_SEC="${ETA_MIN_EXEC_SEC:-30}"

UNFREEZE_BLOCKS_STR="${UNFREEZE_BLOCKS_STR:-0 2 4 12}"
read -r -a UNFREEZE_BLOCKS <<<"$UNFREEZE_BLOCKS_STR"
TASKS_STR="${TASKS_STR:-cls ret seg}"
read -r -a TASKS <<<"$TASKS_STR"
RUN_ABLATION="${RUN_ABLATION:-1}"

GENERAL_WEIGHTS="${GENERAL_WEIGHTS:-dinov3/output/a6000/1_pretrain/dinov3_vits16/2_imagenet1k/3_stage3_high_res_adapt/eval/training_29999/teacher_checkpoint.pth}"
DOMAIN_WEIGHTS="${DOMAIN_WEIGHTS:-dinov3/output/a6000/1_pretrain/dinov3_vits16/3_cagcontfm3m/3_stage3_high_res_adapt/eval/training_29999/teacher_checkpoint.pth}"

# Classification
CLS_TRAIN_DIR="${CLS_TRAIN_DIR:-input/Stent-Ref/train/}"
CLS_VALID_DIR="${CLS_VALID_DIR:-input/Stent-Ref/valid/}"
CLS_TEST_DIR="${CLS_TEST_DIR:-input/Stent-Ref/test/}"
CLS_CONFIG="${CLS_CONFIG:-configs_classification/stent.yaml}"
CLS_BATCH_SIZE="${CLS_BATCH_SIZE:-32}"
CLS_MAX_EPOCHS="${CLS_MAX_EPOCHS:-1000}"
CLS_NUM_WORKERS="${CLS_NUM_WORKERS:-24}"
CLS_EARLY_STOP_PATIENCE="${CLS_EARLY_STOP_PATIENCE:-30}"
CLS_EARLY_STOP_MIN_DELTA="${CLS_EARLY_STOP_MIN_DELTA:-0.0}"
CLS_EARLY_STOP_MONITOR="${CLS_EARLY_STOP_MONITOR:-val_macro_f1}"
CLS_EVAL_THRESHOLD_MODE="${CLS_EVAL_THRESHOLD_MODE:-tune_val_macro_f1}"
CLS_EVAL_THRESHOLD="${CLS_EVAL_THRESHOLD:-0.5}"
CLS_EVAL_THRESHOLD_POS_CLASS="${CLS_EVAL_THRESHOLD_POS_CLASS:-stent}"
CLS_HEAD_LR="${CLS_HEAD_LR:-0.001}"
CLS_FT_HEAD_LR="${CLS_FT_HEAD_LR:-0.0001}"
CLS_FT_BACKBONE_LR="${CLS_FT_BACKBONE_LR:-0.00001}"

# Retrieval
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

# Segmentation
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

LORA_TARGET="attn_qkv_proj"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"

TRAIN_ROOT="${EXPERIMENT_ROOT}/train"
EVAL_ROOT="${EXPERIMENT_ROOT}/eval"
SUMMARY_ROOT="${EXPERIMENT_ROOT}/summary"
REPORT_ROOT="${EXPERIMENT_ROOT}/report"
CONFIG_ROOT="${EXPERIMENT_ROOT}/configs"
ALIGN_CONFIG="${CONFIG_ROOT}/alignment_selected.yaml"
MANIFEST="${CONFIG_ROOT}/run_manifest.csv"

mkdir -p "$TRAIN_ROOT" "$EVAL_ROOT" "$SUMMARY_ROOT" "$REPORT_ROOT" "$CONFIG_ROOT"

LOG_PREFIX=""
if [[ "$SMOKE_TEST" == "1" ]]; then
  LOG_PREFIX="[SMOKE] "
  UNFREEZE_BLOCKS=(0)
  CLS_MAX_EPOCHS=1
  RET_MAX_EPOCHS=1
  SEG_MAX_EPOCHS=1
  CLS_EARLY_STOP_PATIENCE=1
  RET_EARLY_STOP_PATIENCE=1
  SEG_EARLY_STOP_PATIENCE=1
fi

BATCH_START_TS="$(date +%s)"
RUNS_TOTAL=0
RUNS_DONE=0
RUNS_ELAPSED_SUM=0
RUNS_EXEC_DONE=0
RUNS_EXEC_ELAPSED_SUM=0
RUNS_ETA_DONE=0
RUNS_ETA_ELAPSED_SUM=0
RUN_STEP_STATUS="unknown"

format_seconds() {
  local seconds="$1"
  if (( seconds < 0 )); then
    seconds=0
  fi
  local h=$((seconds / 3600))
  local m=$(((seconds % 3600) / 60))
  local s=$((seconds % 60))
  printf "%02d:%02d:%02d" "$h" "$m" "$s"
}

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${LOG_PREFIX}$*"
}

task_enabled() {
  local target="$1"
  for t in "${TASKS[@]}"; do
    if [[ "$t" == "$target" ]]; then
      return 0
    fi
  done
  return 1
}

run_with_eta() {
  local label="$1"
  shift

  local step_start_ts
  step_start_ts="$(date +%s)"
  RUN_STEP_STATUS="unknown"

  log "[RUN $((RUNS_DONE + 1))/${RUNS_TOTAL}] START ${label}"
  "$@" &
  local run_pid=$!

  local next_heartbeat_ts=$((step_start_ts + ETA_HEARTBEAT_SEC))
  while kill -0 "$run_pid" 2>/dev/null; do
    local now_ts
    now_ts="$(date +%s)"
    if (( now_ts >= next_heartbeat_ts )); then
      local step_elapsed=$((now_ts - step_start_ts))
      local remaining_runs_after_current=$((RUNS_TOTAL - RUNS_DONE - 1))

      if (( RUNS_ETA_DONE > 0 )); then
        local avg_eta_run=$((RUNS_ETA_ELAPSED_SUM / RUNS_ETA_DONE))
        local remaining_current_est=$((avg_eta_run - step_elapsed))
        if (( remaining_current_est < 0 )); then
          remaining_current_est=0
        fi
        local remaining_est=$((remaining_current_est + avg_eta_run * remaining_runs_after_current))
        local eta_ts=$((now_ts + remaining_est))
        local eta_str
        eta_str="$(date -d "@${eta_ts}" '+%Y-%m-%d %H:%M:%S')"
        log "[ETA] ${label} running $(format_seconds "$step_elapsed") | remaining~$(format_seconds "$remaining_est") | eta=${eta_str} | avg_eta/run=$(format_seconds "$avg_eta_run")"
      else
        local remaining_est=$((step_elapsed * remaining_runs_after_current))
        local eta_ts=$((now_ts + remaining_est))
        local eta_str
        eta_str="$(date -d "@${eta_ts}" '+%Y-%m-%d %H:%M:%S')"
        log "[ETA] ${label} running $(format_seconds "$step_elapsed") | remaining~$(format_seconds "$remaining_est") | eta=${eta_str} | provisional(current run only; avg_eta pending)"
      fi
      next_heartbeat_ts=$((now_ts + ETA_HEARTBEAT_SEC))
    fi
    sleep 1
  done

  set +e
  wait "$run_pid"
  local run_rc=$?
  set -e
  if (( run_rc != 0 )); then
    log "[RUN $((RUNS_DONE + 1))/${RUNS_TOTAL}] FAIL ${label} | exit_code=${run_rc}"
    return "$run_rc"
  fi

  local step_end_ts
  step_end_ts="$(date +%s)"
  local step_elapsed=$((step_end_ts - step_start_ts))

  RUNS_DONE=$((RUNS_DONE + 1))
  RUNS_ELAPSED_SUM=$((RUNS_ELAPSED_SUM + step_elapsed))
  if [[ "$RUN_STEP_STATUS" != "skipped" ]]; then
    RUNS_EXEC_DONE=$((RUNS_EXEC_DONE + 1))
    RUNS_EXEC_ELAPSED_SUM=$((RUNS_EXEC_ELAPSED_SUM + step_elapsed))
    if (( step_elapsed >= ETA_MIN_EXEC_SEC )); then
      RUNS_ETA_DONE=$((RUNS_ETA_DONE + 1))
      RUNS_ETA_ELAPSED_SUM=$((RUNS_ETA_ELAPSED_SUM + step_elapsed))
    fi
  fi

  local total_elapsed=$((step_end_ts - BATCH_START_TS))
  local remaining_runs=$((RUNS_TOTAL - RUNS_DONE))
  local avg_all=$((RUNS_ELAPSED_SUM / RUNS_DONE))
  local avg_exec=0
  local avg_eta=0
  local remaining_est=0
  local avg_label="provisional_from_last_run"
  local avg_value="$step_elapsed"
  if (( RUNS_EXEC_DONE > 0 )); then
    avg_exec=$((RUNS_EXEC_ELAPSED_SUM / RUNS_EXEC_DONE))
  fi
  if (( RUNS_ETA_DONE > 0 )); then
    avg_eta=$((RUNS_ETA_ELAPSED_SUM / RUNS_ETA_DONE))
    remaining_est=$((avg_eta * remaining_runs))
    avg_label="avg_eta"
    avg_value="$avg_eta"
  else
    remaining_est=$((step_elapsed * remaining_runs))
    if (( RUNS_EXEC_DONE > 0 )); then
      avg_label="avg_exec_insufficient_for_eta"
      avg_value="$avg_exec"
    fi
  fi
  local eta_ts=$((step_end_ts + remaining_est))
  local eta_str
  eta_str="$(date -d "@${eta_ts}" '+%Y-%m-%d %H:%M:%S')"

  log "[RUN ${RUNS_DONE}/${RUNS_TOTAL}] DONE ${label} | status=${RUN_STEP_STATUS} | took=$(format_seconds "$step_elapsed") | elapsed=$(format_seconds "$total_elapsed") | remaining~$(format_seconds "$remaining_est") | eta=${eta_str} | ${avg_label}/run=$(format_seconds "$avg_value") | avg_all/run=$(format_seconds "$avg_all")"
}

run_logged() {
  local log_file="$1"
  shift
  mkdir -p "$(dirname "$log_file")"
  PYTHONUNBUFFERED=1 stdbuf -oL -eL "$@" 2>&1 | tee -a "$log_file"
}

has_glob_match() {
  local pattern="$1"
  compgen -G "$pattern" >/dev/null
}

assert_file_exists() {
  local file_path="$1"
  if [[ ! -f "$file_path" ]]; then
    log "[ERROR] Missing file: $file_path"
    exit 1
  fi
}

assert_dir_exists() {
  local dir_path="$1"
  if [[ ! -d "$dir_path" ]]; then
    log "[ERROR] Missing directory: $dir_path"
    exit 1
  fi
}

is_cls_train_complete() { [[ -f "$1/model.pth" ]]; }
is_ret_train_complete() { [[ -f "$1/model.pth" ]]; }
is_seg_train_complete() { [[ -f "$1/final_model.pth" ]]; }

is_cls_eval_complete() { has_glob_match "$1/per_image_predictions_*.csv"; }
is_ret_eval_complete() { has_glob_match "$1/retrieval_result_*.txt"; }
is_seg_eval_complete() { [[ -f "$1/metrics.json" ]]; }

select_cls_ckpt() {
  if [[ -f "$1/best_model.pth" ]]; then
    echo "$1/best_model.pth"
  else
    echo "$1/model.pth"
  fi
}

select_ret_ckpt() {
  if [[ -f "$1/best_model.pth" ]]; then
    echo "$1/best_model.pth"
  else
    echo "$1/model.pth"
  fi
}

select_seg_ckpt() {
  if [[ -f "$1/${SEG_BEST_CHECKPOINT}" ]]; then
    echo "$1/${SEG_BEST_CHECKPOINT}"
  else
    echo "$1/final_model.pth"
  fi
}

append_manifest() {
  local kind="$1"
  local task="$2"
  local case_id="$3"
  local method="$4"
  local backbone_type="$5"
  local unfreeze_blocks="$6"
  local lora_rank="$7"
  local train_dir="$8"
  local eval_dir="$9"
  echo "${kind},${task},${case_id},${method},${backbone_type},${unfreeze_blocks},${lora_rank},${train_dir},${eval_dir}" >> "$MANIFEST"
}

read_align_value() {
  local task="$1"
  local key="$2"
  python -c "import yaml; d=yaml.safe_load(open('${ALIGN_CONFIG}')); print(d['tasks']['${task}']['${key}'])"
}

run_cls_one() {
  local train_dir="$1"
  local eval_dir="$2"
  local weights="$3"
  local unfreeze_blocks="$4"
  local head_size="$5"
  local head_hidden_dim="$6"
  local enable_lora="$7"
  local lora_rank="$8"

  local train_log="${train_dir}/train.log"
  local eval_log="${eval_dir}/eval.log"
  local done_marker="${eval_dir}/.done"
  RUN_STEP_STATUS="unknown"

  if [[ "$SKIP_EXISTING" == "1" && -f "$done_marker" ]]; then
    RUN_STEP_STATUS="skipped"
    return
  fi
  if [[ "$SKIP_EXISTING" == "1" ]] && is_cls_eval_complete "$eval_dir"; then
    touch "$done_marker"
    RUN_STEP_STATUS="skipped"
    return
  fi

  mkdir -p "$train_dir" "$eval_dir"

  local args=(
    python scripts/train/train_classifier.py
    --train-dir "$CLS_TRAIN_DIR"
    --valid-dir "$CLS_VALID_DIR"
    --weights "$weights"
    --repo-dir "$REPO_DIR"
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
    --head-size "$head_size"
    --save-config-json "${train_dir}/run_config.json"
    --unfreeze-blocks "$unfreeze_blocks"
  )

  if [[ "$head_size" == "big" ]]; then
    args+=(--head-hidden-dim "$head_hidden_dim")
  fi
  if [[ "$enable_lora" == "1" ]]; then
    args+=(
      --enable-lora
      --lora-rank "$lora_rank"
      --lora-alpha "$lora_rank"
      --lora-dropout "$LORA_DROPOUT"
      --lora-target "$LORA_TARGET"
    )
  fi

  if (( unfreeze_blocks == 0 )); then
    args+=(-lr "$CLS_HEAD_LR")
  else
    args+=(--fine-tune -lr "$CLS_FT_HEAD_LR" --backbone-lr "$CLS_FT_BACKBONE_LR")
  fi

  if [[ "$SKIP_EXISTING" != "1" ]] || ! is_cls_train_complete "$train_dir"; then
    run_logged "$train_log" "${args[@]}"
  fi

  local ckpt
  ckpt="$(select_cls_ckpt "$train_dir")"
  assert_file_exists "$ckpt"

  if [[ "$SKIP_EXISTING" != "1" ]] || ! is_cls_eval_complete "$eval_dir"; then
    local eval_args=(
      python scripts/eval/eval_classifier.py
      --weights "$ckpt"
      --input "$CLS_TEST_DIR"
      --config "$CLS_CONFIG"
      --model-name "$MODEL_NAME"
      --repo-dir "$REPO_DIR"
      --out-dir "$eval_dir"
      --threshold-mode "$CLS_EVAL_THRESHOLD_MODE"
    )

    if [[ -n "$CLS_EVAL_THRESHOLD_POS_CLASS" ]]; then
      eval_args+=(--threshold-positive-class "$CLS_EVAL_THRESHOLD_POS_CLASS")
    fi
    if [[ "$CLS_EVAL_THRESHOLD_MODE" == "fixed" ]]; then
      eval_args+=(--threshold "$CLS_EVAL_THRESHOLD")
    elif [[ "$CLS_EVAL_THRESHOLD_MODE" == "tune_val_macro_f1" ]]; then
      eval_args+=(--threshold-val-input "$CLS_VALID_DIR")
    fi

    run_logged "$eval_log" "${eval_args[@]}"
  fi

  touch "$done_marker"
  RUN_STEP_STATUS="executed"
}

run_ret_one() {
  local train_dir="$1"
  local eval_dir="$2"
  local weights="$3"
  local unfreeze_blocks="$4"
  local head_size="$5"
  local head_hidden_dim="$6"
  local enable_lora="$7"
  local lora_rank="$8"

  local train_log="${train_dir}/train.log"
  local eval_log="${eval_dir}/eval.log"
  local done_marker="${eval_dir}/.done"
  RUN_STEP_STATUS="unknown"

  if [[ "$SKIP_EXISTING" == "1" && -f "$done_marker" ]]; then
    RUN_STEP_STATUS="skipped"
    return
  fi
  if [[ "$SKIP_EXISTING" == "1" ]] && is_ret_eval_complete "$eval_dir"; then
    touch "$done_marker"
    RUN_STEP_STATUS="skipped"
    return
  fi

  mkdir -p "$train_dir" "$eval_dir"

  local args=(
    python scripts/train/train_retrieval.py
    --train-dir "$RET_TRAIN_DIR"
    --valid-dir "$RET_VALID_DIR"
    --weights "$weights"
    --repo-dir "$REPO_DIR"
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
    --head-size "$head_size"
    --save-config-json "${train_dir}/run_config.json"
    --unfreeze-blocks "$unfreeze_blocks"
  )

  if [[ "$head_size" == "big" ]]; then
    args+=(--head-hidden-dim "$head_hidden_dim")
  fi
  if [[ "$enable_lora" == "1" ]]; then
    args+=(
      --enable-lora
      --lora-rank "$lora_rank"
      --lora-alpha "$lora_rank"
      --lora-dropout "$LORA_DROPOUT"
      --lora-target "$LORA_TARGET"
    )
  fi

  if (( unfreeze_blocks == 0 )); then
    args+=(--lr "$RET_HEAD_LR")
  else
    args+=(--fine-tune --lr "$RET_FT_HEAD_LR" --backbone-lr "$RET_FT_BACKBONE_LR")
  fi

  if [[ "$SKIP_EXISTING" != "1" ]] || ! is_ret_train_complete "$train_dir"; then
    run_logged "$train_log" "${args[@]}"
  fi

  local ckpt
  ckpt="$(select_ret_ckpt "$train_dir")"
  assert_file_exists "$ckpt"

  if [[ "$SKIP_EXISTING" != "1" ]] || ! is_ret_eval_complete "$eval_dir"; then
    run_logged "$eval_log" \
      python scripts/eval/eval_retrieval.py \
      --input "$RET_TEST_DIR" \
      --config "$RET_CONFIG" \
      --model-name "$MODEL_NAME" \
      --repo-dir "$REPO_DIR" \
      --weights "$ckpt" \
      --proj-dim "$RET_PROJ_DIM" \
      --out-dir "$eval_dir"
  fi

  touch "$done_marker"
  RUN_STEP_STATUS="executed"
}

run_seg_one() {
  local train_dir="$1"
  local eval_dir="$2"
  local weights="$3"
  local unfreeze_blocks="$4"
  local head_size="$5"
  local decoder_hidden_channels="$6"
  local enable_lora="$7"
  local lora_rank="$8"

  local train_log="${train_dir}/train.log"
  local eval_log="${eval_dir}/eval.log"
  local done_marker="${eval_dir}/.done"
  RUN_STEP_STATUS="unknown"

  if [[ "$SKIP_EXISTING" == "1" && -f "$done_marker" ]]; then
    RUN_STEP_STATUS="skipped"
    return
  fi
  if [[ "$SKIP_EXISTING" == "1" ]] && is_seg_eval_complete "$eval_dir"; then
    touch "$done_marker"
    RUN_STEP_STATUS="skipped"
    return
  fi

  mkdir -p "$train_dir" "$eval_dir"

  local args=(
    python scripts/train/train_segmentation.py
    --train-images "$SEG_TRAIN_IMAGES"
    --train-masks "$SEG_TRAIN_MASKS"
    --valid-images "$SEG_VALID_IMAGES"
    --valid-masks "$SEG_VALID_MASKS"
    --config "$SEG_CONFIG"
    --weights "$weights"
    --repo-dir "$REPO_DIR"
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
    --head-size "$head_size"
    --decoder-hidden-channels "$decoder_hidden_channels"
    --save-config-json "${train_dir}/run_config.json"
    --unfreeze-blocks "$unfreeze_blocks"
  )

  if [[ "$enable_lora" == "1" ]]; then
    args+=(
      --enable-lora
      --lora-rank "$lora_rank"
      --lora-alpha "$lora_rank"
      --lora-dropout "$LORA_DROPOUT"
      --lora-target "$LORA_TARGET"
    )
  fi

  if (( unfreeze_blocks == 0 )); then
    args+=(--lr "$SEG_HEAD_LR")
  else
    args+=(--fine-tune --lr "$SEG_FT_HEAD_LR" --backbone-lr "$SEG_FT_BACKBONE_LR")
  fi

  if [[ "$SKIP_EXISTING" != "1" ]] || ! is_seg_train_complete "$train_dir"; then
    run_logged "$train_log" "${args[@]}"
  fi

  local ckpt
  ckpt="$(select_seg_ckpt "$train_dir")"
  assert_file_exists "$ckpt"

  if [[ "$SKIP_EXISTING" != "1" ]] || ! is_seg_eval_complete "$eval_dir"; then
    run_logged "$eval_log" \
      python scripts/eval/eval_segmentation.py \
      --eval-images "$SEG_TEST_IMAGES" \
      --eval-masks "$SEG_TEST_MASKS" \
      --config "$SEG_CONFIG" \
      --weights "$ckpt" \
      --out-dir "$eval_dir" \
      --imgsz "$SEG_IMG_W" "$SEG_IMG_H" \
      --batch "$SEG_EVAL_BATCH_SIZE" \
      --num-workers "$SEG_NUM_WORKERS" \
      --model-name "$MODEL_NAME" \
      --repo-dir "$REPO_DIR"
  fi

  touch "$done_marker"
  RUN_STEP_STATUS="executed"
}

log "===== LoRA Trade-off Batch Start ====="
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
log "MODEL_NAME=${MODEL_NAME}"
log "REPO_DIR=${REPO_DIR}"
log "EXPERIMENT_ROOT=${EXPERIMENT_ROOT}"
log "SKIP_EXISTING=${SKIP_EXISTING}"
log "TASKS=${TASKS_STR}"
log "RUN_ABLATION=${RUN_ABLATION}"
log "ETA output: enabled (rolling average per run)"

for task in "${TASKS[@]}"; do
  case "$task" in
    cls|ret|seg) ;;
    *)
      log "[ERROR] Invalid task in TASKS_STR: $task (allowed: cls ret seg)"
      exit 1
      ;;
  esac
done

ENABLED_TASK_COUNT=0
for task in cls ret seg; do
  if task_enabled "$task"; then
    ENABLED_TASK_COUNT=$((ENABLED_TASK_COUNT + 1))
  fi
done
if (( ENABLED_TASK_COUNT == 0 )); then
  log "[ERROR] No tasks enabled. Set TASKS_STR to one or more of: cls ret seg"
  exit 1
fi

if [[ "$SMOKE_TEST" == "1" ]]; then
  RUNS_TOTAL=$((ENABLED_TASK_COUNT * ${#UNFREEZE_BLOCKS[@]}))
else
  RUNS_TOTAL=$((ENABLED_TASK_COUNT * 3 * ${#UNFREEZE_BLOCKS[@]}))
  if [[ "$RUN_ABLATION" == "1" ]]; then
    RUNS_TOTAL=$((RUNS_TOTAL + ENABLED_TASK_COUNT))
  fi
fi
log "Planned run units: ${RUNS_TOTAL}"
log "ETA will stabilize after a few completed runs."

assert_dir_exists "$REPO_DIR"
assert_file_exists "$GENERAL_WEIGHTS"
assert_file_exists "$DOMAIN_WEIGHTS"
if task_enabled cls; then
  assert_dir_exists "$CLS_TRAIN_DIR"
  assert_dir_exists "$CLS_VALID_DIR"
  assert_dir_exists "$CLS_TEST_DIR"
fi
if task_enabled ret; then
  assert_dir_exists "$RET_TRAIN_DIR"
  assert_dir_exists "$RET_VALID_DIR"
  assert_dir_exists "$RET_TEST_DIR"
fi
if task_enabled seg; then
  assert_dir_exists "$SEG_TRAIN_IMAGES"
  assert_dir_exists "$SEG_TRAIN_MASKS"
  assert_dir_exists "$SEG_VALID_IMAGES"
  assert_dir_exists "$SEG_VALID_MASKS"
  assert_dir_exists "$SEG_TEST_IMAGES"
  assert_dir_exists "$SEG_TEST_MASKS"
fi

log "Calibrating parameter alignment"
python scripts/exp/calibrate_tradeoff_alignment.py \
  --model-name "$MODEL_NAME" \
  --cls-config "$CLS_CONFIG" \
  --seg-config "$SEG_CONFIG" \
  --ret-proj-dim "$RET_PROJ_DIM" \
  --out-config "$ALIGN_CONFIG"

CLS_BIG_HIDDEN="$(read_align_value cls selected_big_head_hidden_dim)"
RET_BIG_HIDDEN="$(read_align_value ret selected_big_head_hidden_dim)"
SEG_BIG_CH="$(read_align_value seg selected_big_decoder_hidden_channels)"
CLS_LORA_RANK="$(read_align_value cls selected_lora_rank)"
RET_LORA_RANK="$(read_align_value ret selected_lora_rank)"
SEG_LORA_RANK="$(read_align_value seg selected_lora_rank)"

log "Alignment | cls(big=${CLS_BIG_HIDDEN}, rank=${CLS_LORA_RANK}) | ret(big=${RET_BIG_HIDDEN}, rank=${RET_LORA_RANK}) | seg(big=${SEG_BIG_CH}, rank=${SEG_LORA_RANK})"

# reset manifest for this run plan
cat > "$MANIFEST" <<CSV
kind,task,case,method,backbone_type,unfreeze_blocks,lora_rank,train_dir,eval_dir
CSV

for unfreeze_blocks in "${UNFREEZE_BLOCKS[@]}"; do
  run_tag="u$(printf '%02d' "$unfreeze_blocks")"

  if task_enabled cls; then
    # CLS tradeoff
    cls_a_train="${TRAIN_ROOT}/tradeoff/cls/caseA/${run_tag}"
    cls_a_eval="${EVAL_ROOT}/tradeoff/cls/caseA/${run_tag}"
    append_manifest tradeoff cls A case_a_general_bighead general "$unfreeze_blocks" 0 "$cls_a_train" "$cls_a_eval"
    run_with_eta "tradeoff cls caseA ${run_tag}" run_cls_one "$cls_a_train" "$cls_a_eval" "$GENERAL_WEIGHTS" "$unfreeze_blocks" big "$CLS_BIG_HIDDEN" 0 0

    if [[ "$SMOKE_TEST" != "1" ]]; then
      cls_b_train="${TRAIN_ROOT}/tradeoff/cls/caseB/${run_tag}"
      cls_b_eval="${EVAL_ROOT}/tradeoff/cls/caseB/${run_tag}"
      append_manifest tradeoff cls B case_b_domain_bighead domain "$unfreeze_blocks" 0 "$cls_b_train" "$cls_b_eval"
      run_with_eta "tradeoff cls caseB ${run_tag}" run_cls_one "$cls_b_train" "$cls_b_eval" "$DOMAIN_WEIGHTS" "$unfreeze_blocks" big "$CLS_BIG_HIDDEN" 0 0

      cls_c_train="${TRAIN_ROOT}/tradeoff/cls/caseC/${run_tag}"
      cls_c_eval="${EVAL_ROOT}/tradeoff/cls/caseC/${run_tag}"
      append_manifest tradeoff cls C case_c_general_lora_small general "$unfreeze_blocks" "$CLS_LORA_RANK" "$cls_c_train" "$cls_c_eval"
      run_with_eta "tradeoff cls caseC ${run_tag}" run_cls_one "$cls_c_train" "$cls_c_eval" "$GENERAL_WEIGHTS" "$unfreeze_blocks" small 0 1 "$CLS_LORA_RANK"
    fi
  fi

  if task_enabled ret; then
    # RET tradeoff
    ret_a_train="${TRAIN_ROOT}/tradeoff/ret/caseA/${run_tag}"
    ret_a_eval="${EVAL_ROOT}/tradeoff/ret/caseA/${run_tag}"
    append_manifest tradeoff ret A case_a_general_bighead general "$unfreeze_blocks" 0 "$ret_a_train" "$ret_a_eval"
    run_with_eta "tradeoff ret caseA ${run_tag}" run_ret_one "$ret_a_train" "$ret_a_eval" "$GENERAL_WEIGHTS" "$unfreeze_blocks" big "$RET_BIG_HIDDEN" 0 0

    if [[ "$SMOKE_TEST" != "1" ]]; then
      ret_b_train="${TRAIN_ROOT}/tradeoff/ret/caseB/${run_tag}"
      ret_b_eval="${EVAL_ROOT}/tradeoff/ret/caseB/${run_tag}"
      append_manifest tradeoff ret B case_b_domain_bighead domain "$unfreeze_blocks" 0 "$ret_b_train" "$ret_b_eval"
      run_with_eta "tradeoff ret caseB ${run_tag}" run_ret_one "$ret_b_train" "$ret_b_eval" "$DOMAIN_WEIGHTS" "$unfreeze_blocks" big "$RET_BIG_HIDDEN" 0 0

      ret_c_train="${TRAIN_ROOT}/tradeoff/ret/caseC/${run_tag}"
      ret_c_eval="${EVAL_ROOT}/tradeoff/ret/caseC/${run_tag}"
      append_manifest tradeoff ret C case_c_general_lora_small general "$unfreeze_blocks" "$RET_LORA_RANK" "$ret_c_train" "$ret_c_eval"
      run_with_eta "tradeoff ret caseC ${run_tag}" run_ret_one "$ret_c_train" "$ret_c_eval" "$GENERAL_WEIGHTS" "$unfreeze_blocks" small 0 1 "$RET_LORA_RANK"
    fi
  fi

  if task_enabled seg; then
    # SEG tradeoff
    seg_a_train="${TRAIN_ROOT}/tradeoff/seg/caseA/${run_tag}"
    seg_a_eval="${EVAL_ROOT}/tradeoff/seg/caseA/${run_tag}"
    append_manifest tradeoff seg A case_a_general_bighead general "$unfreeze_blocks" 0 "$seg_a_train" "$seg_a_eval"
    run_with_eta "tradeoff seg caseA ${run_tag}" run_seg_one "$seg_a_train" "$seg_a_eval" "$GENERAL_WEIGHTS" "$unfreeze_blocks" big "$SEG_BIG_CH" 0 0

    if [[ "$SMOKE_TEST" != "1" ]]; then
      seg_b_train="${TRAIN_ROOT}/tradeoff/seg/caseB/${run_tag}"
      seg_b_eval="${EVAL_ROOT}/tradeoff/seg/caseB/${run_tag}"
      append_manifest tradeoff seg B case_b_domain_bighead domain "$unfreeze_blocks" 0 "$seg_b_train" "$seg_b_eval"
      run_with_eta "tradeoff seg caseB ${run_tag}" run_seg_one "$seg_b_train" "$seg_b_eval" "$DOMAIN_WEIGHTS" "$unfreeze_blocks" big "$SEG_BIG_CH" 0 0

      seg_c_train="${TRAIN_ROOT}/tradeoff/seg/caseC/${run_tag}"
      seg_c_eval="${EVAL_ROOT}/tradeoff/seg/caseC/${run_tag}"
      append_manifest tradeoff seg C case_c_general_lora_small general "$unfreeze_blocks" "$SEG_LORA_RANK" "$seg_c_train" "$seg_c_eval"
      run_with_eta "tradeoff seg caseC ${run_tag}" run_seg_one "$seg_c_train" "$seg_c_eval" "$GENERAL_WEIGHTS" "$unfreeze_blocks" small 256 1 "$SEG_LORA_RANK"
    fi
  fi
done

# N=0 head ablation rows and extra small-head runs
if [[ "$SMOKE_TEST" != "1" && "$RUN_ABLATION" == "1" ]]; then
  run_tag="u00"

  if task_enabled cls; then
    # CLS ablation: run small-head baseline + reuse caseA/caseC N=0
    cls_small_train="${TRAIN_ROOT}/ablation/cls/general_small_head/${run_tag}"
    cls_small_eval="${EVAL_ROOT}/ablation/cls/general_small_head/${run_tag}"
    append_manifest ablation cls A general_small_head general 0 0 "$cls_small_train" "$cls_small_eval"
    run_with_eta "ablation cls general_small_head ${run_tag}" run_cls_one "$cls_small_train" "$cls_small_eval" "$GENERAL_WEIGHTS" 0 small 0 0 0
    append_manifest ablation cls A general_big_head general 0 0 "${TRAIN_ROOT}/tradeoff/cls/caseA/${run_tag}" "${EVAL_ROOT}/tradeoff/cls/caseA/${run_tag}"
    append_manifest ablation cls C general_lora_small_head general 0 "$CLS_LORA_RANK" "${TRAIN_ROOT}/tradeoff/cls/caseC/${run_tag}" "${EVAL_ROOT}/tradeoff/cls/caseC/${run_tag}"
  fi

  if task_enabled ret; then
    # RET ablation
    ret_small_train="${TRAIN_ROOT}/ablation/ret/general_small_head/${run_tag}"
    ret_small_eval="${EVAL_ROOT}/ablation/ret/general_small_head/${run_tag}"
    append_manifest ablation ret A general_small_head general 0 0 "$ret_small_train" "$ret_small_eval"
    run_with_eta "ablation ret general_small_head ${run_tag}" run_ret_one "$ret_small_train" "$ret_small_eval" "$GENERAL_WEIGHTS" 0 small 0 0 0
    append_manifest ablation ret A general_big_head general 0 0 "${TRAIN_ROOT}/tradeoff/ret/caseA/${run_tag}" "${EVAL_ROOT}/tradeoff/ret/caseA/${run_tag}"
    append_manifest ablation ret C general_lora_small_head general 0 "$RET_LORA_RANK" "${TRAIN_ROOT}/tradeoff/ret/caseC/${run_tag}" "${EVAL_ROOT}/tradeoff/ret/caseC/${run_tag}"
  fi

  if task_enabled seg; then
    # SEG ablation
    seg_small_train="${TRAIN_ROOT}/ablation/seg/general_small_head/${run_tag}"
    seg_small_eval="${EVAL_ROOT}/ablation/seg/general_small_head/${run_tag}"
    append_manifest ablation seg A general_small_head general 0 0 "$seg_small_train" "$seg_small_eval"
    run_with_eta "ablation seg general_small_head ${run_tag}" run_seg_one "$seg_small_train" "$seg_small_eval" "$GENERAL_WEIGHTS" 0 small 256 0 0
    append_manifest ablation seg A general_big_head general 0 0 "${TRAIN_ROOT}/tradeoff/seg/caseA/${run_tag}" "${EVAL_ROOT}/tradeoff/seg/caseA/${run_tag}"
    append_manifest ablation seg C general_lora_small_head general 0 "$SEG_LORA_RANK" "${TRAIN_ROOT}/tradeoff/seg/caseC/${run_tag}" "${EVAL_ROOT}/tradeoff/seg/caseC/${run_tag}"
  fi
else
  log "Skipping head ablation runs (SMOKE_TEST=${SMOKE_TEST}, RUN_ABLATION=${RUN_ABLATION})"
fi

log "Generating summary figures and analysis report"
python scripts/exp/summarize_lora_tradeoff.py \
  --manifest "$MANIFEST" \
  --summary-dir "$SUMMARY_ROOT" \
  --report-dir "$REPORT_ROOT"

log "===== LoRA Trade-off Batch Done ====="
log "Summary dir: ${SUMMARY_ROOT}"
log "Report dir: ${REPORT_ROOT}"
