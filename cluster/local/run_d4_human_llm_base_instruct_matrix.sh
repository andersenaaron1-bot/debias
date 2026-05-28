#!/bin/bash
# Run a generic base-vs-instruct human-vs-LLM stage contrast locally.
#
# This is intended for model-family replication runs such as Qwen base/IT.
# It keeps the Tulu stage ladder separate and only compares one base model
# against one post-trained/instruct model, plus optional likelihood controls.

set -euo pipefail

WORKDIR="${WORKDIR:-$(pwd)}"
ARTROOT="${ARTROOT:-$WORKDIR}"
PYTHON="${PYTHON:-python}"
RUN_TAG="${RUN_TAG:-base_instruct_stage_local_v1}"
PAIR_JSONL="${PAIR_JSONL:-$ARTROOT/data/derived/d4_human_llm_alignment_pairs_strat10k_v3/pairs.jsonl}"
PAIR_OUT_DIR="${PAIR_OUT_DIR:-$ARTROOT/data/derived/d4_human_llm_stage_contrast_pairs_${RUN_TAG}}"
BT_PAIRS_JSONL="${BT_PAIRS_JSONL:-$PAIR_OUT_DIR/bt_pairs.jsonl}"
OUT_ROOT="${OUT_ROOT:-$ARTROOT/artifacts/mechanistic/d4_hllm_stage_${RUN_TAG}}"
SUMMARY_OUT_DIR="${SUMMARY_OUT_DIR:-$ARTROOT/artifacts/mechanistic/d4_human_llm_stage_contrast_summary_${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-$OUT_ROOT/logs}"

BASE_LABEL="${BASE_LABEL:-qwen25_base}"
BASE_MODEL_ID="${BASE_MODEL_ID:-Qwen/Qwen2.5-7B}"
BASE_PROMPT_STYLE="${BASE_PROMPT_STYLE:-plain}"
INSTRUCT_LABEL="${INSTRUCT_LABEL:-qwen25_instruct}"
INSTRUCT_MODEL_ID="${INSTRUCT_MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
INSTRUCT_PROMPT_STYLE="${INSTRUCT_PROMPT_STYLE:-chat_template}"

GPU_A="${GPU_A:-1}"
GPU_B="${GPU_B:-7}"

MAX_SOURCE_PAIRS="${MAX_SOURCE_PAIRS:-1000}"
SCORE_MAX_PAIRS="${SCORE_MAX_PAIRS:-0}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-2}"
LIKELIHOOD_BATCH_SIZE="${LIKELIHOOD_BATCH_SIZE:-2}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
USE_4BIT="${USE_4BIT:-0}"
INCLUDE_RESPONSE_LIKELIHOOD="${INCLUDE_RESPONSE_LIKELIHOOD:-0}"
INCLUDE_FORCED_CHOICE="${INCLUDE_FORCED_CHOICE:-1}"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"
COMPARISON_TEMPLATE="${COMPARISON_TEMPLATE:-standard}"
LABELS="${LABELS:-A,B}"

mkdir -p "$LOG_DIR"
cd "$WORKDIR"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$WORKDIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$ARTROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

if [[ ! -f "$PAIR_JSONL" ]]; then
  echo "Missing PAIR_JSONL: $PAIR_JSONL" >&2
  echo "Build or copy d4_human_llm_alignment_pairs first, then rerun." >&2
  exit 2
fi

echo "Running local base/instruct human-vs-LLM stage matrix"
echo "  run_tag=$RUN_TAG"
echo "  pair_jsonl=$PAIR_JSONL"
echo "  max_source_pairs=$MAX_SOURCE_PAIRS"
echo "  base=$BASE_LABEL :: $BASE_MODEL_ID :: $BASE_PROMPT_STYLE"
echo "  instruct=$INSTRUCT_LABEL :: $INSTRUCT_MODEL_ID :: $INSTRUCT_PROMPT_STYLE"
echo "  gpus=$GPU_A,$GPU_B"
echo "  out_root=$OUT_ROOT"
echo "  include_forced_choice=$INCLUDE_FORCED_CHOICE"
echo "  include_response_likelihood=$INCLUDE_RESPONSE_LIKELIHOOD"
echo "  comparison_template=$COMPARISON_TEMPLATE"
echo "  labels=$LABELS"

if [[ "$INCLUDE_FORCED_CHOICE" != "1" && "$INCLUDE_RESPONSE_LIKELIHOOD" != "1" && "$SUMMARY_ONLY" != "1" ]]; then
  echo "Nothing to run: set INCLUDE_FORCED_CHOICE=1, INCLUDE_RESPONSE_LIKELIHOOD=1, or SUMMARY_ONLY=1." >&2
  exit 2
fi

if [[ "$SUMMARY_ONLY" != "1" ]]; then
  "$PYTHON" -m aisafety.scripts.build_d4_human_llm_stage_contrast_pairs \
    --workspace-root "$WORKDIR" \
    --pair-jsonl "$PAIR_JSONL" \
    --out-dir "$PAIR_OUT_DIR" \
    --max-pairs "$MAX_SOURCE_PAIRS" \
    --include-order-swaps
elif [[ ! -f "$BT_PAIRS_JSONL" ]]; then
  echo "SUMMARY_ONLY=1 but missing BT_PAIRS_JSONL: $BT_PAIRS_JSONL" >&2
  exit 2
fi

run_stage() {
  local label="$1"
  local model_id="$2"
  local scoring_mode="$3"
  local prompt_style="$4"
  local out_dir="$5"
  local batch_size="$6"
  local gpu_id="$7"

  mkdir -p "$out_dir"
  echo "launch label=$label gpu=$gpu_id model=$model_id mode=$scoring_mode prompt_style=$prompt_style"
  (
    export CUDA_VISIBLE_DEVICES="$gpu_id"
    cmd=(
      "$PYTHON" -m aisafety.scripts.run_d4_human_llm_stage_contrast
      --workspace-root "$WORKDIR"
      --bt-pairs-jsonl "$BT_PAIRS_JSONL"
      --scoring-mode "$scoring_mode"
      --stage-label "$label"
      --model-id "$model_id"
      --cache-dir "$HF_HOME"
      --prompt-style "$prompt_style"
      --comparison-template "$COMPARISON_TEMPLATE"
      --labels "$LABELS"
      --max-pairs "$SCORE_MAX_PAIRS"
      --score-batch-size "$batch_size"
      --max-length "$MAX_LENGTH"
      --out-dir "$out_dir"
    )
    if [[ "$USE_4BIT" == "1" ]]; then
      cmd+=(--use-4bit)
    fi
    "${cmd[@]}"
  ) >"$LOG_DIR/${label}.out" 2>"$LOG_DIR/${label}.err" &
}

wait_wave() {
  local failed=0
  local pid
  for pid in "$@"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" != "0" ]]; then
    echo "At least one stage failed. Check logs in $LOG_DIR" >&2
    exit 1
  fi
}

run_entries=()

if [[ "$INCLUDE_FORCED_CHOICE" == "1" && "$SUMMARY_ONLY" != "1" ]]; then
  run_stage "$BASE_LABEL" "$BASE_MODEL_ID" "forced_choice" "$BASE_PROMPT_STYLE" \
    "$OUT_ROOT/${BASE_LABEL}_forced" "$SCORE_BATCH_SIZE" "$GPU_A"
  pid_base="$!"
  run_stage "$INSTRUCT_LABEL" "$INSTRUCT_MODEL_ID" "forced_choice" "$INSTRUCT_PROMPT_STYLE" \
    "$OUT_ROOT/${INSTRUCT_LABEL}_forced" "$SCORE_BATCH_SIZE" "$GPU_B"
  pid_instruct="$!"
  wait_wave "$pid_base" "$pid_instruct"
fi

if [[ "$INCLUDE_FORCED_CHOICE" == "1" ]]; then
  run_entries+=(
    "$BASE_LABEL=$OUT_ROOT/${BASE_LABEL}_forced"
    "$INSTRUCT_LABEL=$OUT_ROOT/${INSTRUCT_LABEL}_forced"
  )
fi

if [[ "$INCLUDE_RESPONSE_LIKELIHOOD" == "1" && "$SUMMARY_ONLY" != "1" ]]; then
  run_stage "${BASE_LABEL}_like" "$BASE_MODEL_ID" "response_likelihood" "$BASE_PROMPT_STYLE" \
    "$OUT_ROOT/${BASE_LABEL}_response_likelihood" "$LIKELIHOOD_BATCH_SIZE" "$GPU_A"
  pid_base_like="$!"
  run_stage "${INSTRUCT_LABEL}_like" "$INSTRUCT_MODEL_ID" "response_likelihood" "$INSTRUCT_PROMPT_STYLE" \
    "$OUT_ROOT/${INSTRUCT_LABEL}_response_likelihood" "$LIKELIHOOD_BATCH_SIZE" "$GPU_B"
  pid_instruct_like="$!"
  wait_wave "$pid_base_like" "$pid_instruct_like"
fi

if [[ "$INCLUDE_RESPONSE_LIKELIHOOD" == "1" ]]; then
  run_entries+=(
    "${BASE_LABEL}_like=$OUT_ROOT/${BASE_LABEL}_response_likelihood"
    "${INSTRUCT_LABEL}_like=$OUT_ROOT/${INSTRUCT_LABEL}_response_likelihood"
  )
fi

if [[ "${#run_entries[@]}" -eq 0 ]]; then
  echo "No run entries available for summary. Check INCLUDE_* flags." >&2
  exit 2
fi

summary_args=(
  "$PYTHON" -m aisafety.scripts.summarize_d4_human_llm_stage_contrasts
  --workspace-root "$WORKDIR"
)
for entry in "${run_entries[@]}"; do
  summary_args+=(--run "$entry")
done
if [[ "$INCLUDE_FORCED_CHOICE" == "1" ]]; then
  summary_args+=(--contrast "${INSTRUCT_LABEL}_minus_${BASE_LABEL}=${INSTRUCT_LABEL}-${BASE_LABEL}")
fi
if [[ "$INCLUDE_RESPONSE_LIKELIHOOD" == "1" ]]; then
  summary_args+=(--contrast "${INSTRUCT_LABEL}_like_minus_${BASE_LABEL}_like=${INSTRUCT_LABEL}_like-${BASE_LABEL}_like")
fi
summary_args+=(--out-dir "$SUMMARY_OUT_DIR")

"${summary_args[@]}"

echo "summary_out_dir=$SUMMARY_OUT_DIR"
echo "logs=$LOG_DIR"
