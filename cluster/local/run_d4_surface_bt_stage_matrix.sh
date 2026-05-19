#!/bin/bash
# Run surface-cue BT stage contrasts on a local GPU host.

set -euo pipefail

WORKDIR="${WORKDIR:-$(pwd)}"
ARTROOT="${ARTROOT:-$WORKDIR}"
PYTHON="${PYTHON:-python}"
RUN_TAG="${RUN_TAG:-surface_bt_stage_local_v1}"
COUNTERFACTUAL_JSONL="${COUNTERFACTUAL_JSONL:-$ARTROOT/data/derived/d4_surface_counterfactual_pairs_v1/counterfactuals.jsonl}"
BT_PAIR_OUT_DIR="${BT_PAIR_OUT_DIR:-$ARTROOT/data/derived/d4_bt_stage_contrast_pairs_${RUN_TAG}}"
BT_PAIRS_JSONL="${BT_PAIRS_JSONL:-$BT_PAIR_OUT_DIR/bt_pairs.jsonl}"
OUT_ROOT="${OUT_ROOT:-$ARTROOT/artifacts/mechanistic/d4_bt_surface_stage_${RUN_TAG}}"
SUMMARY_OUT_DIR="${SUMMARY_OUT_DIR:-$ARTROOT/artifacts/mechanistic/d4_bt_surface_stage_summary_${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-$OUT_ROOT/logs}"

GPU_A="${GPU_A:-1}"
GPU_B="${GPU_B:-7}"
GPU_C="${GPU_C:-0}"

AXES="${AXES:-structured_assistant_packaging,answer_likeness_packaging,formal_institutional_packaging}"
DIRECTIONS="${DIRECTIONS:-}"
ROLES="${ROLES:-}"
MAX_COUNTERFACTUALS="${MAX_COUNTERFACTUALS:-0}"
INCLUDE_ORDER_SWAPS="${INCLUDE_ORDER_SWAPS:-1}"
SCORE_MAX_PAIRS="${SCORE_MAX_PAIRS:-0}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-2}"
LIKELIHOOD_BATCH_SIZE="${LIKELIHOOD_BATCH_SIZE:-2}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
USE_4BIT="${USE_4BIT:-0}"
INCLUDE_META_INSTRUCT="${INCLUDE_META_INSTRUCT:-1}"
INCLUDE_RESPONSE_LIKELIHOOD="${INCLUDE_RESPONSE_LIKELIHOOD:-0}"
INCLUDE_FORCED_CHOICE="${INCLUDE_FORCED_CHOICE:-1}"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"
COMPARISON_TEMPLATE="${COMPARISON_TEMPLATE:-standard}"
TULU_PROMPT_STYLE="${TULU_PROMPT_STYLE:-chat_template}"
META_INSTRUCT_PROMPT_STYLE="${META_INSTRUCT_PROMPT_STYLE:-chat_template}"

mkdir -p "$LOG_DIR"
cd "$WORKDIR"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$WORKDIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$ARTROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

if [[ ! -f "$COUNTERFACTUAL_JSONL" ]]; then
  echo "Missing COUNTERFACTUAL_JSONL: $COUNTERFACTUAL_JSONL" >&2
  exit 2
fi

echo "Running local surface BT stage matrix"
echo "  run_tag=$RUN_TAG"
echo "  counterfactual_jsonl=$COUNTERFACTUAL_JSONL"
echo "  bt_pairs_jsonl=$BT_PAIRS_JSONL"
echo "  axes=$AXES"
echo "  roles=$ROLES"
echo "  comparison_template=$COMPARISON_TEMPLATE"
echo "  include_forced_choice=$INCLUDE_FORCED_CHOICE"
echo "  include_response_likelihood=$INCLUDE_RESPONSE_LIKELIHOOD"
echo "  gpus=$GPU_A,$GPU_B,$GPU_C"

if [[ "$INCLUDE_FORCED_CHOICE" != "1" && "$INCLUDE_RESPONSE_LIKELIHOOD" != "1" && "$SUMMARY_ONLY" != "1" ]]; then
  echo "Nothing to run: set INCLUDE_FORCED_CHOICE=1, INCLUDE_RESPONSE_LIKELIHOOD=1, or SUMMARY_ONLY=1." >&2
  exit 2
fi

if [[ "$SUMMARY_ONLY" != "1" ]]; then
  pair_cmd=(
    "$PYTHON" -m aisafety.scripts.build_d4_bt_stage_contrast_pairs
    --workspace-root "$WORKDIR"
    --counterfactual-jsonl "$COUNTERFACTUAL_JSONL"
    --out-dir "$BT_PAIR_OUT_DIR"
    --axes "$AXES"
    --directions "$DIRECTIONS"
    --roles "$ROLES"
    --max-counterfactuals "$MAX_COUNTERFACTUALS"
  )
  if [[ "$INCLUDE_ORDER_SWAPS" == "1" ]]; then
    pair_cmd+=(--include-order-swaps)
  else
    pair_cmd+=(--no-include-order-swaps)
  fi
  "${pair_cmd[@]}"
elif [[ ! -f "$BT_PAIRS_JSONL" ]]; then
  echo "SUMMARY_ONLY=1 but missing BT_PAIRS_JSONL: $BT_PAIRS_JSONL" >&2
  exit 2
fi

run_stage() {
  local label="$1"
  local model_id="$2"
  local stage="$3"
  local scoring_mode="$4"
  local prompt_style="$5"
  local out_dir="$6"
  local batch_size="$7"
  local gpu_id="$8"

  mkdir -p "$out_dir"
  echo "launch label=$label gpu=$gpu_id model=$model_id mode=$scoring_mode prompt_style=$prompt_style"
  (
    export CUDA_VISIBLE_DEVICES="$gpu_id"
    cmd=(
      "$PYTHON" -m aisafety.scripts.run_d4_bt_stage_contrast
      --workspace-root "$WORKDIR"
      --bt-pairs-jsonl "$BT_PAIRS_JSONL"
      --stage "$stage"
      --scoring-mode "$scoring_mode"
      --stage-label "$label"
      --model-id "$model_id"
      --cache-dir "$HF_HOME"
      --prompt-style "$prompt_style"
      --comparison-template "$COMPARISON_TEMPLATE"
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
  run_stage "llama31_base" "meta-llama/Llama-3.1-8B" "base_lm" "forced_choice" "plain" \
    "$OUT_ROOT/llama31_base_forced_plain" "$SCORE_BATCH_SIZE" "$GPU_A"
  pid_base="$!"
  run_stage "tulu3_sft" "allenai/Llama-3.1-Tulu-3-8B-SFT" "it_lm" "forced_choice" "$TULU_PROMPT_STYLE" \
    "$OUT_ROOT/tulu3_sft_forced_chat" "$SCORE_BATCH_SIZE" "$GPU_B"
  pid_sft="$!"
  run_stage "tulu3_dpo" "allenai/Llama-3.1-Tulu-3-8B-DPO" "it_lm" "forced_choice" "$TULU_PROMPT_STYLE" \
    "$OUT_ROOT/tulu3_dpo_forced_chat" "$SCORE_BATCH_SIZE" "$GPU_C"
  pid_dpo="$!"
  wait_wave "$pid_base" "$pid_sft" "$pid_dpo"

  run_stage "tulu3_final" "allenai/Llama-3.1-Tulu-3-8B" "it_lm" "forced_choice" "$TULU_PROMPT_STYLE" \
    "$OUT_ROOT/tulu3_final_forced_chat" "$SCORE_BATCH_SIZE" "$GPU_A"
  pid_final="$!"
  wave2_pids=("$pid_final")
  if [[ "$INCLUDE_META_INSTRUCT" == "1" ]]; then
    run_stage "llama31_instruct" "meta-llama/Llama-3.1-8B-Instruct" "it_lm" "forced_choice" "$META_INSTRUCT_PROMPT_STYLE" \
      "$OUT_ROOT/llama31_instruct_forced_chat" "$SCORE_BATCH_SIZE" "$GPU_B"
    wave2_pids+=("$!")
  fi
  wait_wave "${wave2_pids[@]}"
fi

if [[ "$INCLUDE_FORCED_CHOICE" == "1" ]]; then
  run_entries+=(
    "$COMPARISON_TEMPLATE:llama31_base=$OUT_ROOT/llama31_base_forced_plain"
    "$COMPARISON_TEMPLATE:tulu3_sft=$OUT_ROOT/tulu3_sft_forced_chat"
    "$COMPARISON_TEMPLATE:tulu3_dpo=$OUT_ROOT/tulu3_dpo_forced_chat"
    "$COMPARISON_TEMPLATE:tulu3_final=$OUT_ROOT/tulu3_final_forced_chat"
  )
  if [[ "$INCLUDE_META_INSTRUCT" == "1" ]]; then
    run_entries+=("$COMPARISON_TEMPLATE:llama31_instruct=$OUT_ROOT/llama31_instruct_forced_chat")
  fi
fi

if [[ "$INCLUDE_RESPONSE_LIKELIHOOD" == "1" && "$SUMMARY_ONLY" != "1" ]]; then
  run_stage "llama31_base_like" "meta-llama/Llama-3.1-8B" "base_lm" "response_likelihood" "plain" \
    "$OUT_ROOT/llama31_base_response_likelihood" "$LIKELIHOOD_BATCH_SIZE" "$GPU_A"
  pid_base_like="$!"
  run_stage "tulu3_sft_like" "allenai/Llama-3.1-Tulu-3-8B-SFT" "it_lm" "response_likelihood" "$TULU_PROMPT_STYLE" \
    "$OUT_ROOT/tulu3_sft_response_likelihood" "$LIKELIHOOD_BATCH_SIZE" "$GPU_B"
  pid_sft_like="$!"
  run_stage "tulu3_dpo_like" "allenai/Llama-3.1-Tulu-3-8B-DPO" "it_lm" "response_likelihood" "$TULU_PROMPT_STYLE" \
    "$OUT_ROOT/tulu3_dpo_response_likelihood" "$LIKELIHOOD_BATCH_SIZE" "$GPU_C"
  pid_dpo_like="$!"
  wait_wave "$pid_base_like" "$pid_sft_like" "$pid_dpo_like"
  run_entries+=(
    "$COMPARISON_TEMPLATE:llama31_base_like=$OUT_ROOT/llama31_base_response_likelihood"
    "$COMPARISON_TEMPLATE:tulu3_sft_like=$OUT_ROOT/tulu3_sft_response_likelihood"
    "$COMPARISON_TEMPLATE:tulu3_dpo_like=$OUT_ROOT/tulu3_dpo_response_likelihood"
  )
fi

if [[ "${#run_entries[@]}" -eq 0 ]]; then
  echo "No run entries available for summary. Check INCLUDE_* flags." >&2
  exit 2
fi

summary_args=(
  "$PYTHON" -m aisafety.scripts.summarize_d4_bt_stage_templates
  --workspace-root "$WORKDIR"
)
for entry in "${run_entries[@]}"; do
  summary_args+=(--run "$entry")
done
summary_args+=(--out-dir "$SUMMARY_OUT_DIR")

"${summary_args[@]}"

echo "summary_out_dir=$SUMMARY_OUT_DIR"
echo "logs=$LOG_DIR"
