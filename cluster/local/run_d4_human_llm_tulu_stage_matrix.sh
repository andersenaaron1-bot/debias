#!/bin/bash
# Run the Tulu/Llama human-vs-LLM training-stage contrast on a local GPU host.
#
# This is a non-Slurm fallback for outages or local GPU clusters. It pins one
# model-scoring process per selected GPU with CUDA_VISIBLE_DEVICES, waits for
# each wave, then runs the CPU summary step.

set -euo pipefail

WORKDIR="${WORKDIR:-$(pwd)}"
ARTROOT="${ARTROOT:-$WORKDIR}"
PYTHON="${PYTHON:-python}"
RUN_TAG="${RUN_TAG:-tulu_stage_scout_local_v1}"
PAIR_JSONL="${PAIR_JSONL:-$ARTROOT/data/derived/d4_human_llm_alignment_pairs_strat10k_v3/pairs.jsonl}"
PAIR_OUT_DIR="${PAIR_OUT_DIR:-$ARTROOT/data/derived/d4_human_llm_stage_contrast_pairs_${RUN_TAG}}"
BT_PAIRS_JSONL="${BT_PAIRS_JSONL:-$PAIR_OUT_DIR/bt_pairs.jsonl}"
OUT_ROOT="${OUT_ROOT:-$ARTROOT/artifacts/mechanistic/d4_hllm_stage_${RUN_TAG}}"
SUMMARY_OUT_DIR="${SUMMARY_OUT_DIR:-$ARTROOT/artifacts/mechanistic/d4_human_llm_stage_contrast_summary_${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-$OUT_ROOT/logs}"

# Defaults target the visible A100s from the provided nvidia-smi:
# GPU 1 = A100 80GB, GPU 7 = A100 80GB, GPU 0 = A100 40GB.
# Avoid V100/RTX8000 by default because the scorer loads CausalLMs in bf16.
GPU_A="${GPU_A:-1}"
GPU_B="${GPU_B:-7}"
GPU_C="${GPU_C:-0}"

# Keep the default as a scout. Set MAX_SOURCE_PAIRS=0 for the full broad file.
MAX_SOURCE_PAIRS="${MAX_SOURCE_PAIRS:-1000}"
SCORE_MAX_PAIRS="${SCORE_MAX_PAIRS:-0}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-2}"
LIKELIHOOD_BATCH_SIZE="${LIKELIHOOD_BATCH_SIZE:-2}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
USE_4BIT="${USE_4BIT:-0}"
INCLUDE_META_INSTRUCT="${INCLUDE_META_INSTRUCT:-1}"
INCLUDE_RESPONSE_LIKELIHOOD="${INCLUDE_RESPONSE_LIKELIHOOD:-0}"
INCLUDE_FORCED_CHOICE="${INCLUDE_FORCED_CHOICE:-1}"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"
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

if [[ ! -f "$PAIR_JSONL" ]]; then
  echo "Missing PAIR_JSONL: $PAIR_JSONL" >&2
  echo "Build or copy d4_human_llm_alignment_pairs first, then rerun." >&2
  exit 2
fi

echo "Running local Tulu/Llama stage matrix"
echo "  run_tag=$RUN_TAG"
echo "  workdir=$WORKDIR"
echo "  artroot=$ARTROOT"
echo "  pair_jsonl=$PAIR_JSONL"
echo "  max_source_pairs=$MAX_SOURCE_PAIRS"
echo "  gpus=$GPU_A,$GPU_B,$GPU_C"
echo "  out_root=$OUT_ROOT"
echo "  include_forced_choice=$INCLUDE_FORCED_CHOICE"
echo "  include_response_likelihood=$INCLUDE_RESPONSE_LIKELIHOOD"

if [[ "$INCLUDE_FORCED_CHOICE" != "1" && "$INCLUDE_RESPONSE_LIKELIHOOD" != "1" && "$SUMMARY_ONLY" != "1" ]]; then
  echo "Nothing to run: set INCLUDE_FORCED_CHOICE=1, INCLUDE_RESPONSE_LIKELIHOOD=1, or SUMMARY_ONLY=1." >&2
  exit 2
fi

"$PYTHON" -m aisafety.scripts.build_d4_human_llm_stage_contrast_pairs \
  --workspace-root "$WORKDIR" \
  --pair-jsonl "$PAIR_JSONL" \
  --out-dir "$PAIR_OUT_DIR" \
  --max-pairs "$MAX_SOURCE_PAIRS" \
  --include-order-swaps

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
  run_stage "llama31_base" "meta-llama/Llama-3.1-8B" "forced_choice" "plain" \
    "$OUT_ROOT/llama31_base_forced_plain" "$SCORE_BATCH_SIZE" "$GPU_A"
  pid_base="$!"
  run_stage "tulu3_sft" "allenai/Llama-3.1-Tulu-3-8B-SFT" "forced_choice" "$TULU_PROMPT_STYLE" \
    "$OUT_ROOT/tulu3_sft_forced_chat" "$SCORE_BATCH_SIZE" "$GPU_B"
  pid_sft="$!"
  run_stage "tulu3_dpo" "allenai/Llama-3.1-Tulu-3-8B-DPO" "forced_choice" "$TULU_PROMPT_STYLE" \
    "$OUT_ROOT/tulu3_dpo_forced_chat" "$SCORE_BATCH_SIZE" "$GPU_C"
  pid_dpo="$!"
  wait_wave "$pid_base" "$pid_sft" "$pid_dpo"

  run_stage "tulu3_final" "allenai/Llama-3.1-Tulu-3-8B" "forced_choice" "$TULU_PROMPT_STYLE" \
    "$OUT_ROOT/tulu3_final_forced_chat" "$SCORE_BATCH_SIZE" "$GPU_A"
  pid_final="$!"
  wave2_pids=("$pid_final")
  if [[ "$INCLUDE_META_INSTRUCT" == "1" ]]; then
    run_stage "llama31_instruct" "meta-llama/Llama-3.1-8B-Instruct" "forced_choice" "$META_INSTRUCT_PROMPT_STYLE" \
      "$OUT_ROOT/llama31_instruct_forced_chat" "$SCORE_BATCH_SIZE" "$GPU_B"
    wave2_pids+=("$!")
  fi
  wait_wave "${wave2_pids[@]}"
fi

if [[ "$INCLUDE_FORCED_CHOICE" == "1" ]]; then
  run_entries+=(
    "llama31_base=$OUT_ROOT/llama31_base_forced_plain"
    "tulu3_sft=$OUT_ROOT/tulu3_sft_forced_chat"
    "tulu3_dpo=$OUT_ROOT/tulu3_dpo_forced_chat"
    "tulu3_final=$OUT_ROOT/tulu3_final_forced_chat"
  )
  if [[ "$INCLUDE_META_INSTRUCT" == "1" ]]; then
    run_entries+=("llama31_instruct=$OUT_ROOT/llama31_instruct_forced_chat")
  fi
fi

if [[ "$INCLUDE_RESPONSE_LIKELIHOOD" == "1" && "$SUMMARY_ONLY" != "1" ]]; then
  run_stage "llama31_base_like" "meta-llama/Llama-3.1-8B" "response_likelihood" "plain" \
    "$OUT_ROOT/llama31_base_response_likelihood" "$LIKELIHOOD_BATCH_SIZE" "$GPU_A"
  pid_base_like="$!"
  run_stage "tulu3_sft_like" "allenai/Llama-3.1-Tulu-3-8B-SFT" "response_likelihood" "$TULU_PROMPT_STYLE" \
    "$OUT_ROOT/tulu3_sft_response_likelihood" "$LIKELIHOOD_BATCH_SIZE" "$GPU_B"
  pid_sft_like="$!"
  run_stage "tulu3_dpo_like" "allenai/Llama-3.1-Tulu-3-8B-DPO" "response_likelihood" "$TULU_PROMPT_STYLE" \
    "$OUT_ROOT/tulu3_dpo_response_likelihood" "$LIKELIHOOD_BATCH_SIZE" "$GPU_C"
  pid_dpo_like="$!"
  wait_wave "$pid_base_like" "$pid_sft_like" "$pid_dpo_like"
  run_entries+=(
    "llama31_base_like=$OUT_ROOT/llama31_base_response_likelihood"
    "tulu3_sft_like=$OUT_ROOT/tulu3_sft_response_likelihood"
    "tulu3_dpo_like=$OUT_ROOT/tulu3_dpo_response_likelihood"
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
  summary_args+=(
    --contrast "tulu3_sft_minus_base=tulu3_sft-llama31_base"
    --contrast "tulu3_dpo_minus_sft=tulu3_dpo-tulu3_sft"
    --contrast "tulu3_final_minus_dpo=tulu3_final-tulu3_dpo"
    --contrast "tulu3_final_minus_base=tulu3_final-llama31_base"
  )
  if [[ "$INCLUDE_META_INSTRUCT" == "1" ]]; then
    summary_args+=(
      --contrast "llama31_instruct_minus_base=llama31_instruct-llama31_base"
      --contrast "llama31_instruct_minus_tulu3_dpo=llama31_instruct-tulu3_dpo"
    )
  fi
fi
if [[ "$INCLUDE_RESPONSE_LIKELIHOOD" == "1" ]]; then
  summary_args+=(
    --contrast "tulu3_sft_like_minus_base_like=tulu3_sft_like-llama31_base_like"
    --contrast "tulu3_dpo_like_minus_sft_like=tulu3_dpo_like-tulu3_sft_like"
  )
fi
summary_args+=(--out-dir "$SUMMARY_OUT_DIR")

"${summary_args[@]}"

echo "summary_out_dir=$SUMMARY_OUT_DIR"
echo "logs=$LOG_DIR"
