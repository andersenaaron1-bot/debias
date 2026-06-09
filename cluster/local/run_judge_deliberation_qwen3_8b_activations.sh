#!/bin/bash
# Run the frozen-suite Qwen3-8B activation analysis on ipe-monster.

set -euo pipefail

WORKDIR="${WORKDIR:-$(pwd)}"
ARTROOT="${ARTROOT:-$WORKDIR}"
PYTHON="${PYTHON:-$WORKDIR/.venv/bin/python}"
RUN_TAG="${RUN_TAG:-judge_deliberation_qwen3_8b_activations_v1}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
RUN_LABEL="${RUN_LABEL:-qwen3_8b_it}"
GPU="${GPU:-7}"

SUITE_DIR="${SUITE_DIR:?Set SUITE_DIR to the frozen deliberation comparison suite}"
OUT_ROOT="${OUT_ROOT:-$ARTROOT/artifacts/mechanistic/$RUN_TAG}"
TRACE_DIR="${TRACE_DIR:-$OUT_ROOT/trajectories_qwen3_8b_it}"
DECODER_DIR="${DECODER_DIR:-$OUT_ROOT/fixed_decoders}"
LOG_DIR="${LOG_DIR:-$OUT_ROOT/logs}"

SELECTED_LAYERS="${SELECTED_LAYERS:-4,8,12,16,20,24,28,32}"
PROBE_TARGETS="${PROBE_TARGETS:-final_choice,target_option,presentation_order}"
BRANCHES_PER_COMPARISON="${BRANCHES_PER_COMPARISON:-3}"
MAX_PAIRS="${MAX_PAIRS:-0}"
TRAJECTORY_POINTS="${TRAJECTORY_POINTS:-33}"
SHARD_SIZE="${SHARD_SIZE:-32}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_NEW_TOKENS_THINKING="${MAX_NEW_TOKENS_THINKING:-2048}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
USE_4BIT="${USE_4BIT:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
ANALYZE_ONLY="${ANALYZE_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"
SEED="${SEED:-1234}"

cd "$WORKDIR"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$WORKDIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$ARTROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$LOG_DIR"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python executable is missing or not executable: $PYTHON" >&2
  exit 1
fi
if [[ ! -s "$SUITE_DIR/comparisons.jsonl" ]]; then
  echo "Missing frozen comparison suite: $SUITE_DIR/comparisons.jsonl" >&2
  exit 1
fi

echo "Running local Qwen3-8B deliberation activation pass"
echo "  run_tag=$RUN_TAG"
echo "  model=$RUN_LABEL :: $MODEL_ID :: gpu $GPU"
echo "  suite_dir=$SUITE_DIR"
echo "  trace_dir=$TRACE_DIR"
echo "  decoder_dir=$DECODER_DIR"
echo "  selected_layers=$SELECTED_LAYERS"
echo "  trajectory_points=$TRAJECTORY_POINTS"
echo "  max_thinking_tokens=$MAX_NEW_TOKENS_THINKING"
echo "  logs=$LOG_DIR"

run_trace() {
  if [[ "$SKIP_EXISTING" == "1" && -s "$TRACE_DIR/manifest.json" ]]; then
    echo "skip existing activation trace -> $TRACE_DIR"
    return 0
  fi
  local args=(
    "$PYTHON" -m aisafety.scripts.run_judge_reasoning_trajectories
    --workspace-root "$WORKDIR"
    --comparisons-jsonl "$SUITE_DIR/comparisons.jsonl"
    --run-label "$RUN_LABEL"
    --model-id "$MODEL_ID"
    --cache-dir "$HF_HOME"
    --prompt-style chat_template
    --reasoning-modes thinking
    --branches-per-comparison "$BRANCHES_PER_COMPARISON"
    --max-pairs "$MAX_PAIRS"
    --cap-strategy source_round_robin
    --selected-layers "$SELECTED_LAYERS"
    --trajectory-points "$TRAJECTORY_POINTS"
    --shard-size "$SHARD_SIZE"
    --compress-shards
    --max-prompt-length "$MAX_PROMPT_LENGTH"
    --max-new-tokens-thinking "$MAX_NEW_TOKENS_THINKING"
    --temperature "$TEMPERATURE"
    --top-p "$TOP_P"
    --top-k "$TOP_K"
    --seed "$SEED"
    --out-dir "$TRACE_DIR"
  )
  if [[ "$USE_4BIT" == "1" ]]; then
    args+=(--use-4bit)
  fi
  if [[ -s "$TRACE_DIR/traces.jsonl" ]]; then
    args+=(--resume)
  fi

  echo "launch activation trace gpu=$GPU model=$MODEL_ID"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '  CUDA_VISIBLE_DEVICES=%q ' "$GPU"
    printf '%q ' "${args[@]}"
    printf '\n'
    return 0
  fi
  CUDA_VISIBLE_DEVICES="$GPU" "${args[@]}" \
    >"$LOG_DIR/activation_trace.out" 2>"$LOG_DIR/activation_trace.err"
}

run_fixed_decoders() {
  if [[ "$SKIP_EXISTING" == "1" && -s "$DECODER_DIR/manifest.json" ]]; then
    echo "skip existing fixed-decoder analysis -> $DECODER_DIR"
    return 0
  fi
  if [[ "$DRY_RUN" != "1" && ! -s "$TRACE_DIR/manifest.json" ]]; then
    echo "Missing completed activation trace: $TRACE_DIR/manifest.json" >&2
    exit 1
  fi
  echo "run fixed decoders"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  PYTHONWARNINGS="ignore::FutureWarning" "$PYTHON" \
    -m aisafety.scripts.analyze_judge_reasoning_fixed_decoders \
    --workspace-root "$WORKDIR" \
    --trace-dir "$TRACE_DIR" \
    --reasoning-mode thinking \
    --anchor-target final_choice \
    --probe-targets "$PROBE_TARGETS" \
    --seed "$SEED" \
    --out-dir "$DECODER_DIR" \
    >"$LOG_DIR/fixed_decoders.out" 2>"$LOG_DIR/fixed_decoders.err"
}

if [[ "$ANALYZE_ONLY" != "1" ]]; then
  run_trace
fi
run_fixed_decoders

echo "COMPLETE"
echo "out_root=$OUT_ROOT"
echo "trace_dir=$TRACE_DIR"
echo "decoder_dir=$DECODER_DIR"
echo "logs=$LOG_DIR"
