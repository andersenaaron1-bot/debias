#!/bin/bash
# Capture criterion-switch activations, fit decoders, and optionally patch.

set -euo pipefail

WORKDIR="${WORKDIR:-$(pwd)}"
ARTROOT="${ARTROOT:-$WORKDIR}"
PYTHON="${PYTHON:-$WORKDIR/.venv/bin/python}"
RUN_TAG="${RUN_TAG:-judge_criterion_switch_qwen3_8b_scout_v1}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
RUN_LABEL="${RUN_LABEL:-qwen3_8b_it}"
GPU_0="${GPU_0:-0}"
GPU_1="${GPU_1:-1}"
OUT_ROOT="${OUT_ROOT:-$ARTROOT/artifacts/mechanistic/$RUN_TAG}"
BEHAVIOR_0="${BEHAVIOR_0:-$OUT_ROOT/behavior_shard_0}"
BEHAVIOR_1="${BEHAVIOR_1:-$OUT_ROOT/behavior_shard_1}"
ACTIVATION_0="${ACTIVATION_0:-$OUT_ROOT/activation_shard_0}"
ACTIVATION_1="${ACTIVATION_1:-$OUT_ROOT/activation_shard_1}"
DECODER_DIR="${DECODER_DIR:-$OUT_ROOT/criterion_decoders}"
PATCH_DIR="${PATCH_DIR:-$OUT_ROOT/criterion_patching}"
LOG_DIR="${LOG_DIR:-$OUT_ROOT/logs}"

SELECTED_LAYERS="${SELECTED_LAYERS:-4,8,12,16,20,24,28,32}"
SHARD_SIZE="${SHARD_SIZE:-32}"
USE_4BIT="${USE_4BIT:-0}"
RUN_PATCHING="${RUN_PATCHING:-0}"
PATCH_ALPHAS="${PATCH_ALPHAS:-0.5,1.0}"
PATCH_MAX_PAIRS="${PATCH_MAX_PAIRS:-0}"
PATCH_MAX_NEW_TOKENS="${PATCH_MAX_NEW_TOKENS:-384}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SEED="${SEED:-1234}"

cd "$WORKDIR"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$WORKDIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$ARTROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$LOG_DIR"

run_activation() {
  local index="$1"
  local gpu="$2"
  local behavior="$OUT_ROOT/behavior_shard_${index}"
  local output="$OUT_ROOT/activation_shard_${index}"
  if [[ "$SKIP_EXISTING" == "1" && -s "$output/manifest.json" ]]; then
    echo "skip existing activation shard $index -> $output"
    return 0
  fi
  local args=(
    "$PYTHON" -m aisafety.scripts.run_judge_criterion_switch_activations
    --workspace-root "$WORKDIR"
    --behavior-dir "$behavior"
    --model-id "$MODEL_ID"
    --run-label "$RUN_LABEL"
    --cache-dir "$HF_HOME"
    --selected-layers "$SELECTED_LAYERS"
    --shard-size "$SHARD_SIZE"
    --compress-shards
    --out-dir "$output"
  )
  if [[ "$USE_4BIT" == "1" ]]; then args+=(--use-4bit); fi
  if [[ -s "$output/traces.jsonl" ]]; then args+=(--resume); fi
  echo "launch activation shard=$index gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "${args[@]}" \
    >"$LOG_DIR/activation_shard_${index}.out" \
    2>"$LOG_DIR/activation_shard_${index}.err"
}

if [[ ! -s "$BEHAVIOR_0/manifest.json" || ! -s "$BEHAVIOR_1/manifest.json" ]]; then
  echo "Both behavioral shards must complete before activation capture." >&2
  exit 1
fi

run_activation 0 "$GPU_0" &
pid_0=$!
run_activation 1 "$GPU_1" &
pid_1=$!
status=0
wait "$pid_0" || status=1
wait "$pid_1" || status=1
if [[ "$status" != "0" ]]; then
  echo "At least one activation shard failed. Check $LOG_DIR." >&2
  exit 1
fi

if [[ "$SKIP_EXISTING" != "1" || ! -s "$DECODER_DIR/manifest.json" ]]; then
  echo "run multiclass criterion decoders"
  "$PYTHON" -m aisafety.scripts.analyze_judge_criterion_switch_decoders \
    --workspace-root "$WORKDIR" \
    --trace-dir "$ACTIVATION_0" \
    --trace-dir "$ACTIVATION_1" \
    --targets active_criterion,criterion_target,final_choice,presentation_order \
    --seed "$SEED" \
    --out-dir "$DECODER_DIR" \
    >"$LOG_DIR/criterion_decoders.out" \
    2>"$LOG_DIR/criterion_decoders.err"
fi

if [[ "$RUN_PATCHING" == "1" ]]; then
  args=(
    "$PYTHON" -m aisafety.scripts.run_judge_criterion_switch_patching
    --workspace-root "$WORKDIR"
    --behavior-dir "$BEHAVIOR_0"
    --behavior-dir "$BEHAVIOR_1"
    --trace-dir "$ACTIVATION_0"
    --trace-dir "$ACTIVATION_1"
    --decoder-dir "$DECODER_DIR"
    --model-id "$MODEL_ID"
    --cache-dir "$HF_HOME"
    --layer-target active_criterion
    --point-name phase2_prompt_end
    --alphas "$PATCH_ALPHAS"
    --include-negative
    --include-shuffled
    --include-placebo
    --max-pairs "$PATCH_MAX_PAIRS"
    --max-new-tokens "$PATCH_MAX_NEW_TOKENS"
    --seed "$SEED"
    --out-dir "$PATCH_DIR"
  )
  if [[ "$USE_4BIT" == "1" ]]; then args+=(--use-4bit); fi
  echo "run criterion patching gpu=$GPU_1"
  CUDA_VISIBLE_DEVICES="$GPU_1" "${args[@]}" \
    >"$LOG_DIR/criterion_patching.out" \
    2>"$LOG_DIR/criterion_patching.err"
fi

echo "COMPLETE"
echo "decoder_dir=$DECODER_DIR"
echo "patch_dir=$PATCH_DIR"
