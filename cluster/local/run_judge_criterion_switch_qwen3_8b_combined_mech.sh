#!/bin/bash
# Capture and decode the combined criterion-switch scout and extension.

set -euo pipefail

WORKDIR="${WORKDIR:-$(pwd)}"
ARTROOT="${ARTROOT:-$WORKDIR}"
PYTHON="${PYTHON:-$WORKDIR/.venv/bin/python}"
RUN_TAG="${RUN_TAG:-judge_criterion_switch_qwen3_8b_combined_v1}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
RUN_LABEL="${RUN_LABEL:-qwen3_8b_it}"
GPU_0="${GPU_0:-1}"
GPU_1="${GPU_1:-7}"
SOURCE_TAGS="${SOURCE_TAGS:-judge_criterion_switch_qwen3_8b_scout_v1,judge_criterion_switch_qwen3_8b_extension_v1}"
INCLUDE_CONDITIONS="${INCLUDE_CONDITIONS:-reminder,switch,placebo}"
SELECTED_LAYERS="${SELECTED_LAYERS:-4,8,12,16,20,24,28,32}"
OUT_ROOT="${OUT_ROOT:-$ARTROOT/artifacts/mechanistic/$RUN_TAG}"
DECODER_DIR="${DECODER_DIR:-$OUT_ROOT/criterion_decoders}"
LOG_DIR="${LOG_DIR:-$OUT_ROOT/logs}"
USE_4BIT="${USE_4BIT:-0}"
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

IFS=',' read -r -a source_tags <<<"$SOURCE_TAGS"
if [[ "${#source_tags[@]}" -lt 1 ]]; then
  echo "SOURCE_TAGS must contain at least one run tag." >&2
  exit 1
fi

run_activation() {
  local source_tag="$1"
  local shard_index="$2"
  local gpu="$3"
  local behavior_dir="$ARTROOT/artifacts/mechanistic/$source_tag/behavior_shard_$shard_index"
  local output_dir="$OUT_ROOT/activation_${source_tag}_shard_$shard_index"
  local log_name="activation_${source_tag}_shard_$shard_index"
  if [[ ! -s "$behavior_dir/manifest.json" ]]; then
    echo "Missing behavior shard: $behavior_dir" >&2
    return 1
  fi
  if [[ "$SKIP_EXISTING" == "1" && -s "$output_dir/manifest.json" ]]; then
    echo "skip existing activation -> $output_dir"
    return 0
  fi
  local args=(
    "$PYTHON" -m aisafety.scripts.run_judge_criterion_switch_activations
    --workspace-root "$WORKDIR"
    --behavior-dir "$behavior_dir"
    --model-id "$MODEL_ID"
    --run-label "$RUN_LABEL"
    --cache-dir "$HF_HOME"
    --include-conditions "$INCLUDE_CONDITIONS"
    --selected-layers "$SELECTED_LAYERS"
    --shard-size 32
    --compress-shards
    --out-dir "$output_dir"
  )
  if [[ "$USE_4BIT" == "1" ]]; then args+=(--use-4bit); fi
  if [[ -s "$output_dir/traces.jsonl" ]]; then args+=(--resume); fi
  echo "launch source=$source_tag shard=$shard_index gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "${args[@]}" \
    >"$LOG_DIR/$log_name.out" 2>"$LOG_DIR/$log_name.err"
}

trace_args=()
for source_tag in "${source_tags[@]}"; do
  run_activation "$source_tag" 0 "$GPU_0" &
  pid_0=$!
  run_activation "$source_tag" 1 "$GPU_1" &
  pid_1=$!
  status=0
  wait "$pid_0" || status=1
  wait "$pid_1" || status=1
  if [[ "$status" != "0" ]]; then
    echo "Activation capture failed for $source_tag. Check $LOG_DIR." >&2
    exit 1
  fi
  trace_args+=(--trace-dir "$OUT_ROOT/activation_${source_tag}_shard_0")
  trace_args+=(--trace-dir "$OUT_ROOT/activation_${source_tag}_shard_1")
done

if [[ "$SKIP_EXISTING" != "1" || ! -s "$DECODER_DIR/manifest.json" ]]; then
  echo "run combined multiclass decoders"
  "$PYTHON" -m aisafety.scripts.analyze_judge_criterion_switch_decoders \
    --workspace-root "$WORKDIR" \
    "${trace_args[@]}" \
    --targets active_criterion,criterion_target,final_choice,presentation_order \
    --seed "$SEED" \
    --out-dir "$DECODER_DIR" \
    >"$LOG_DIR/criterion_decoders.out" \
    2>"$LOG_DIR/criterion_decoders.err"
fi

echo "COMPLETE"
echo "out_root=$OUT_ROOT"
echo "decoder_dir=$DECODER_DIR"
