#!/bin/bash
# Replay completed factual budget traces at the exact forced-decision boundary.

set -euo pipefail

WORKDIR="${WORKDIR:-$(pwd)}"
ARTROOT="${ARTROOT:-$WORKDIR}"
PYTHON="${PYTHON:-$WORKDIR/.venv/bin/python}"
RUN_TAG="${RUN_TAG:-judge_factual_readout_qwen3_8b_v1}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
RUN_LABEL="${RUN_LABEL:-qwen3_8b_it}"
GPU_0="${GPU_0:-1}"
GPU_1="${GPU_1:-7}"
BUDGET_SOURCE_TAG="${BUDGET_SOURCE_TAG:-judge_deliberation_qwen3_8b_budget_scout_v1}"
BUDGET_RUN_DIR="${BUDGET_RUN_DIR:-$ARTROOT/artifacts/mechanistic/$BUDGET_SOURCE_TAG/budget_sweep}"
DATASETS_0="${DATASETS_0:-arc_challenge,bbh_logical_deduction,gsm8k_verification}"
DATASETS_1="${DATASETS_1:-math500_verification,truthfulqa}"
BUDGET_TOKENS="${BUDGET_TOKENS:-0,128,512,2048}"
SELECTED_LAYERS="${SELECTED_LAYERS:-4,8,12,16,20,24,28,32}"
MAX_SCORE_LENGTH="${MAX_SCORE_LENGTH:-8192}"
OUT_ROOT="${OUT_ROOT:-$ARTROOT/artifacts/mechanistic/$RUN_TAG}"
ACTIVATION_0="${ACTIVATION_0:-$OUT_ROOT/activation_shard_0}"
ACTIVATION_1="${ACTIVATION_1:-$OUT_ROOT/activation_shard_1}"
ANALYSIS_DIR="${ANALYSIS_DIR:-$OUT_ROOT/pair_analysis}"
LOG_DIR="${LOG_DIR:-$OUT_ROOT/logs}"
USE_4BIT="${USE_4BIT:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
BOOTSTRAP="${BOOTSTRAP:-1000}"
SEED="${SEED:-1234}"

cd "$WORKDIR"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$WORKDIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$ARTROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$LOG_DIR"

if [[ ! -s "$BUDGET_RUN_DIR/manifest.json" ]]; then
  echo "Missing completed budget run: $BUDGET_RUN_DIR" >&2
  exit 1
fi

run_capture() {
  local gpu="$1"
  local datasets="$2"
  local output_dir="$3"
  local log_name="$4"
  if [[ "$SKIP_EXISTING" == "1" && -s "$output_dir/manifest.json" ]]; then
    echo "skip existing factual activation -> $output_dir"
    return 0
  fi
  local args=(
    "$PYTHON" -m aisafety.scripts.run_judge_factual_readout_activations
    --workspace-root "$WORKDIR"
    --budget-run-dir "$BUDGET_RUN_DIR"
    --model-id "$MODEL_ID"
    --run-label "$RUN_LABEL"
    --cache-dir "$HF_HOME"
    --include-datasets "$datasets"
    --budget-tokens "$BUDGET_TOKENS"
    --selected-layers "$SELECTED_LAYERS"
    --max-score-length "$MAX_SCORE_LENGTH"
    --shard-size 32
    --compress-shards
    --out-dir "$output_dir"
  )
  if [[ "$USE_4BIT" == "1" ]]; then args+=(--use-4bit); fi
  if [[ -s "$output_dir/traces.jsonl" ]]; then args+=(--resume); fi
  echo "launch factual readout gpu=$gpu datasets=$datasets"
  CUDA_VISIBLE_DEVICES="$gpu" "${args[@]}" \
    >"$LOG_DIR/$log_name.out" 2>"$LOG_DIR/$log_name.err"
}

run_capture "$GPU_0" "$DATASETS_0" "$ACTIVATION_0" activation_0 &
pid_0=$!
run_capture "$GPU_1" "$DATASETS_1" "$ACTIVATION_1" activation_1 &
pid_1=$!
status=0
wait "$pid_0" || status=1
wait "$pid_1" || status=1
if [[ "$status" != "0" ]]; then
  echo "Factual activation capture failed. Check $LOG_DIR." >&2
  exit 1
fi

if [[ "$SKIP_EXISTING" != "1" || ! -s "$ANALYSIS_DIR/manifest.json" ]]; then
  echo "run factual pair-grouped temporal analysis"
  "$PYTHON" -m aisafety.scripts.analyze_judge_criterion_switch_pairs \
    --workspace-root "$WORKDIR" \
    --trace-dir "$ACTIVATION_0" \
    --trace-dir "$ACTIVATION_1" \
    --targets active_criterion,criterion_target,current_choice,final_choice,presentation_order \
    --bootstrap "$BOOTSTRAP" \
    --seed "$SEED" \
    --out-dir "$ANALYSIS_DIR" \
    >"$LOG_DIR/pair_analysis.out" \
    2>"$LOG_DIR/pair_analysis.err"
fi

echo "COMPLETE"
echo "out_root=$OUT_ROOT"
echo "analysis_dir=$ANALYSIS_DIR"
