#!/bin/bash
# Capture and causally patch the locked criterion confirmation on ipe-monster.

set -euo pipefail

WORKDIR="${WORKDIR:-$(pwd)}"
ARTROOT="${ARTROOT:-$WORKDIR}"
PYTHON="${PYTHON:-$WORKDIR/.venv/bin/python}"
SOURCE_TAG="${SOURCE_TAG:-judge_criterion_confirmation_qwen3_8b_v1}"
RUN_TAG="${RUN_TAG:-judge_criterion_confirmation_qwen3_8b_mech_v1}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
RUN_LABEL="${RUN_LABEL:-qwen3_8b_it}"
GPU="${GPU:-7}"

BEHAVIOR_DIR="${BEHAVIOR_DIR:-$ARTROOT/artifacts/mechanistic/$SOURCE_TAG/behavior}"
OUT_ROOT="${OUT_ROOT:-$ARTROOT/artifacts/mechanistic/$RUN_TAG}"
ACTIVATION_DIR="${ACTIVATION_DIR:-$OUT_ROOT/activations}"
ANALYSIS_DIR="${ANALYSIS_DIR:-$OUT_ROOT/activation_analysis}"
PATCH_DIR="${PATCH_DIR:-$OUT_ROOT/patching}"
LOG_DIR="${LOG_DIR:-$OUT_ROOT/logs}"

SELECTED_LAYERS="${SELECTED_LAYERS:-4,8,12,16,20,24,28,32}"
TARGET_LAYERS="${TARGET_LAYERS:-active_criterion:20,criterion_target:32,current_choice:28,final_choice:32,presentation_order:12}"
ALPHAS="${ALPHAS:-1.0}"
MAX_SCORE_LENGTH="${MAX_SCORE_LENGTH:-8192}"
BOOTSTRAP="${BOOTSTRAP:-2000}"
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

if [[ ! -s "$BEHAVIOR_DIR/manifest.json" \
      || ! -s "$BEHAVIOR_DIR/switch_traces.jsonl" ]]; then
  echo "Missing completed confirmation behavior: $BEHAVIOR_DIR" >&2
  exit 1
fi

echo "Running Qwen3-8B criterion confirmation mechanisms"
echo "  source_tag=$SOURCE_TAG"
echo "  run_tag=$RUN_TAG"
echo "  model=$RUN_LABEL :: $MODEL_ID"
echo "  gpu=$GPU"
echo "  activation_traces=408"
echo "  patch_rows_upper_bound=648"

if [[ "$SKIP_EXISTING" != "1" || ! -s "$ACTIVATION_DIR/manifest.json" ]]; then
  activation_args=(
    "$PYTHON" -m aisafety.scripts.run_judge_criterion_switch_activations
    --workspace-root "$WORKDIR"
    --behavior-dir "$BEHAVIOR_DIR"
    --model-id "$MODEL_ID"
    --run-label "$RUN_LABEL"
    --cache-dir "$HF_HOME"
    --selected-layers "$SELECTED_LAYERS"
    --point-mode readout
    --max-score-length "$MAX_SCORE_LENGTH"
    --shard-size 32
    --compress-shards
    --out-dir "$ACTIVATION_DIR"
  )
  if [[ "$USE_4BIT" == "1" ]]; then activation_args+=(--use-4bit); fi
  if [[ -s "$ACTIVATION_DIR/traces.jsonl" ]]; then
    activation_args+=(--resume)
  fi
  echo "run activation capture gpu=$GPU"
  CUDA_VISIBLE_DEVICES="$GPU" "${activation_args[@]}" \
    >"$LOG_DIR/activations.out" 2>"$LOG_DIR/activations.err"
else
  echo "skip existing activation capture -> $ACTIVATION_DIR"
fi

if [[ "$SKIP_EXISTING" != "1" || ! -s "$ANALYSIS_DIR/manifest.json" ]]; then
  echo "run fixed-layer pair-grouped activation analysis"
  "$PYTHON" \
    -m aisafety.scripts.analyze_judge_criterion_confirmation_activations \
    --workspace-root "$WORKDIR" \
    --trace-dir "$ACTIVATION_DIR" \
    --target-layers "$TARGET_LAYERS" \
    --c-value 1.0 \
    --cv-folds 5 \
    --bootstrap "$BOOTSTRAP" \
    --seed "$SEED" \
    --out-dir "$ANALYSIS_DIR" \
    >"$LOG_DIR/activation_analysis.out" \
    2>"$LOG_DIR/activation_analysis.err"
else
  echo "skip existing activation analysis -> $ANALYSIS_DIR"
fi

if [[ "$SKIP_EXISTING" != "1" || ! -s "$PATCH_DIR/manifest.json" ]]; then
  patch_args=(
    "$PYTHON" -m aisafety.scripts.run_judge_criterion_confirmation_patching
    --workspace-root "$WORKDIR"
    --behavior-dir "$BEHAVIOR_DIR"
    --trace-dir "$ACTIVATION_DIR"
    --model-id "$MODEL_ID"
    --cache-dir "$HF_HOME"
    --labels A,B,C
    --alphas "$ALPHAS"
    --include-orders original,swapped
    --branch-index 0
    --max-score-length "$MAX_SCORE_LENGTH"
    --bootstrap "$BOOTSTRAP"
    --seed "$SEED"
    --out-dir "$PATCH_DIR"
  )
  if [[ "$USE_4BIT" == "1" ]]; then patch_args+=(--use-4bit); fi
  if [[ -s "$PATCH_DIR/patch_rows.jsonl" ]]; then patch_args+=(--resume); fi
  echo "run matched activation patching gpu=$GPU"
  CUDA_VISIBLE_DEVICES="$GPU" "${patch_args[@]}" \
    >"$LOG_DIR/patching.out" 2>"$LOG_DIR/patching.err"
else
  echo "skip existing patching -> $PATCH_DIR"
fi

echo "run compact mechanistic readout"
"$PYTHON" \
  -m aisafety.scripts.read_judge_criterion_confirmation_mechanistic \
  --workspace-root "$WORKDIR" \
  --analysis-dir "$ANALYSIS_DIR" \
  --patch-dir "$PATCH_DIR" \
  >"$OUT_ROOT/readout.txt"
cat "$OUT_ROOT/readout.txt"

echo "COMPLETE"
echo "out_root=$OUT_ROOT"
echo "activation_dir=$ACTIVATION_DIR"
echo "analysis_dir=$ANALYSIS_DIR"
echo "patch_dir=$PATCH_DIR"
