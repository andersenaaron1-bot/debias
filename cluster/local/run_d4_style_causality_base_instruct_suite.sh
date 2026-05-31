#!/bin/bash
# Run binary-judge and layerwise activation style-causality diagnostics locally.

set -euo pipefail

WORKDIR="${WORKDIR:-$(pwd)}"
ARTROOT="${ARTROOT:-$WORKDIR}"
PYTHON="${PYTHON:-python}"
RUN_TAG="${RUN_TAG:-qwen25_style_causality_v1}"
PAIR_JSONL="${PAIR_JSONL:-$ARTROOT/data/derived/d4_human_llm_alignment_pairs_matched_lenlex_relaxed_v1/pairs.jsonl}"
COMPOSITE_COUNTERFACTUAL_JSONL="${COMPOSITE_COUNTERFACTUAL_JSONL:-$ARTROOT/data/derived/d4_surface_counterfactual_pairs_matched_lenlex_relaxed_v1/counterfactuals.jsonl}"
ATOMIC_OUT_DIR="${ATOMIC_OUT_DIR:-$ARTROOT/data/derived/d4_assistant_style_atomic_counterfactual_pairs_v1}"
GENERATED_COUNTERFACTUAL_JSONL="${GENERATED_COUNTERFACTUAL_JSONL:-$ARTROOT/data/derived/d4_assistant_style_generated_counterfactual_pairs_v1/counterfactuals.jsonl}"
OUT_ROOT="${OUT_ROOT:-$ARTROOT/artifacts/mechanistic/d4_style_causality_${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-$OUT_ROOT/logs}"

BASE_LABEL="${BASE_LABEL:-qwen25_base}"
BASE_MODEL_ID="${BASE_MODEL_ID:-Qwen/Qwen2.5-7B}"
BASE_PROMPT_STYLE="${BASE_PROMPT_STYLE:-plain}"
INSTRUCT_LABEL="${INSTRUCT_LABEL:-qwen25_instruct}"
INSTRUCT_MODEL_ID="${INSTRUCT_MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
INSTRUCT_PROMPT_STYLE="${INSTRUCT_PROMPT_STYLE:-chat_template}"
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"

ATOMIC_MAX_SOURCE_PAIRS="${ATOMIC_MAX_SOURCE_PAIRS:-300}"
MAX_COMPOSITE_COUNTERFACTUALS="${MAX_COMPOSITE_COUNTERFACTUALS:-1500}"
MAX_ATOMIC_COUNTERFACTUALS="${MAX_ATOMIC_COUNTERFACTUALS:-1500}"
MAX_GENERATED_COUNTERFACTUALS="${MAX_GENERATED_COUNTERFACTUALS:-0}"
ACTIVATION_MAX_COUNTERFACTUALS="${ACTIVATION_MAX_COUNTERFACTUALS:-800}"
ACTIVATION_LAYER_STRIDE="${ACTIVATION_LAYER_STRIDE:-4}"
ACTIVATION_TAIL_LAYERS="${ACTIVATION_TAIL_LAYERS:-2}"
ACTIVATION_SELECTED_LAYERS="${ACTIVATION_SELECTED_LAYERS:-}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-1}"
ACTIVATION_BATCH_SIZE="${ACTIVATION_BATCH_SIZE:-2}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
COMPARISON_TEMPLATE="${COMPARISON_TEMPLATE:-standard}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

cd "$WORKDIR"
export PYTHONPATH="$WORKDIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$ARTROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$LOG_DIR"

if [[ ! -f "$PAIR_JSONL" ]]; then
  echo "Missing PAIR_JSONL: $PAIR_JSONL" >&2
  exit 2
fi

echo "Building deterministic atomic assistant-style counterfactuals"
"$PYTHON" -m aisafety.scripts.build_d4_assistant_style_atomic_counterfactual_pairs \
  --workspace-root "$WORKDIR" \
  --pair-jsonl "$PAIR_JSONL" \
  --out-dir "$ATOMIC_OUT_DIR" \
  --max-pairs "$ATOMIC_MAX_SOURCE_PAIRS"

datasets=()
if [[ -f "$COMPOSITE_COUNTERFACTUAL_JSONL" ]]; then
  datasets+=("composite|$COMPOSITE_COUNTERFACTUAL_JSONL|$MAX_COMPOSITE_COUNTERFACTUALS|structured_assistant_packaging,answer_likeness_packaging,formal_institutional_packaging")
fi
datasets+=("atomic|$ATOMIC_OUT_DIR/counterfactuals.jsonl|$MAX_ATOMIC_COUNTERFACTUALS|")
if [[ -f "$GENERATED_COUNTERFACTUAL_JSONL" ]]; then
  datasets+=("generated|$GENERATED_COUNTERFACTUAL_JSONL|$MAX_GENERATED_COUNTERFACTUALS|")
else
  echo "Generated rewrite file not present; running deterministic families only."
  echo "Prepare it later with cluster/local/prepare_d4_generated_assistant_style_rewrites.sh"
fi

wait_wave() {
  local failed=0
  local pid
  for pid in "$@"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" != "0" ]]; then
    echo "At least one GPU stage failed. Check $LOG_DIR" >&2
    exit 1
  fi
}

score_stage() {
  local dataset="$1"
  local label="$2"
  local model_id="$3"
  local stage="$4"
  local prompt_style="$5"
  local gpu="$6"
  local bt_jsonl="$7"
  local out_dir="$8"
  if [[ "$SKIP_EXISTING" == "1" && -s "$out_dir/bt_stage_scores.csv" ]]; then
    echo "skip existing score $dataset/$label"
    (true) &
    return
  fi
  mkdir -p "$out_dir"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$PYTHON" -m aisafety.scripts.run_d4_bt_stage_contrast \
      --workspace-root "$WORKDIR" \
      --bt-pairs-jsonl "$bt_jsonl" \
      --stage "$stage" \
      --scoring-mode forced_choice \
      --stage-label "$label" \
      --model-id "$model_id" \
      --cache-dir "$HF_HOME" \
      --prompt-style "$prompt_style" \
      --comparison-template "$COMPARISON_TEMPLATE" \
      --score-batch-size "$SCORE_BATCH_SIZE" \
      --max-length "$MAX_LENGTH" \
      --out-dir "$out_dir"
  ) >"$LOG_DIR/${dataset}_${label}_score.out" 2>"$LOG_DIR/${dataset}_${label}_score.err" &
}

activation_stage() {
  local dataset="$1"
  local label="$2"
  local model_id="$3"
  local gpu="$4"
  local counterfactual_jsonl="$5"
  local score_dir="$6"
  local out_dir="$7"
  local axes="$8"
  if [[ "$SKIP_EXISTING" == "1" && -s "$out_dir/layer_summary.csv" ]]; then
    echo "skip existing activation $dataset/$label"
    (true) &
    return
  fi
  mkdir -p "$out_dir"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    args=(
      "$PYTHON" -m aisafety.scripts.run_d4_lm_style_activation_contrast
      --workspace-root "$WORKDIR"
      --counterfactual-jsonl "$counterfactual_jsonl"
      --bt-scores "$score_dir"
      --run-label "$label"
      --model-id "$model_id"
      --cache-dir "$HF_HOME"
      --max-counterfactuals "$ACTIVATION_MAX_COUNTERFACTUALS"
      --layer-stride "$ACTIVATION_LAYER_STRIDE"
      --tail-layers "$ACTIVATION_TAIL_LAYERS"
      --batch-size "$ACTIVATION_BATCH_SIZE"
      --max-length "$MAX_LENGTH"
      --out-dir "$out_dir"
    )
    if [[ -n "$ACTIVATION_SELECTED_LAYERS" ]]; then
      args+=(--selected-layers "$ACTIVATION_SELECTED_LAYERS")
    fi
    if [[ -n "$axes" ]]; then
      args+=(--axes "$axes")
    fi
    "${args[@]}"
  ) >"$LOG_DIR/${dataset}_${label}_activation.out" 2>"$LOG_DIR/${dataset}_${label}_activation.err" &
}

for dataset_spec in "${datasets[@]}"; do
  IFS='|' read -r dataset counterfactual_jsonl max_counterfactuals axes <<< "$dataset_spec"
  echo "Running dataset=$dataset counterfactuals=$counterfactual_jsonl"
  dataset_root="$OUT_ROOT/$dataset"
  pair_dir="$dataset_root/bt_pairs"
  score_base="$dataset_root/scores/$BASE_LABEL"
  score_instruct="$dataset_root/scores/$INSTRUCT_LABEL"
  activation_base="$dataset_root/activations/$BASE_LABEL"
  activation_instruct="$dataset_root/activations/$INSTRUCT_LABEL"
  mkdir -p "$pair_dir"

  pair_args=(
    "$PYTHON" -m aisafety.scripts.build_d4_bt_stage_contrast_pairs
    --workspace-root "$WORKDIR"
    --counterfactual-jsonl "$counterfactual_jsonl"
    --out-dir "$pair_dir"
    --max-counterfactuals "$max_counterfactuals"
    --include-order-swaps
  )
  if [[ -n "$axes" ]]; then
    pair_args+=(--axes "$axes")
  fi
  "${pair_args[@]}"

  score_stage "$dataset" "$BASE_LABEL" "$BASE_MODEL_ID" base_lm "$BASE_PROMPT_STYLE" "$GPU_A" "$pair_dir/bt_pairs.jsonl" "$score_base"
  pid_base="$!"
  score_stage "$dataset" "$INSTRUCT_LABEL" "$INSTRUCT_MODEL_ID" it_lm "$INSTRUCT_PROMPT_STYLE" "$GPU_B" "$pair_dir/bt_pairs.jsonl" "$score_instruct"
  pid_instruct="$!"
  wait_wave "$pid_base" "$pid_instruct"

  "$PYTHON" -m aisafety.scripts.summarize_d4_bt_stage_templates \
    --workspace-root "$WORKDIR" \
    --run "$COMPARISON_TEMPLATE:$BASE_LABEL=$score_base" \
    --run "$COMPARISON_TEMPLATE:$INSTRUCT_LABEL=$score_instruct" \
    --stage-contrast "${INSTRUCT_LABEL}_minus_${BASE_LABEL}=${INSTRUCT_LABEL}-${BASE_LABEL}" \
    --out-dir "$dataset_root/binary_summary"

  activation_stage "$dataset" "$BASE_LABEL" "$BASE_MODEL_ID" "$GPU_A" "$counterfactual_jsonl" "$score_base" "$activation_base" "$axes"
  pid_base="$!"
  activation_stage "$dataset" "$INSTRUCT_LABEL" "$INSTRUCT_MODEL_ID" "$GPU_B" "$counterfactual_jsonl" "$score_instruct" "$activation_instruct" "$axes"
  pid_instruct="$!"
  wait_wave "$pid_base" "$pid_instruct"

  "$PYTHON" -m aisafety.scripts.summarize_d4_lm_style_activation_contrasts \
    --workspace-root "$WORKDIR" \
    --base "$activation_base" \
    --instruct "$activation_instruct" \
    --contrast-label "${INSTRUCT_LABEL}_minus_${BASE_LABEL}" \
    --out-dir "$dataset_root/activation_summary"
done

"$PYTHON" -m aisafety.scripts.read_d4_style_causality_suite --input "$OUT_ROOT"

echo "out_root=$OUT_ROOT"
echo "logs=$LOG_DIR"
