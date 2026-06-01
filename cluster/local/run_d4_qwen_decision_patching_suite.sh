#!/bin/bash
# Run Qwen base/instruct decision-state patching and low-rank suppression locally.

set -euo pipefail

WORKDIR="${WORKDIR:-$(pwd)}"
ARTROOT="${ARTROOT:-$WORKDIR}"
PYTHON="${PYTHON:-python}"
RUN_TAG="${RUN_TAG:-qwen25_decision_patching_v1}"
STYLE_CAUSALITY_ROOT="${STYLE_CAUSALITY_ROOT:-$ARTROOT/artifacts/mechanistic/d4_style_causality_qwen25_style_causality_v1}"
OUT_ROOT="${OUT_ROOT:-$ARTROOT/artifacts/mechanistic/d4_lm_judge_decision_patching_${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-$OUT_ROOT/logs}"

BASE_LABEL="${BASE_LABEL:-qwen25_base}"
BASE_MODEL_ID="${BASE_MODEL_ID:-Qwen/Qwen2.5-7B}"
BASE_PROMPT_STYLE="${BASE_PROMPT_STYLE:-plain}"
INSTRUCT_LABEL="${INSTRUCT_LABEL:-qwen25_instruct}"
INSTRUCT_MODEL_ID="${INSTRUCT_MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
INSTRUCT_PROMPT_STYLE="${INSTRUCT_PROMPT_STYLE:-chat_template}"
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"

FIT_DATASET="${FIT_DATASET:-generated}"
MAX_FIT_COUNTERFACTUALS="${MAX_FIT_COUNTERFACTUALS:-0}"
MAX_EVAL_COUNTERFACTUALS="${MAX_EVAL_COUNTERFACTUALS:-300}"
SELECTED_LAYERS="${SELECTED_LAYERS:-}"
LAYER_STRIDE="${LAYER_STRIDE:-4}"
TAIL_LAYERS="${TAIL_LAYERS:-2}"
PATCH_BATCH_SIZE="${PATCH_BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-3072}"
FIT_FRAC="${FIT_FRAC:-0.5}"
SUBSPACE_RANK="${SUBSPACE_RANK:-3}"
SUPPRESSION_ALPHA="${SUPPRESSION_ALPHA:-1.0}"
COMPONENT_MAX_COUNTERFACTUALS="${COMPONENT_MAX_COUNTERFACTUALS:-32}"
COMPONENT_VERIFY_TOP_K="${COMPONENT_VERIFY_TOP_K:-12}"
SKIP_COMPONENT_SCOUT="${SKIP_COMPONENT_SCOUT:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

cd "$WORKDIR"
export PYTHONPATH="$WORKDIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$ARTROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$LOG_DIR"

declare -A BT_FILES
BT_FILES[generated]="$STYLE_CAUSALITY_ROOT/generated/bt_pairs/bt_pairs.jsonl"
BT_FILES[atomic]="$STYLE_CAUSALITY_ROOT/atomic/bt_pairs/bt_pairs.jsonl"
BT_FILES[composite]="$STYLE_CAUSALITY_ROOT/composite/bt_pairs/bt_pairs.jsonl"

if [[ ! -s "${BT_FILES[$FIT_DATASET]:-}" ]]; then
  echo "Missing fit probe BT file: dataset=$FIT_DATASET path=${BT_FILES[$FIT_DATASET]:-unset}" >&2
  echo "Run cluster/local/run_d4_style_causality_base_instruct_suite.sh first." >&2
  exit 2
fi

eval_args=()
for dataset in generated atomic composite; do
  if [[ -s "${BT_FILES[$dataset]}" ]]; then
    eval_args+=(--eval "$dataset=${BT_FILES[$dataset]}")
  fi
done

run_model() {
  local label="$1"
  local model_id="$2"
  local prompt_style="$3"
  local gpu="$4"
  local out_dir="$OUT_ROOT/$label"
  if [[ "$SKIP_EXISTING" == "1" && -s "$out_dir/manifest.json" ]]; then
    echo "skip existing $label -> $out_dir"
    return
  fi
  mkdir -p "$out_dir"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    args=(
      "$PYTHON" -m aisafety.scripts.run_d4_lm_judge_decision_patching
      --workspace-root "$WORKDIR"
      --fit "$FIT_DATASET=${BT_FILES[$FIT_DATASET]}"
      "${eval_args[@]}"
      --run-label "$label"
      --model-id "$model_id"
      --cache-dir "$HF_HOME"
      --prompt-style "$prompt_style"
      --comparison-template standard
      --max-fit-counterfactuals "$MAX_FIT_COUNTERFACTUALS"
      --max-eval-counterfactuals "$MAX_EVAL_COUNTERFACTUALS"
      --layer-stride "$LAYER_STRIDE"
      --tail-layers "$TAIL_LAYERS"
      --batch-size "$PATCH_BATCH_SIZE"
      --max-length "$MAX_LENGTH"
      --fit-frac "$FIT_FRAC"
      --subspace-rank "$SUBSPACE_RANK"
      --suppression-alpha "$SUPPRESSION_ALPHA"
      --component-max-counterfactuals "$COMPONENT_MAX_COUNTERFACTUALS"
      --component-verify-top-k "$COMPONENT_VERIFY_TOP_K"
      --out-dir "$out_dir"
    )
    if [[ -n "$SELECTED_LAYERS" ]]; then
      args+=(--selected-layers "$SELECTED_LAYERS")
    fi
    if [[ "$SKIP_COMPONENT_SCOUT" == "1" ]]; then
      args+=(--skip-component-scout)
    fi
    "${args[@]}"
  ) >"$LOG_DIR/${label}.out" 2>"$LOG_DIR/${label}.err"
}

echo "Running Qwen decision-state patching suite"
echo "  fit_dataset=$FIT_DATASET"
echo "  style_causality_root=$STYLE_CAUSALITY_ROOT"
echo "  out_root=$OUT_ROOT"
echo "  gpus=$GPU_A,$GPU_B"
echo "  max_length=$MAX_LENGTH"
echo "  component_scout=$((1 - SKIP_COMPONENT_SCOUT))"

run_model "$BASE_LABEL" "$BASE_MODEL_ID" "$BASE_PROMPT_STYLE" "$GPU_A" &
pid_base="$!"
run_model "$INSTRUCT_LABEL" "$INSTRUCT_MODEL_ID" "$INSTRUCT_PROMPT_STYLE" "$GPU_B" &
pid_instruct="$!"

failed=0
if ! wait "$pid_base"; then
  failed=1
fi
if ! wait "$pid_instruct"; then
  failed=1
fi
if [[ "$failed" != "0" ]]; then
  echo "At least one patching stage failed. Check $LOG_DIR" >&2
  exit 1
fi

"$PYTHON" -m aisafety.scripts.read_d4_lm_judge_decision_patching --input "$OUT_ROOT"

echo "out_root=$OUT_ROOT"
echo "logs=$LOG_DIR"
