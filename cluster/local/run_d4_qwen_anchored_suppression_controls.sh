#!/bin/bash
# Sweep anchored Qwen-Instruct suppression settings and matched-rank controls.

set -euo pipefail

WORKDIR="${WORKDIR:-$(pwd)}"
ARTROOT="${ARTROOT:-$WORKDIR}"
PYTHON="${PYTHON:-python}"
ANCHORED_BT_ROOT="${ANCHORED_BT_ROOT:-$ARTROOT/data/derived/d4_anchored_style_bt_pairs_v1}"
RUN_TAG="${RUN_TAG:-qwen25_anchored_suppression_controls_v1}"
OUT_ROOT="${OUT_ROOT:-$ARTROOT/artifacts/mechanistic/d4_lm_judge_decision_patching_${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-$OUT_ROOT/logs}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
RUN_LABEL="${RUN_LABEL:-qwen25_instruct}"
PROMPT_STYLE="${PROMPT_STYLE:-chat_template}"
GPU="${GPU:-7}"
SELECTED_LAYERS="${SELECTED_LAYERS:-20,24,27,28}"
MAX_FIT_COUNTERFACTUALS="${MAX_FIT_COUNTERFACTUALS:-0}"
MAX_EVAL_COUNTERFACTUALS="${MAX_EVAL_COUNTERFACTUALS:-300}"
MAX_LENGTH="${MAX_LENGTH:-3072}"
PATCH_BATCH_SIZE="${PATCH_BATCH_SIZE:-1}"
PATCH_FIT_FRAC="${PATCH_FIT_FRAC:-0.5}"
CONTROL_REPEATS="${CONTROL_REPEATS:-5}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

cd "$WORKDIR"
export PYTHONPATH="$WORKDIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$ARTROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$LOG_DIR"

WORKDIR="$WORKDIR" ARTROOT="$ARTROOT" PYTHON="$PYTHON" OUT_ROOT="$ANCHORED_BT_ROOT" \
  bash cluster/local/prepare_d4_anchored_style_bt_pairs.sh

fit_path="$ANCHORED_BT_ROOT/generated/bt_pairs.jsonl"
if [[ ! -s "$fit_path" ]]; then
  echo "Missing anchored generated fit probe: $fit_path" >&2
  exit 2
fi

eval_args=()
for dataset in generated atomic composite original_hllm laurito_hllm preference_retention; do
  path="$ANCHORED_BT_ROOT/$dataset/bt_pairs.jsonl"
  if [[ -s "$path" ]]; then
    eval_args+=(--eval "$dataset=$path")
  fi
done

run_setting() {
  local setting="$1"
  local rank="$2"
  local alpha="$3"
  local basis_control="$4"
  local basis_seed="$5"
  local out_dir="$OUT_ROOT/$setting"
  if [[ "$SKIP_EXISTING" == "1" && -s "$out_dir/manifest.json" ]]; then
    echo "skip existing $setting"
    return
  fi
  echo "run setting=$setting rank=$rank alpha=$alpha control=$basis_control seed=$basis_seed"
  mkdir -p "$out_dir"
  (
    export CUDA_VISIBLE_DEVICES="$GPU"
    "$PYTHON" -m aisafety.scripts.run_d4_lm_judge_decision_patching \
      --workspace-root "$WORKDIR" \
      --fit "generated=$fit_path" \
      "${eval_args[@]}" \
      --run-label "$RUN_LABEL" \
      --model-id "$MODEL_ID" \
      --cache-dir "$HF_HOME" \
      --prompt-style "$PROMPT_STYLE" \
      --comparison-template standard \
      --max-fit-counterfactuals "$MAX_FIT_COUNTERFACTUALS" \
      --max-eval-counterfactuals "$MAX_EVAL_COUNTERFACTUALS" \
      --selected-layers "$SELECTED_LAYERS" \
      --batch-size "$PATCH_BATCH_SIZE" \
      --max-length "$MAX_LENGTH" \
      --fit-frac "$PATCH_FIT_FRAC" \
      --subspace-rank "$rank" \
      --suppression-alpha "$alpha" \
      --basis-control "$basis_control" \
      --basis-control-seed "$basis_seed" \
      --skip-residual-patches \
      --skip-component-scout \
      --out-dir "$out_dir"
  ) >"$LOG_DIR/${setting}.out" 2>"$LOG_DIR/${setting}.err"
}

echo "Running anchored Qwen-Instruct suppression controls"
echo "  out_root=$OUT_ROOT"
echo "  gpu=$GPU"
echo "  layers=$SELECTED_LAYERS"

for rank in 1 2 3 5 8; do
  run_setting "fitted_rank${rank}_alpha1" "$rank" "1.0" fitted 1234
done
for alpha in 0.25 0.5 0.75; do
  label="${alpha/./p}"
  run_setting "fitted_rank3_alpha${label}" 3 "$alpha" fitted 1234
done
for basis_control in random shuffled_pair; do
  for repeat in $(seq 1 "$CONTROL_REPEATS"); do
    seed=$((2100 + repeat))
    run_setting "${basis_control}_rank3_alpha1_seed${seed}" 3 "1.0" "$basis_control" "$seed"
  done
done

"$PYTHON" -m aisafety.scripts.read_d4_lm_judge_suppression_controls --input "$OUT_ROOT" --hidden-layer 27

echo "out_root=$OUT_ROOT"
echo "logs=$LOG_DIR"
