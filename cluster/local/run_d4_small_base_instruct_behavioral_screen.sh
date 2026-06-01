#!/bin/bash
# Run a small matched base-vs-instruct behavioral screen on ipe-monster GPUs.

set -euo pipefail

WORKDIR="${WORKDIR:-$(pwd)}"
ARTROOT="${ARTROOT:-$WORKDIR}"
RUN_TAG="${RUN_TAG:-small_base_instruct_behavioral_screen_v1}"
PAIR_JSONL="${PAIR_JSONL:-$ARTROOT/data/derived/d4_human_llm_alignment_pairs_matched_lenlex_relaxed_v1/pairs.jsonl}"
GENERATED_COUNTERFACTUAL_JSONL="${GENERATED_COUNTERFACTUAL_JSONL:-$ARTROOT/data/derived/d4_assistant_style_generated_counterfactual_pairs_v1/counterfactuals.jsonl}"
BASE_LABEL="${BASE_LABEL:-olmo3_7b_base}"
BASE_MODEL_ID="${BASE_MODEL_ID:-allenai/Olmo-3-7B}"
BASE_PROMPT_STYLE="${BASE_PROMPT_STYLE:-plain}"
INSTRUCT_LABEL="${INSTRUCT_LABEL:-olmo3_7b_instruct}"
INSTRUCT_MODEL_ID="${INSTRUCT_MODEL_ID:-allenai/Olmo-3-7B-Instruct}"
INSTRUCT_PROMPT_STYLE="${INSTRUCT_PROMPT_STYLE:-chat_template}"
GPU_A="${GPU_A:-1}"
GPU_B="${GPU_B:-7}"
MAX_SOURCE_PAIRS="${MAX_SOURCE_PAIRS:-100}"
MAX_REWRITE_COUNTERFACTUALS="${MAX_REWRITE_COUNTERFACTUALS:-100}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
USE_4BIT="${USE_4BIT:-0}"

cd "$WORKDIR"
export PYTHONPATH="$WORKDIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$ARTROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

common_env=(
  WORKDIR="$WORKDIR"
  ARTROOT="$ARTROOT"
  BASE_LABEL="$BASE_LABEL"
  BASE_MODEL_ID="$BASE_MODEL_ID"
  BASE_PROMPT_STYLE="$BASE_PROMPT_STYLE"
  INSTRUCT_LABEL="$INSTRUCT_LABEL"
  INSTRUCT_MODEL_ID="$INSTRUCT_MODEL_ID"
  INSTRUCT_PROMPT_STYLE="$INSTRUCT_PROMPT_STYLE"
  GPU_A="$GPU_A"
  GPU_B="$GPU_B"
  SCORE_BATCH_SIZE="$SCORE_BATCH_SIZE"
  MAX_LENGTH="$MAX_LENGTH"
  USE_4BIT="$USE_4BIT"
  INCLUDE_RESPONSE_LIKELIHOOD=0
)

echo "Running small matched base/IT behavioral screen"
echo "  run_tag=$RUN_TAG"
echo "  base=$BASE_LABEL :: $BASE_MODEL_ID"
echo "  instruct=$INSTRUCT_LABEL :: $INSTRUCT_MODEL_ID"
echo "  gpus=$GPU_A,$GPU_B"

env "${common_env[@]}" \
  RUN_TAG="${RUN_TAG}_hllm" \
  PAIR_JSONL="$PAIR_JSONL" \
  MAX_SOURCE_PAIRS="$MAX_SOURCE_PAIRS" \
  bash cluster/local/run_d4_human_llm_base_instruct_matrix.sh

if [[ ! -s "$GENERATED_COUNTERFACTUAL_JSONL" ]]; then
  echo "Missing generated rewrite counterfactuals: $GENERATED_COUNTERFACTUAL_JSONL" >&2
  exit 2
fi

env "${common_env[@]}" \
  RUN_TAG="${RUN_TAG}_generated" \
  COUNTERFACTUAL_JSONL="$GENERATED_COUNTERFACTUAL_JSONL" \
  AXES=generated_assistant_style \
  MAX_COUNTERFACTUALS="$MAX_REWRITE_COUNTERFACTUALS" \
  bash cluster/local/run_d4_surface_bt_base_instruct_matrix.sh

echo "hllm_summary=$ARTROOT/artifacts/mechanistic/d4_human_llm_stage_contrast_summary_${RUN_TAG}_hllm"
echo "generated_summary=$ARTROOT/artifacts/mechanistic/d4_bt_surface_stage_summary_${RUN_TAG}_generated"
