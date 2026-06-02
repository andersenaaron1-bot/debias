#!/bin/bash
# Build a small common panel and score hosted IT/chat judges through OpenRouter.

set -euo pipefail

WORKDIR="${WORKDIR:-$(pwd)}"
ARTROOT="${ARTROOT:-$WORKDIR}"
PYTHON="${PYTHON:-python}"
RUN_TAG="${RUN_TAG:-openrouter_judge_screen_v1}"
PAIR_JSONL="${PAIR_JSONL:-$ARTROOT/data/derived/d4_human_llm_alignment_pairs_matched_lenlex_relaxed_v1/pairs.jsonl}"
GENERATED_COUNTERFACTUAL_JSONL="${GENERATED_COUNTERFACTUAL_JSONL:-$ARTROOT/data/derived/d4_assistant_style_generated_counterfactual_pairs_v1/counterfactuals.jsonl}"
PAIR_ROOT="${PAIR_ROOT:-$ARTROOT/data/derived/d4_openrouter_judge_screen_pairs_${RUN_TAG}}"
OUT_DIR="${OUT_DIR:-$ARTROOT/artifacts/mechanistic/d4_openrouter_judge_screen_${RUN_TAG}}"
MAX_SOURCE_COMPARISONS="${MAX_SOURCE_COMPARISONS:-100}"
ESTIMATE_ONLY="${ESTIMATE_ONLY:-0}"
MODEL_SPECS="${MODEL_SPECS:-qwen35_9b_it=qwen/qwen3.5-9b|gemma3_12b_it=google/gemma-3-12b-it|mistral32_24b_it=mistralai/mistral-small-3.2-24b-instruct|llama33_70b_it=meta-llama/llama-3.3-70b-instruct|qwen35_397b_a17b=qwen/qwen3.5-397b-a17b|gemini31_flash_lite=google/gemini-3.1-flash-lite|gpt54_nano=openai/gpt-5.4-nano|gpt54_mini=openai/gpt-5.4-mini}"
CONTRAST_SPECS="${CONTRAST_SPECS:-}"

cd "$WORKDIR"
export PYTHONPATH="$WORKDIR/src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$PAIR_ROOT"
"$PYTHON" -m aisafety.scripts.build_d4_human_llm_stage_contrast_pairs \
  --workspace-root "$WORKDIR" \
  --pair-jsonl "$PAIR_JSONL" \
  --out-dir "$PAIR_ROOT/hllm" \
  --max-pairs "$MAX_SOURCE_COMPARISONS" \
  --include-order-swaps

"$PYTHON" -m aisafety.scripts.build_d4_bt_stage_contrast_pairs \
  --workspace-root "$WORKDIR" \
  --counterfactual-jsonl "$GENERATED_COUNTERFACTUAL_JSONL" \
  --out-dir "$PAIR_ROOT/generated" \
  --axes generated_assistant_style \
  --max-counterfactuals "$MAX_SOURCE_COMPARISONS" \
  --include-order-swaps

args=(
  "$PYTHON" -m aisafety.scripts.run_d4_openrouter_judge_screen
  --workspace-root "$WORKDIR"
  --dataset "original_hllm=$PAIR_ROOT/hllm/bt_pairs.jsonl"
  --dataset "generated_ai_tone=$PAIR_ROOT/generated/bt_pairs.jsonl"
  --max-source-comparisons "$MAX_SOURCE_COMPARISONS"
  --out-dir "$OUT_DIR"
)

IFS='|' read -r -a model_specs <<< "$MODEL_SPECS"
for spec in "${model_specs[@]}"; do
  args+=(--model "$spec")
done
IFS='|' read -r -a contrast_specs <<< "$CONTRAST_SPECS"
for spec in "${contrast_specs[@]}"; do
  if [[ -n "$spec" ]]; then
    args+=(--contrast "$spec")
  fi
done
if [[ "$ESTIMATE_ONLY" == "1" ]]; then
  args+=(--estimate-only)
fi

"${args[@]}"
echo "out_dir=$OUT_DIR"
