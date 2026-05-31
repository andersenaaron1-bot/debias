#!/bin/bash
# Generate paired plain-versus-assistant rewrites for the D4 style-causality suite.

set -euo pipefail

WORKDIR="${WORKDIR:-$(pwd)}"
ARTROOT="${ARTROOT:-$WORKDIR}"
PYTHON="${PYTHON:-python}"
PAIR_JSONL="${PAIR_JSONL:-$ARTROOT/data/derived/d4_human_llm_alignment_pairs_matched_lenlex_relaxed_v1/pairs.jsonl}"
SEED_OUT_DIR="${SEED_OUT_DIR:-$ARTROOT/data/derived/d4_assistant_style_generated_seeds_v1}"
STYLE_PAIR_OUT_DIR="${STYLE_PAIR_OUT_DIR:-$ARTROOT/data/derived/d4_assistant_style_openrouter_pairs_v1}"
COUNTERFACTUAL_OUT_DIR="${COUNTERFACTUAL_OUT_DIR:-$ARTROOT/data/derived/d4_assistant_style_generated_counterfactual_pairs_v1}"
GENERATOR_MODEL="${GENERATOR_MODEL:-openai/gpt-4.1-mini}"
NUM_SEEDS="${NUM_SEEDS:-300}"
MAX_TOKENS="${MAX_TOKENS:-512}"
OVERWRITE="${OVERWRITE:-0}"

cd "$WORKDIR"
export PYTHONPATH="$WORKDIR/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY must be set for generated rewrites." >&2
  exit 2
fi

"$PYTHON" -m aisafety.scripts.build_d4_style_rewrite_seeds \
  --workspace-root "$WORKDIR" \
  --pair-jsonl "$PAIR_JSONL" \
  --out-dir "$SEED_OUT_DIR" \
  --max-seeds "$NUM_SEEDS"

generator_args=(
  "$PYTHON" -m aisafety.scripts.build_openrouter_style_pairs
  --dimensions ai_tone
  --seed-source jsonl
  --seed-jsonl "$SEED_OUT_DIR/seeds.jsonl"
  --out-dir "$STYLE_PAIR_OUT_DIR"
  --num-seeds "$NUM_SEEDS"
  --model "$GENERATOR_MODEL"
  --temperature 0.2
  --max-tokens "$MAX_TOKENS"
  --require-english
  --drop-cjk
  --skip-failed
  --error-log "$STYLE_PAIR_OUT_DIR/errors.log"
)
if [[ "$OVERWRITE" == "1" ]]; then
  generator_args+=(--overwrite)
fi
"${generator_args[@]}"

"$PYTHON" -m aisafety.scripts.convert_d4_generated_style_counterfactual_pairs \
  --workspace-root "$WORKDIR" \
  --style-pairs-jsonl "$STYLE_PAIR_OUT_DIR/ai_tone.jsonl" \
  --out-dir "$COUNTERFACTUAL_OUT_DIR"

echo "generated_counterfactuals=$COUNTERFACTUAL_OUT_DIR/counterfactuals.jsonl"
