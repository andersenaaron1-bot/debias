#!/bin/bash
# Build/reuse Laurito movie/product/paper human-vs-LLM pairs, then run the
# generic base-vs-instruct human-vs-LLM stage matrix on that pair set.

set -euo pipefail

WORKDIR="${WORKDIR:-$(pwd)}"
ARTROOT="${ARTROOT:-$WORKDIR}"
PYTHON="${PYTHON:-python}"
RUN_TAG="${RUN_TAG:-laurito_base_instruct_hllm_stage_v1}"

LAURITO_PAIR_OUT_DIR="${LAURITO_PAIR_OUT_DIR:-$ARTROOT/data/derived/d4_laurito_human_llm_pairs_v1}"
LAURITO_PAIR_JSONL="${LAURITO_PAIR_JSONL:-$LAURITO_PAIR_OUT_DIR/pairs.jsonl}"
LAURITO_TRIALS_CSV="${LAURITO_TRIALS_CSV:-}"
LAURITO_INCLUDE_ITEM_TYPES="${LAURITO_INCLUDE_ITEM_TYPES:-movie,paper,product}"
LAURITO_MIN_TOKENS="${LAURITO_MIN_TOKENS:-20}"
LAURITO_MAX_TOKENS="${LAURITO_MAX_TOKENS:-900}"
LAURITO_MAX_PAIRS_PER_ITEM_TYPE="${LAURITO_MAX_PAIRS_PER_ITEM_TYPE:-0}"
LAURITO_MAX_TOTAL_PAIRS="${LAURITO_MAX_TOTAL_PAIRS:-0}"
REBUILD_LAURITO_PAIRS="${REBUILD_LAURITO_PAIRS:-0}"

cd "$WORKDIR"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$WORKDIR/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$REBUILD_LAURITO_PAIRS" == "1" || ! -f "$LAURITO_PAIR_JSONL" ]]; then
  echo "Building Laurito human-vs-LLM pairs"
  echo "  out_dir=$LAURITO_PAIR_OUT_DIR"
  echo "  trials_csv=${LAURITO_TRIALS_CSV:-<local movie/product/paper data>}"
  echo "  item_types=$LAURITO_INCLUDE_ITEM_TYPES"
  echo "  token_range=$LAURITO_MIN_TOKENS..$LAURITO_MAX_TOKENS"
  build_args=(
    "$PYTHON" -m aisafety.scripts.build_d4_laurito_human_llm_pairs
    --workspace-root "$WORKDIR"
    --out-dir "$LAURITO_PAIR_OUT_DIR"
    --include-item-types "$LAURITO_INCLUDE_ITEM_TYPES"
    --min-tokens "$LAURITO_MIN_TOKENS"
    --max-tokens "$LAURITO_MAX_TOKENS"
    --max-pairs-per-item-type "$LAURITO_MAX_PAIRS_PER_ITEM_TYPE"
    --max-total-pairs "$LAURITO_MAX_TOTAL_PAIRS"
  )
  if [[ -n "$LAURITO_TRIALS_CSV" ]]; then
    build_args+=(--trials-csv "$LAURITO_TRIALS_CSV")
  fi
  "${build_args[@]}"
else
  echo "Reusing Laurito pairs: $LAURITO_PAIR_JSONL"
fi

PAIR_JSONL="$LAURITO_PAIR_JSONL" \
RUN_TAG="$RUN_TAG" \
bash cluster/local/run_d4_human_llm_base_instruct_matrix.sh
