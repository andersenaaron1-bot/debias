#!/bin/bash
# Build anchored style probes and fixed-reference retention controls locally.

set -euo pipefail

WORKDIR="${WORKDIR:-$(pwd)}"
ARTROOT="${ARTROOT:-$WORKDIR}"
PYTHON="${PYTHON:-python}"
PAIR_JSONL="${PAIR_JSONL:-$ARTROOT/data/derived/d4_human_llm_alignment_pairs_matched_lenlex_relaxed_v1/pairs.jsonl}"
LAURITO_PAIR_JSONL="${LAURITO_PAIR_JSONL:-$ARTROOT/data/derived/d4_laurito_human_llm_pairs_v1/pairs.jsonl}"
PREF_PAIR_JSONL="${PREF_PAIR_JSONL:-$ARTROOT/data/derived/pref_pairs_shp2/pref_pairs_val.jsonl}"
COMPOSITE_COUNTERFACTUAL_JSONL="${COMPOSITE_COUNTERFACTUAL_JSONL:-$ARTROOT/data/derived/d4_surface_counterfactual_pairs_matched_lenlex_relaxed_v1/counterfactuals.jsonl}"
ATOMIC_COUNTERFACTUAL_JSONL="${ATOMIC_COUNTERFACTUAL_JSONL:-$ARTROOT/data/derived/d4_assistant_style_atomic_counterfactual_pairs_v1/counterfactuals.jsonl}"
GENERATED_COUNTERFACTUAL_JSONL="${GENERATED_COUNTERFACTUAL_JSONL:-$ARTROOT/data/derived/d4_assistant_style_generated_counterfactual_pairs_v1/counterfactuals.jsonl}"
OUT_ROOT="${OUT_ROOT:-$ARTROOT/data/derived/d4_anchored_style_bt_pairs_v1}"
MAX_CONTROL_PAIRS="${MAX_CONTROL_PAIRS:-1000}"

cd "$WORKDIR"
export PYTHONPATH="$WORKDIR/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -s "$PAIR_JSONL" ]]; then
  echo "Missing matched pair file: $PAIR_JSONL" >&2
  exit 2
fi

build_style() {
  local dataset="$1"
  local counterfactual_jsonl="$2"
  if [[ ! -s "$counterfactual_jsonl" ]]; then
    echo "skip missing anchored style probe $dataset -> $counterfactual_jsonl"
    return
  fi
  "$PYTHON" -m aisafety.scripts.build_d4_anchored_style_bt_pairs \
    --workspace-root "$WORKDIR" \
    --counterfactual-jsonl "$counterfactual_jsonl" \
    --pair-jsonl "$PAIR_JSONL" \
    --out-dir "$OUT_ROOT/$dataset"
}

build_control() {
  local dataset="$1"
  local mode="$2"
  local pair_jsonl="$3"
  if [[ ! -s "$pair_jsonl" ]]; then
    echo "skip missing fixed-reference control $dataset -> $pair_jsonl"
    return
  fi
  "$PYTHON" -m aisafety.scripts.build_d4_fixed_reference_bt_controls \
    --workspace-root "$WORKDIR" \
    --pair-jsonl "$pair_jsonl" \
    --mode "$mode" \
    --max-pairs "$MAX_CONTROL_PAIRS" \
    --out-dir "$OUT_ROOT/$dataset"
}

echo "Building anchored style probes"
echo "  out_root=$OUT_ROOT"
build_style generated "$GENERATED_COUNTERFACTUAL_JSONL"
build_style atomic "$ATOMIC_COUNTERFACTUAL_JSONL"
build_style composite "$COMPOSITE_COUNTERFACTUAL_JSONL"
build_control original_hllm human_llm "$PAIR_JSONL"
build_control laurito_hllm human_llm "$LAURITO_PAIR_JSONL"
build_control preference_retention preference "$PREF_PAIR_JSONL"

echo "out_root=$OUT_ROOT"
