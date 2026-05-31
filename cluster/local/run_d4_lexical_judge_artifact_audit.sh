#!/bin/bash
# CPU-only lexical/edit-fragment audit over existing D4 judge-score summaries.

set -euo pipefail

WORKDIR="${WORKDIR:-$(pwd)}"
ARTROOT="${ARTROOT:-$WORKDIR}"
PYTHON="${PYTHON:-python}"
OUT_DIR="${OUT_DIR:-$ARTROOT/artifacts/mechanistic/d4_lexical_judge_artifact_audit_v1}"
HLLM_PAIRS_JSONL="${HLLM_PAIRS_JSONL:-}"
SURFACE_COUNTERFACTUAL_JSONL="${SURFACE_COUNTERFACTUAL_JSONL:-$ARTROOT/data/derived/d4_surface_counterfactual_pairs_matched_lenlex_relaxed_v1/counterfactuals.jsonl}"
HLLM_SUMMARY_SPECS="${HLLM_SUMMARY_SPECS:-}"
SURFACE_SUMMARY_SPECS="${SURFACE_SUMMARY_SPECS:-}"
HLLM_TARGET_CONTRASTS="${HLLM_TARGET_CONTRASTS:-}"
TOP_K="${TOP_K:-250}"
WORD_MAX_FEATURES="${WORD_MAX_FEATURES:-12000}"
CHAR_MAX_FEATURES="${CHAR_MAX_FEATURES:-12000}"
WORD_MIN_DF="${WORD_MIN_DF:-5}"
CHAR_MIN_DF="${CHAR_MIN_DF:-10}"
CV_FOLDS="${CV_FOLDS:-5}"
SURFACE_FRAGMENT_MIN_DF="${SURFACE_FRAGMENT_MIN_DF:-5}"

cd "$WORKDIR"
export PYTHONPATH="$WORKDIR/src${PYTHONPATH:+:$PYTHONPATH}"

args=(
  "$PYTHON" -m aisafety.scripts.audit_d4_lexical_judge_artifacts
  --workspace-root "$WORKDIR"
  --out-dir "$OUT_DIR"
  --top-k "$TOP_K"
  --word-max-features "$WORD_MAX_FEATURES"
  --char-max-features "$CHAR_MAX_FEATURES"
  --word-min-df "$WORD_MIN_DF"
  --char-min-df "$CHAR_MIN_DF"
  --cv-folds "$CV_FOLDS"
  --surface-fragment-min-df "$SURFACE_FRAGMENT_MIN_DF"
)

if [[ -n "$HLLM_PAIRS_JSONL" ]]; then
  IFS='|' read -r -a hllm_pair_paths <<< "$HLLM_PAIRS_JSONL"
  for path in "${hllm_pair_paths[@]}"; do
    [[ -n "$path" ]] && args+=(--hllm-pairs-jsonl "$path")
  done
fi
if [[ -n "$HLLM_SUMMARY_SPECS" ]]; then
  IFS='|' read -r -a hllm_specs <<< "$HLLM_SUMMARY_SPECS"
  for spec in "${hllm_specs[@]}"; do
    [[ -n "$spec" ]] && args+=(--hllm-summary "$spec")
  done
fi
if [[ -n "$HLLM_TARGET_CONTRASTS" ]]; then
  IFS='|' read -r -a hllm_contrasts <<< "$HLLM_TARGET_CONTRASTS"
  for contrast in "${hllm_contrasts[@]}"; do
    [[ -n "$contrast" ]] && args+=(--hllm-target-contrast "$contrast")
  done
fi
if [[ -n "$SURFACE_SUMMARY_SPECS" ]]; then
  args+=(--surface-counterfactual-jsonl "$SURFACE_COUNTERFACTUAL_JSONL")
  IFS='|' read -r -a surface_specs <<< "$SURFACE_SUMMARY_SPECS"
  for spec in "${surface_specs[@]}"; do
    [[ -n "$spec" ]] && args+=(--surface-summary "$spec")
  done
fi

echo "Running CPU-only D4 lexical judge-artifact audit"
echo "  out_dir=$OUT_DIR"
echo "  hllm_pairs_jsonl=$HLLM_PAIRS_JSONL"
echo "  hllm_summary_specs=$HLLM_SUMMARY_SPECS"
echo "  hllm_target_contrasts=$HLLM_TARGET_CONTRASTS"
echo "  surface_counterfactual_jsonl=$SURFACE_COUNTERFACTUAL_JSONL"
echo "  surface_summary_specs=$SURFACE_SUMMARY_SPECS"

"${args[@]}"
