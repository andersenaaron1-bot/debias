#!/bin/bash
# Build anchored probes and run the Qwen base/instruct causal patching comparison.

set -euo pipefail

WORKDIR="${WORKDIR:-$(pwd)}"
ARTROOT="${ARTROOT:-$WORKDIR}"
PYTHON="${PYTHON:-python}"
ANCHORED_BT_ROOT="${ANCHORED_BT_ROOT:-$ARTROOT/data/derived/d4_anchored_style_bt_pairs_v1}"

cd "$WORKDIR"
export PYTHONPATH="$WORKDIR/src${PYTHONPATH:+:$PYTHONPATH}"

WORKDIR="$WORKDIR" ARTROOT="$ARTROOT" PYTHON="$PYTHON" OUT_ROOT="$ANCHORED_BT_ROOT" \
  bash cluster/local/prepare_d4_anchored_style_bt_pairs.sh

extra_eval=()
for dataset in original_hllm laurito_hllm preference_retention; do
  path="$ANCHORED_BT_ROOT/$dataset/bt_pairs.jsonl"
  if [[ -s "$path" ]]; then
    extra_eval+=("$dataset=$path")
  fi
done
extra_eval_specs=""
if [[ "${#extra_eval[@]}" -gt 0 ]]; then
  extra_eval_specs="$(IFS='|'; echo "${extra_eval[*]}")"
fi

BT_GENERATED="$ANCHORED_BT_ROOT/generated/bt_pairs.jsonl" \
BT_ATOMIC="$ANCHORED_BT_ROOT/atomic/bt_pairs.jsonl" \
BT_COMPOSITE="$ANCHORED_BT_ROOT/composite/bt_pairs.jsonl" \
EXTRA_EVAL_SPECS="$extra_eval_specs" \
  bash cluster/local/run_d4_qwen_decision_patching_suite.sh
