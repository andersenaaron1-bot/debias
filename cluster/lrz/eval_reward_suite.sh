#!/bin/bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-google/gemma-2-9b-it}"
WORKDIR="${WORKDIR:-/workspace}"
RUN_DIR="${1:?usage: eval_reward_suite.sh /workspace/outputs/m2_full_v1}"
OUTPUT_DIR="${2:-$RUN_DIR/eval}"

ADAPTER_DIR="$RUN_DIR/lora_adapter"
VALUE_HEAD="$RUN_DIR/value_head.pt"

mkdir -p "$OUTPUT_DIR"
cd "$WORKDIR"

python -m aisafety.scripts.eval_pref_retention \
  --pref-jsonl "$WORKDIR/data/derived/pref_pairs_shp2/pref_pairs_val.jsonl" \
  --model-id "$MODEL_ID" \
  --lora-adapter-dir "$ADAPTER_DIR" \
  --value-head "$VALUE_HEAD" \
  --out-json "$OUTPUT_DIR/pref_retention.json"

python -m aisafety.scripts.eval_style_sensitivity \
  --style-jsonl "$WORKDIR/data/derived/style_groups/m2_publishable_v1/style_groups_val.jsonl" \
  --model-id "$MODEL_ID" \
  --lora-adapter-dir "$ADAPTER_DIR" \
  --value-head "$VALUE_HEAD" \
  --out-json "$OUTPUT_DIR/style_sensitivity.json" \
  --out-csv "$OUTPUT_DIR/style_sensitivity.csv"
