#!/bin/bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-}"
WORKDIR="${WORKDIR:-/workspace}"
RUN_DIR="${1:?usage: eval_reward_suite.sh /workspace/outputs/m2_full_v1}"
OUTPUT_DIR="${2:-$RUN_DIR/eval}"
BENCHMARKS="${BENCHMARKS:-arc_challenge,hellaswag,winogrande,piqa,social_iqa,boolq,mmlu}"
BENCHMARK_MAX_EXAMPLES="${BENCHMARK_MAX_EXAMPLES:-250}"
TRIAD_REWRITE_JSONL="${TRIAD_REWRITE_JSONL:-}"
LAURITO_TRIALS_CSV="${LAURITO_TRIALS_CSV:-}"
CACHE_DIR="${CACHE_DIR:-${HF_HOME:-}}"

mkdir -p "$OUTPUT_DIR"
cd "$WORKDIR"

CMD=(
  python -m aisafety.scripts.run_full_reward_eval
  --run-dir "$RUN_DIR"
  --out-dir "$OUTPUT_DIR"
  --workspace-root "$WORKDIR"
  --benchmark "$BENCHMARKS"
  --benchmark-max-examples "$BENCHMARK_MAX_EXAMPLES"
)

if [[ -n "$MODEL_ID" ]]; then
  CMD+=(--model-id "$MODEL_ID")
fi
if [[ -n "$CACHE_DIR" ]]; then
  CMD+=(--cache-dir "$CACHE_DIR")
fi
if [[ -n "$TRIAD_REWRITE_JSONL" ]]; then
  CMD+=(--triad-rewrite-jsonl "$TRIAD_REWRITE_JSONL")
fi
if [[ -n "$LAURITO_TRIALS_CSV" ]]; then
  CMD+=(--laurito-trials-csv "$LAURITO_TRIALS_CSV")
fi

"${CMD[@]}"
