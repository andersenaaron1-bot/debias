#!/bin/bash
# Run a focused factual direct-vs-CoT behavior sweep on ipe-monster.

set -euo pipefail

WORKDIR="${WORKDIR:-$(pwd)}"
ARTROOT="${ARTROOT:-$WORKDIR}"
PYTHON="${PYTHON:-$WORKDIR/.venv/bin/python}"
RUN_TAG="${RUN_TAG:-judge_factual_cot_effect_qwen3_8b_v1}"
GPU="${GPU:-7}"

OUT_ROOT="${OUT_ROOT:-$ARTROOT/artifacts/mechanistic/$RUN_TAG}"
BUDGET_DIR="${BUDGET_DIR:-$OUT_ROOT/budget_sweep}"
BUDGET_ANALYSIS_DIR="${BUDGET_ANALYSIS_DIR:-$OUT_ROOT/budget_analysis}"
FACTUAL_ANALYSIS_DIR="${FACTUAL_ANALYSIS_DIR:-$OUT_ROOT/factual_cot_effect}"
LOG_DIR="${LOG_DIR:-$OUT_ROOT/logs}"

INCLUDE_DATASETS="${INCLUDE_DATASETS:-gsm8k_verification,math500_verification,bbh_logical_deduction,arc_challenge,truthfulqa}"
MAX_PAIRS_PER_DATASET="${MAX_PAIRS_PER_DATASET:-24}"
BRANCHES_PER_COMPARISON="${BRANCHES_PER_COMPARISON:-3}"
BUDGET_TOKENS="${BUDGET_TOKENS:-0,128,512,2048}"
ENDPOINT_BUDGET="${ENDPOINT_BUDGET:-2048}"
BOOTSTRAP="${BOOTSTRAP:-5000}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-4}"
USE_4BIT="${USE_4BIT:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
SEED="${SEED:-1234}"

cd "$WORKDIR"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$WORKDIR/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$LOG_DIR"

echo "Running factual direct-vs-CoT effect suite"
echo "  run_tag=$RUN_TAG"
echo "  gpu=$GPU"
echo "  include_datasets=$INCLUDE_DATASETS"
echo "  max_pairs_per_dataset=$MAX_PAIRS_PER_DATASET"
echo "  branches_per_comparison=$BRANCHES_PER_COMPARISON"
echo "  budget_tokens=$BUDGET_TOKENS"
echo "  endpoint_budget=$ENDPOINT_BUDGET"
echo "  out_root=$OUT_ROOT"

RUN_TAG="$RUN_TAG" \
GPU="$GPU" \
OUT_ROOT="$OUT_ROOT" \
BUDGET_DIR="$BUDGET_DIR" \
ANALYSIS_DIR="$BUDGET_ANALYSIS_DIR" \
LOG_DIR="$LOG_DIR" \
INCLUDE_DATASETS="$INCLUDE_DATASETS" \
MAX_PAIRS_PER_DATASET="$MAX_PAIRS_PER_DATASET" \
BRANCHES_PER_COMPARISON="$BRANCHES_PER_COMPARISON" \
BUDGET_TOKENS="$BUDGET_TOKENS" \
BOOTSTRAP="$BOOTSTRAP" \
SCORE_BATCH_SIZE="$SCORE_BATCH_SIZE" \
USE_4BIT="$USE_4BIT" \
SKIP_EXISTING="$SKIP_EXISTING" \
DRY_RUN="$DRY_RUN" \
SEED="$SEED" \
bash "$WORKDIR/cluster/local/run_judge_deliberation_qwen3_8b_budget.sh"

echo "run factual_cot_effect_analysis"
if [[ "$DRY_RUN" == "1" ]]; then
  printf '  %q ' "$PYTHON" -m aisafety.scripts.analyze_judge_factual_cot_effect \
    --workspace-root "$WORKDIR" \
    --run-dir "$BUDGET_DIR" \
    --include-datasets "$INCLUDE_DATASETS" \
    --endpoint-budget "$ENDPOINT_BUDGET" \
    --bootstrap "$BOOTSTRAP" \
    --seed "$SEED" \
    --out-dir "$FACTUAL_ANALYSIS_DIR"
  printf '\n'
else
  "$PYTHON" -m aisafety.scripts.analyze_judge_factual_cot_effect \
    --workspace-root "$WORKDIR" \
    --run-dir "$BUDGET_DIR" \
    --include-datasets "$INCLUDE_DATASETS" \
    --endpoint-budget "$ENDPOINT_BUDGET" \
    --bootstrap "$BOOTSTRAP" \
    --seed "$SEED" \
    --out-dir "$FACTUAL_ANALYSIS_DIR" \
    >"$LOG_DIR/factual_cot_effect_analysis.out" \
    2>"$LOG_DIR/factual_cot_effect_analysis.err"
fi

echo "run factual_cot_effect_readout"
if [[ "$DRY_RUN" == "1" ]]; then
  printf '  %q ' "$PYTHON" -m aisafety.scripts.read_judge_factual_cot_effect \
    --workspace-root "$WORKDIR" \
    --input "$FACTUAL_ANALYSIS_DIR" \
    --endpoint-budget "$ENDPOINT_BUDGET"
  printf '\n'
else
  "$PYTHON" -m aisafety.scripts.read_judge_factual_cot_effect \
    --workspace-root "$WORKDIR" \
    --input "$FACTUAL_ANALYSIS_DIR" \
    --endpoint-budget "$ENDPOINT_BUDGET"
fi

echo "COMPLETE"
echo "out_root=$OUT_ROOT"
echo "budget_dir=$BUDGET_DIR"
echo "factual_analysis_dir=$FACTUAL_ANALYSIS_DIR"
echo "logs=$LOG_DIR"
