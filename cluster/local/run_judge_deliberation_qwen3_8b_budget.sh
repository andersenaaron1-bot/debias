#!/bin/bash
# Run the Qwen3-8B deliberation token-budget experiment on ipe-monster.

set -euo pipefail

WORKDIR="${WORKDIR:-$(pwd)}"
ARTROOT="${ARTROOT:-$WORKDIR}"
PYTHON="${PYTHON:-$WORKDIR/.venv/bin/python}"
RUN_TAG="${RUN_TAG:-judge_deliberation_qwen3_8b_budget_scout_v1}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
RUN_LABEL="${RUN_LABEL:-qwen3_8b_it}"
GPU="${GPU:-7}"

BASE_SOURCE_DIR="${BASE_SOURCE_DIR:-$ARTROOT/data/derived/judge_reasoning_sources_v1}"
SOURCE_DIR="${SOURCE_DIR:-$ARTROOT/data/derived/judge_deliberation_sources_v1}"
CONFIG="${CONFIG:-$WORKDIR/configs/datasets/judge_deliberation_progression_v1.json}"
SUITE_DIR="${SUITE_DIR:-$ARTROOT/data/derived/judge_deliberation_suite_$RUN_TAG}"
OUT_ROOT="${OUT_ROOT:-$ARTROOT/artifacts/mechanistic/$RUN_TAG}"
BUDGET_DIR="${BUDGET_DIR:-$OUT_ROOT/budget_sweep}"
ANALYSIS_DIR="${ANALYSIS_DIR:-$OUT_ROOT/budget_analysis}"
LOG_DIR="${LOG_DIR:-$OUT_ROOT/logs}"

MAX_SOURCE_PAIRS="${MAX_SOURCE_PAIRS:-200}"
MAX_PAIRS_PER_DATASET="${MAX_PAIRS_PER_DATASET:-30}"
INCLUDE_DATASETS="${INCLUDE_DATASETS:-}"
BRANCHES_PER_COMPARISON="${BRANCHES_PER_COMPARISON:-3}"
BUDGET_TOKENS="${BUDGET_TOKENS:-0,128,256,512,1024,2048}"
MAX_PAIRS="${MAX_PAIRS:-0}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_SCORE_LENGTH="${MAX_SCORE_LENGTH:-8192}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-4}"
MAX_NEW_TOKENS_DIRECT="${MAX_NEW_TOKENS_DIRECT:-16}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
BOOTSTRAP="${BOOTSTRAP:-5000}"
USE_4BIT="${USE_4BIT:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
FORCE_SOURCE_REBUILD="${FORCE_SOURCE_REBUILD:-0}"
FORCE_SUITE_REBUILD="${FORCE_SUITE_REBUILD:-0}"
ANALYZE_ONLY="${ANALYZE_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"
SEED="${SEED:-1234}"

cd "$WORKDIR"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$WORKDIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$ARTROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$LOG_DIR"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python executable is missing or not executable: $PYTHON" >&2
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "Missing progression config: $CONFIG" >&2
  exit 1
fi

echo "Running local Qwen3-8B deliberation budget sweep"
echo "  run_tag=$RUN_TAG"
echo "  workdir=$WORKDIR"
echo "  artroot=$ARTROOT"
echo "  model=$RUN_LABEL :: $MODEL_ID :: gpu $GPU"
echo "  source_dir=$SOURCE_DIR"
echo "  suite_dir=$SUITE_DIR"
echo "  budget_dir=$BUDGET_DIR"
echo "  budgets=$BUDGET_TOKENS"
echo "  branches_per_comparison=$BRANCHES_PER_COMPARISON"
echo "  max_pairs_per_dataset=$MAX_PAIRS_PER_DATASET"
echo "  include_datasets=${INCLUDE_DATASETS:-all}"
echo "  logs=$LOG_DIR"

run_logged() {
  local name="$1"
  shift
  echo "run $name"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '  '
    printf '%q ' "$@"
    printf '\n'
    return 0
  fi
  "$@" >"$LOG_DIR/${name}.out" 2>"$LOG_DIR/${name}.err"
}

check_base_sources() {
  local required=(
    arc_challenge.jsonl
    truthfulqa_binary.jsonl
    ethics_commonsense_hard.jsonl
    helpsteer2_dominated.jsonl
    helpsteer2_tradeoff.jsonl
    d4_human_llm.jsonl
  )
  local missing=0
  local name
  for name in "${required[@]}"; do
    if [[ ! -s "$BASE_SOURCE_DIR/$name" ]]; then
      echo "Missing base source: $BASE_SOURCE_DIR/$name" >&2
      missing=1
    fi
  done
  if [[ "$missing" != "0" ]]; then
    echo "Build judge_reasoning_sources_v1 before running this experiment." >&2
    exit 1
  fi
}

build_sources() {
  if [[ "$FORCE_SOURCE_REBUILD" != "1" && -s "$SOURCE_DIR/manifest.json" ]]; then
    echo "skip existing source pack -> $SOURCE_DIR"
    return 0
  fi
  check_base_sources
  run_logged build_sources \
    "$PYTHON" -m aisafety.scripts.build_judge_deliberation_source_pack \
    --workspace-root "$WORKDIR" \
    --artifact-root "$ARTROOT" \
    --base-source-dir "$BASE_SOURCE_DIR" \
    --cache-dir "$HF_DATASETS_CACHE" \
    --max-pairs-per-dataset "$MAX_SOURCE_PAIRS" \
    --seed "$SEED" \
    --out-dir "$SOURCE_DIR"
}

build_suite() {
  if [[ "$FORCE_SUITE_REBUILD" != "1" && -s "$SUITE_DIR/manifest.json" && -s "$SUITE_DIR/comparisons.jsonl" ]]; then
    echo "skip existing comparison suite -> $SUITE_DIR"
    return 0
  fi
  local args=(
    "$PYTHON" -m aisafety.scripts.build_judge_reasoning_suite
    --workspace-root "$WORKDIR" \
    --input-root "$ARTROOT" \
    --config "$CONFIG" \
    --max-pairs-per-dataset "$MAX_PAIRS_PER_DATASET" \
    --seed "$SEED" \
    --out-dir "$SUITE_DIR"
  )
  if [[ -n "$INCLUDE_DATASETS" ]]; then
    args+=(--include-datasets "$INCLUDE_DATASETS")
  fi
  run_logged build_suite "${args[@]}"
}

run_budget_sweep() {
  if [[ "$SKIP_EXISTING" == "1" && -s "$BUDGET_DIR/manifest.json" ]]; then
    echo "skip existing budget sweep -> $BUDGET_DIR"
    return 0
  fi
  if [[ "$DRY_RUN" != "1" && ! -s "$SUITE_DIR/comparisons.jsonl" ]]; then
    echo "Missing comparison suite: $SUITE_DIR/comparisons.jsonl" >&2
    exit 1
  fi

  local args=(
    "$PYTHON" -m aisafety.scripts.run_judge_reasoning_budget_sweep
    --workspace-root "$WORKDIR"
    --comparisons-jsonl "$SUITE_DIR/comparisons.jsonl"
    --run-label "$RUN_LABEL"
    --model-id "$MODEL_ID"
    --cache-dir "$HF_HOME"
    --prompt-style chat_template
    --budget-tokens "$BUDGET_TOKENS"
    --branches-per-comparison "$BRANCHES_PER_COMPARISON"
    --max-pairs "$MAX_PAIRS"
    --cap-strategy source_round_robin
    --max-prompt-length "$MAX_PROMPT_LENGTH"
    --max-score-length "$MAX_SCORE_LENGTH"
    --score-batch-size "$SCORE_BATCH_SIZE"
    --max-new-tokens-direct "$MAX_NEW_TOKENS_DIRECT"
    --temperature "$TEMPERATURE"
    --top-p "$TOP_P"
    --top-k "$TOP_K"
    --seed "$SEED"
    --out-dir "$BUDGET_DIR"
  )
  if [[ "$USE_4BIT" == "1" ]]; then
    args+=(--use-4bit)
  fi
  if [[ -s "$BUDGET_DIR/budget_scores.jsonl" || -s "$BUDGET_DIR/reasoning_traces.jsonl" ]]; then
    args+=(--resume)
  fi

  echo "launch budget sweep gpu=$GPU model=$MODEL_ID"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '  CUDA_VISIBLE_DEVICES=%q ' "$GPU"
    printf '%q ' "${args[@]}"
    printf '\n'
    return 0
  fi
  CUDA_VISIBLE_DEVICES="$GPU" "${args[@]}" \
    >"$LOG_DIR/budget_sweep.out" 2>"$LOG_DIR/budget_sweep.err"
}

run_analysis() {
  if [[ "$SKIP_EXISTING" == "1" && -s "$ANALYSIS_DIR/manifest.json" ]]; then
    echo "skip existing budget analysis -> $ANALYSIS_DIR"
    return 0
  fi
  if [[ "$DRY_RUN" != "1" && ! -s "$BUDGET_DIR/manifest.json" ]]; then
    echo "Missing completed budget manifest: $BUDGET_DIR/manifest.json" >&2
    exit 1
  fi
  run_logged budget_analysis \
    "$PYTHON" -m aisafety.scripts.analyze_judge_reasoning_budget_sweep \
    --workspace-root "$WORKDIR" \
    --run-dir "$BUDGET_DIR" \
    --bootstrap "$BOOTSTRAP" \
    --seed "$SEED" \
    --out-dir "$ANALYSIS_DIR"
}

if [[ "$ANALYZE_ONLY" != "1" ]]; then
  build_sources
  build_suite
  run_budget_sweep
fi
run_analysis

echo "COMPLETE"
echo "out_root=$OUT_ROOT"
echo "budget_dir=$BUDGET_DIR"
echo "analysis_dir=$ANALYSIS_DIR"
echo "logs=$LOG_DIR"
