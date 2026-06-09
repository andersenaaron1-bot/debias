#!/bin/bash
# Run the matched HelpSteer2 criterion-switching scout on ipe-monster.

set -euo pipefail

WORKDIR="${WORKDIR:-$(pwd)}"
ARTROOT="${ARTROOT:-$WORKDIR}"
PYTHON="${PYTHON:-$WORKDIR/.venv/bin/python}"
RUN_TAG="${RUN_TAG:-helpsteer2_matched_criterion_qwen3_8b_scout_v1}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
RUN_LABEL="${RUN_LABEL:-qwen3_8b_it}"
GPU_0="${GPU_0:-0}"
GPU_1="${GPU_1:-1}"

PAIR_DIR="${PAIR_DIR:-$ARTROOT/data/derived/helpsteer2_matched_criterion_suite_$RUN_TAG}"
OUT_ROOT="${OUT_ROOT:-$ARTROOT/artifacts/mechanistic/$RUN_TAG}"
SHARD_0_DIR="${SHARD_0_DIR:-$OUT_ROOT/budget_shard_0}"
SHARD_1_DIR="${SHARD_1_DIR:-$OUT_ROOT/budget_shard_1}"
ANALYSIS_DIR="${ANALYSIS_DIR:-$OUT_ROOT/analysis}"
LOG_DIR="${LOG_DIR:-$OUT_ROOT/logs}"

MAX_PAIRS_PER_STRATUM="${MAX_PAIRS_PER_STRATUM:-8}"
MIN_PAIRS_PER_STRATUM="${MIN_PAIRS_PER_STRATUM:-4}"
WEIGHTED_TIE_EPSILON="${WEIGHTED_TIE_EPSILON:-0.05}"
BUDGET_TOKENS="${BUDGET_TOKENS:-0,128,512,1024}"
BRANCHES_PER_COMPARISON="${BRANCHES_PER_COMPARISON:-2}"
SCORE_BATCH_SIZE_GPU_0="${SCORE_BATCH_SIZE_GPU_0:-2}"
SCORE_BATCH_SIZE_GPU_1="${SCORE_BATCH_SIZE_GPU_1:-4}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_SCORE_LENGTH="${MAX_SCORE_LENGTH:-8192}"
MAX_NEW_TOKENS_DIRECT="${MAX_NEW_TOKENS_DIRECT:-16}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
BOOTSTRAP="${BOOTSTRAP:-5000}"
USE_4BIT="${USE_4BIT:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
FORCE_PAIR_REBUILD="${FORCE_PAIR_REBUILD:-0}"
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
mkdir -p \
  "$HF_HOME" \
  "$TRANSFORMERS_CACHE" \
  "$HF_DATASETS_CACHE" \
  "$LOG_DIR"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python executable is missing or not executable: $PYTHON" >&2
  exit 1
fi

IFS=',' read -r -a budget_values <<<"$BUDGET_TOKENS"
planned_pairs=$((4 * MAX_PAIRS_PER_STRATUM))
planned_comparisons=$((planned_pairs * 5 * 2))
planned_traces=$((planned_comparisons * BRANCHES_PER_COMPARISON))
planned_scores=$((planned_comparisons + planned_traces * ${#budget_values[@]}))

echo "Running matched HelpSteer2 criterion-switching scout"
echo "  run_tag=$RUN_TAG"
echo "  workdir=$WORKDIR"
echo "  artroot=$ARTROOT"
echo "  model=$RUN_LABEL :: $MODEL_ID"
echo "  gpus=$GPU_0,$GPU_1"
echo "  pair_dir=$PAIR_DIR"
echo "  out_root=$OUT_ROOT"
echo "  pairs_per_stratum=$MAX_PAIRS_PER_STRATUM"
echo "  budgets=$BUDGET_TOKENS"
echo "  branches_per_comparison=$BRANCHES_PER_COMPARISON"
echo "  planned_upper_bound=pairs:$planned_pairs comparisons:$planned_comparisons traces:$planned_traces scores:$planned_scores"
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

build_pairs() {
  if [[ "$FORCE_PAIR_REBUILD" != "1" \
        && -s "$PAIR_DIR/manifest.json" \
        && -s "$PAIR_DIR/comparisons_shard_0.jsonl" \
        && -s "$PAIR_DIR/comparisons_shard_1.jsonl" ]]; then
    echo "skip existing matched suite -> $PAIR_DIR"
    return 0
  fi
  run_logged build_pairs \
    "$PYTHON" -m aisafety.scripts.build_helpsteer2_matched_criterion_suite \
    --workspace-root "$WORKDIR" \
    --artifact-root "$ARTROOT" \
    --cache-dir "$HF_DATASETS_CACHE" \
    --max-pairs-per-stratum "$MAX_PAIRS_PER_STRATUM" \
    --min-pairs-per-stratum "$MIN_PAIRS_PER_STRATUM" \
    --num-shards 2 \
    --weighted-tie-epsilon "$WEIGHTED_TIE_EPSILON" \
    --seed "$SEED" \
    --out-dir "$PAIR_DIR"
}

run_shard() {
  local shard_index="$1"
  local gpu="$2"
  local score_batch_size="$3"
  local shard_file="$PAIR_DIR/comparisons_shard_${shard_index}.jsonl"
  local shard_dir="$OUT_ROOT/budget_shard_${shard_index}"
  local log_name="budget_shard_${shard_index}"

  if [[ "$SKIP_EXISTING" == "1" && -s "$shard_dir/manifest.json" ]]; then
    echo "skip existing shard $shard_index -> $shard_dir"
    return 0
  fi
  if [[ "$DRY_RUN" != "1" && ! -s "$shard_file" ]]; then
    echo "Missing comparison shard: $shard_file" >&2
    return 1
  fi

  local args=(
    "$PYTHON" -m aisafety.scripts.run_judge_reasoning_budget_sweep
    --workspace-root "$WORKDIR"
    --comparisons-jsonl "$shard_file"
    --run-label "$RUN_LABEL"
    --model-id "$MODEL_ID"
    --cache-dir "$HF_HOME"
    --prompt-style chat_template
    --labels A,B,C
    --budget-tokens "$BUDGET_TOKENS"
    --branches-per-comparison "$BRANCHES_PER_COMPARISON"
    --max-pairs 0
    --cap-strategy global
    --max-prompt-length "$MAX_PROMPT_LENGTH"
    --max-score-length "$MAX_SCORE_LENGTH"
    --score-batch-size "$score_batch_size"
    --max-new-tokens-direct "$MAX_NEW_TOKENS_DIRECT"
    --temperature "$TEMPERATURE"
    --top-p "$TOP_P"
    --top-k "$TOP_K"
    --seed "$SEED"
    --out-dir "$shard_dir"
  )
  if [[ "$USE_4BIT" == "1" ]]; then
    args+=(--use-4bit)
  fi
  if [[ -s "$shard_dir/budget_scores.jsonl" \
        || -s "$shard_dir/reasoning_traces.jsonl" ]]; then
    args+=(--resume)
  fi

  echo "launch shard=$shard_index gpu=$gpu comparisons=$shard_file"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '  CUDA_VISIBLE_DEVICES=%q ' "$gpu"
    printf '%q ' "${args[@]}"
    printf '\n'
    return 0
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "${args[@]}" \
    >"$LOG_DIR/${log_name}.out" 2>"$LOG_DIR/${log_name}.err"
}

run_analysis() {
  if [[ "$SKIP_EXISTING" == "1" && -s "$ANALYSIS_DIR/manifest.json" ]]; then
    echo "skip existing matched analysis -> $ANALYSIS_DIR"
    return 0
  fi
  if [[ "$DRY_RUN" != "1" \
        && ( ! -s "$SHARD_0_DIR/manifest.json" \
          || ! -s "$SHARD_1_DIR/manifest.json" ) ]]; then
    echo "Both budget shard manifests are required before analysis." >&2
    exit 1
  fi
  run_logged matched_analysis \
    "$PYTHON" -m aisafety.scripts.analyze_helpsteer2_matched_criterion \
    --workspace-root "$WORKDIR" \
    --run-dir "$SHARD_0_DIR" \
    --run-dir "$SHARD_1_DIR" \
    --bootstrap "$BOOTSTRAP" \
    --seed "$SEED" \
    --out-dir "$ANALYSIS_DIR"
}

if [[ "$ANALYZE_ONLY" != "1" ]]; then
  build_pairs
  if [[ "$DRY_RUN" == "1" ]]; then
    run_shard 0 "$GPU_0" "$SCORE_BATCH_SIZE_GPU_0"
    run_shard 1 "$GPU_1" "$SCORE_BATCH_SIZE_GPU_1"
  else
    run_shard 0 "$GPU_0" "$SCORE_BATCH_SIZE_GPU_0" &
    shard_0_pid=$!
    run_shard 1 "$GPU_1" "$SCORE_BATCH_SIZE_GPU_1" &
    shard_1_pid=$!
    status=0
    wait "$shard_0_pid" || status=1
    wait "$shard_1_pid" || status=1
    if [[ "$status" != "0" ]]; then
      echo "At least one HelpSteer2 budget shard failed. Check $LOG_DIR." >&2
      exit 1
    fi
  fi
fi
run_analysis

echo "COMPLETE"
echo "out_root=$OUT_ROOT"
echo "pair_dir=$PAIR_DIR"
echo "analysis_dir=$ANALYSIS_DIR"
echo "logs=$LOG_DIR"
