#!/bin/bash
# Run the Qwen3 base/IT judge-reasoning trajectory matrix on a local GPU host.

set -euo pipefail

WORKDIR="${WORKDIR:-$(pwd)}"
ARTROOT="${ARTROOT:-$WORKDIR}"
PYTHON="${PYTHON:-python}"
RUN_TAG="${RUN_TAG:-judge_reasoning_qwen3_scout_v1}"
CONFIG="${CONFIG:-$WORKDIR/configs/datasets/judge_reasoning_suite_v1.json}"
PAIR_OUT_DIR="${PAIR_OUT_DIR:-$ARTROOT/data/derived/judge_reasoning_suite_${RUN_TAG}}"
COMPARISONS_JSONL="${COMPARISONS_JSONL:-$PAIR_OUT_DIR/comparisons.jsonl}"
OUT_ROOT="${OUT_ROOT:-$ARTROOT/artifacts/mechanistic/$RUN_TAG}"
LOG_DIR="${LOG_DIR:-$OUT_ROOT/logs}"

BASE_LABEL="${BASE_LABEL:-qwen3_8b_base}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B-Base}"
BASE_PROMPT_STYLE="${BASE_PROMPT_STYLE:-plain}"
BASE_REASONING_MODES="${BASE_REASONING_MODES:-free_reasoning,direct}"
IT_LABEL="${IT_LABEL:-qwen3_8b_it}"
IT_MODEL="${IT_MODEL:-Qwen/Qwen3-8B}"
IT_PROMPT_STYLE="${IT_PROMPT_STYLE:-chat_template}"
IT_REASONING_MODES="${IT_REASONING_MODES:-thinking,direct}"
GPU_A="${GPU_A:-1}"
GPU_B="${GPU_B:-7}"

MAX_PAIRS_PER_DATASET="${MAX_PAIRS_PER_DATASET:-60}"
MAX_PAIRS="${MAX_PAIRS:-60}"
CAP_STRATEGY="${CAP_STRATEGY:-source_round_robin}"
BRANCHES_PER_COMPARISON="${BRANCHES_PER_COMPARISON:-3}"
SELECTED_LAYERS="${SELECTED_LAYERS:-4,8,12,16,20,24,28,32}"
TRAJECTORY_POINTS="${TRAJECTORY_POINTS:-17}"
SHARD_SIZE="${SHARD_SIZE:-32}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_NEW_TOKENS_THINKING="${MAX_NEW_TOKENS_THINKING:-192}"
MAX_NEW_TOKENS_DIRECT="${MAX_NEW_TOKENS_DIRECT:-16}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
USE_4BIT="${USE_4BIT:-0}"
COMPRESS_SHARDS="${COMPRESS_SHARDS:-1}"

PROBE_TARGETS="${PROBE_TARGETS:-final_choice,target_option,target_selected,condition_label,presentation_order}"
GROUP_COLUMNS="${GROUP_COLUMNS:-comparison_dimension,source_dataset,validity_type,difficulty_tier,analysis_split}"
CV_FOLDS="${CV_FOLDS:-5}"
PROBE_C="${PROBE_C:-0.1}"
MIN_PROBE_ROWS="${MIN_PROBE_ROWS:-40}"
MIN_DIMENSION_ROWS="${MIN_DIMENSION_ROWS:-40}"
COMMITMENT_AUC="${COMMITMENT_AUC:-0.75}"
COMMITMENT_PERSISTENCE="${COMMITMENT_PERSISTENCE:-3}"
CHOICE_CONFIDENCE_THRESHOLD="${CHOICE_CONFIDENCE_THRESHOLD:-0.8}"
TARGET_CONFIDENCE_THRESHOLD="${TARGET_CONFIDENCE_THRESHOLD:-0.8}"
SEED="${SEED:-1234}"

SKIP_MISSING="${SKIP_MISSING:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
ANALYZE_ONLY="${ANALYZE_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"

cd "$WORKDIR"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$WORKDIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$ARTROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$LOG_DIR"

echo "Running local Qwen3 judge-reasoning matrix"
echo "  run_tag=$RUN_TAG"
echo "  workdir=$WORKDIR"
echo "  artroot=$ARTROOT"
echo "  pair_out_dir=$PAIR_OUT_DIR"
echo "  out_root=$OUT_ROOT"
echo "  base=$BASE_LABEL :: $BASE_MODEL :: gpu $GPU_A"
echo "  instruct=$IT_LABEL :: $IT_MODEL :: gpu $GPU_B"
echo "  max_pairs=$MAX_PAIRS"
echo "  branches_per_comparison=$BRANCHES_PER_COMPARISON"
echo "  trajectory_points=$TRAJECTORY_POINTS"
echo "  analyze_only=$ANALYZE_ONLY"
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

trace_dir() {
  echo "$OUT_ROOT/trajectories_$1"
}

analysis_dir() {
  echo "$OUT_ROOT/analysis_$1"
}

build_pairs() {
  if [[ "$SKIP_EXISTING" == "1" && -s "$PAIR_OUT_DIR/manifest.json" && -s "$COMPARISONS_JSONL" ]]; then
    echo "skip existing pairs -> $COMPARISONS_JSONL"
    return 0
  fi
  args=(
    "$PYTHON" -m aisafety.scripts.build_judge_reasoning_suite
    --workspace-root "$WORKDIR"
    --input-root "$ARTROOT"
    --config "$CONFIG"
    --max-pairs-per-dataset "$MAX_PAIRS_PER_DATASET"
    --seed "$SEED"
    --out-dir "$PAIR_OUT_DIR"
  )
  if [[ "$SKIP_MISSING" == "1" ]]; then
    args+=(--skip-missing)
  fi
  run_logged build_pairs "${args[@]}"
}

run_trace() {
  local label="$1"
  local model="$2"
  local prompt_style="$3"
  local reasoning_modes="$4"
  local gpu="$5"
  local out_dir
  out_dir="$(trace_dir "$label")"

  if [[ "$SKIP_EXISTING" == "1" && -s "$out_dir/manifest.json" ]]; then
    echo "skip existing trace $label -> $out_dir"
    return 0
  fi
  mkdir -p "$out_dir"
  echo "launch trace $label gpu=$gpu model=$model"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    args=(
      "$PYTHON" -m aisafety.scripts.run_judge_reasoning_trajectories
      --workspace-root "$WORKDIR"
      --comparisons-jsonl "$COMPARISONS_JSONL"
      --run-label "$label"
      --model-id "$model"
      --cache-dir "$HF_HOME"
      --prompt-style "$prompt_style"
      --reasoning-modes "$reasoning_modes"
      --branches-per-comparison "$BRANCHES_PER_COMPARISON"
      --max-pairs "$MAX_PAIRS"
      --cap-strategy "$CAP_STRATEGY"
      --selected-layers "$SELECTED_LAYERS"
      --trajectory-points "$TRAJECTORY_POINTS"
      --shard-size "$SHARD_SIZE"
      --max-prompt-length "$MAX_PROMPT_LENGTH"
      --max-new-tokens-thinking "$MAX_NEW_TOKENS_THINKING"
      --max-new-tokens-direct "$MAX_NEW_TOKENS_DIRECT"
      --temperature "$TEMPERATURE"
      --top-p "$TOP_P"
      --top-k "$TOP_K"
      --seed "$SEED"
      --out-dir "$out_dir"
    )
    if [[ "$USE_4BIT" == "1" ]]; then
      args+=(--use-4bit)
    fi
    if [[ "$COMPRESS_SHARDS" == "1" ]]; then
      args+=(--compress-shards)
    fi
    if [[ -s "$out_dir/traces.jsonl" ]]; then
      args+=(--resume)
    fi
    "${args[@]}"
  ) >"$LOG_DIR/trace_${label}.out" 2>"$LOG_DIR/trace_${label}.err"
}

run_analysis() {
  local label="$1"
  local trace
  local out_dir
  trace="$(trace_dir "$label")"
  out_dir="$(analysis_dir "$label")"
  if [[ "$DRY_RUN" != "1" && ! -s "$trace/manifest.json" ]]; then
    echo "Missing completed trace manifest for $label: $trace/manifest.json" >&2
    return 1
  fi
  if [[ "$SKIP_EXISTING" == "1" && -s "$out_dir/manifest.json" ]]; then
    echo "skip existing analysis $label -> $out_dir"
    return 0
  fi
  run_logged "analysis_${label}" \
    "$PYTHON" -m aisafety.scripts.analyze_judge_reasoning_trajectories \
    --workspace-root "$WORKDIR" \
    --trace-dir "$trace" \
    --probe-targets "$PROBE_TARGETS" \
    --group-columns "$GROUP_COLUMNS" \
    --cv-folds "$CV_FOLDS" \
    --probe-c "$PROBE_C" \
    --min-probe-rows "$MIN_PROBE_ROWS" \
    --min-dimension-rows "$MIN_DIMENSION_ROWS" \
    --commitment-auc "$COMMITMENT_AUC" \
    --commitment-persistence "$COMMITMENT_PERSISTENCE" \
    --choice-confidence-threshold "$CHOICE_CONFIDENCE_THRESHOLD" \
    --target-confidence-threshold "$TARGET_CONFIDENCE_THRESHOLD" \
    --seed "$SEED" \
    --out-dir "$out_dir"
}

if [[ "$ANALYZE_ONLY" != "1" ]]; then
  build_pairs
  run_trace "$BASE_LABEL" "$BASE_MODEL" "$BASE_PROMPT_STYLE" "$BASE_REASONING_MODES" "$GPU_A" &
  base_pid="$!"
  run_trace "$IT_LABEL" "$IT_MODEL" "$IT_PROMPT_STYLE" "$IT_REASONING_MODES" "$GPU_B" &
  it_pid="$!"
  failed=0
  wait "$base_pid" || failed=1
  wait "$it_pid" || failed=1
  if [[ "$failed" != "0" ]]; then
    echo "At least one trace stage failed. Check $LOG_DIR" >&2
    exit 1
  fi
fi

run_analysis "$BASE_LABEL" &
base_analysis_pid="$!"
run_analysis "$IT_LABEL" &
it_analysis_pid="$!"
failed=0
wait "$base_analysis_pid" || failed=1
wait "$it_analysis_pid" || failed=1
if [[ "$failed" != "0" ]]; then
  echo "At least one analysis stage failed. Check $LOG_DIR" >&2
  exit 1
fi

summary_dir="$OUT_ROOT/summary"
run_logged summary \
  "$PYTHON" -m aisafety.scripts.summarize_judge_reasoning_suite \
  --workspace-root "$WORKDIR" \
  --analysis "$BASE_LABEL=$(analysis_dir "$BASE_LABEL")" \
  --analysis "$IT_LABEL=$(analysis_dir "$IT_LABEL")" \
  --out-dir "$summary_dir"

echo "COMPLETE"
echo "out_root=$OUT_ROOT"
echo "summary=$summary_dir"
echo "logs=$LOG_DIR"
