#!/bin/bash
# Run the held-out Qwen3-8B criterion-switch behavioral scout on ipe-monster.

set -euo pipefail

WORKDIR="${WORKDIR:-$(pwd)}"
ARTROOT="${ARTROOT:-$WORKDIR}"
PYTHON="${PYTHON:-$WORKDIR/.venv/bin/python}"
RUN_TAG="${RUN_TAG:-judge_criterion_switch_qwen3_8b_scout_v1}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
RUN_LABEL="${RUN_LABEL:-qwen3_8b_it}"
GPU_0="${GPU_0:-0}"
GPU_1="${GPU_1:-1}"

SUITE_DIR="${SUITE_DIR:-$ARTROOT/data/derived/helpsteer2_criterion_switch_suite_$RUN_TAG}"
OUT_ROOT="${OUT_ROOT:-$ARTROOT/artifacts/mechanistic/$RUN_TAG}"
BEHAVIOR_0="${BEHAVIOR_0:-$OUT_ROOT/behavior_shard_0}"
BEHAVIOR_1="${BEHAVIOR_1:-$OUT_ROOT/behavior_shard_1}"
ANALYSIS_DIR="${ANALYSIS_DIR:-$OUT_ROOT/behavior_analysis}"
LOG_DIR="${LOG_DIR:-$OUT_ROOT/logs}"
DEFAULT_EXCLUDE="$ARTROOT/data/derived/helpsteer2_matched_criterion_suite_helpsteer2_matched_criterion_qwen3_8b_scout_v1/pairs.jsonl"
EXCLUDE_PAIRS_JSONL="${EXCLUDE_PAIRS_JSONL:-$DEFAULT_EXCLUDE}"

MAX_PAIRS_PER_TRANSITION="${MAX_PAIRS_PER_TRANSITION:-8}"
MIN_PAIRS_PER_TRANSITION="${MIN_PAIRS_PER_TRANSITION:-6}"
MIN_CHOICE_GAP="${MIN_CHOICE_GAP:-1.0}"
BRANCHES_PER_EPISODE="${BRANCHES_PER_EPISODE:-3}"
PHASE1_TOKENS="${PHASE1_TOKENS:-128}"
PHASE2_TOKENS="${PHASE2_TOKENS:-384}"
SCORE_BATCH_SIZE_GPU_0="${SCORE_BATCH_SIZE_GPU_0:-2}"
SCORE_BATCH_SIZE_GPU_1="${SCORE_BATCH_SIZE_GPU_1:-4}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_SCORE_LENGTH="${MAX_SCORE_LENGTH:-8192}"
USE_4BIT="${USE_4BIT:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
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

planned_pairs=$((3 * MAX_PAIRS_PER_TRANSITION))
planned_episodes=$((planned_pairs * 5 * 2))
planned_traces=$((planned_episodes * BRANCHES_PER_EPISODE))
echo "Running Qwen3-8B criterion-switch behavioral scout"
echo "  run_tag=$RUN_TAG"
echo "  model=$RUN_LABEL :: $MODEL_ID"
echo "  gpus=$GPU_0,$GPU_1"
echo "  suite_dir=$SUITE_DIR"
echo "  out_root=$OUT_ROOT"
echo "  pairs_per_transition=$MAX_PAIRS_PER_TRANSITION"
echo "  phase_tokens=$PHASE1_TOKENS+$PHASE2_TOKENS"
echo "  branches_per_episode=$BRANCHES_PER_EPISODE"
echo "  planned_upper_bound=pairs:$planned_pairs episodes:$planned_episodes traces:$planned_traces"

run_logged() {
  local name="$1"
  shift
  echo "run $name"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '  '; printf '%q ' "$@"; printf '\n'
    return 0
  fi
  "$@" >"$LOG_DIR/${name}.out" 2>"$LOG_DIR/${name}.err"
}

build_suite() {
  if [[ "$FORCE_SUITE_REBUILD" != "1" \
        && -s "$SUITE_DIR/manifest.json" \
        && -s "$SUITE_DIR/episodes_shard_0.jsonl" \
        && -s "$SUITE_DIR/episodes_shard_1.jsonl" ]]; then
    echo "skip existing criterion-switch suite -> $SUITE_DIR"
    return 0
  fi
  local args=(
    "$PYTHON" -m aisafety.scripts.build_helpsteer2_criterion_switch_suite
    --workspace-root "$WORKDIR"
    --artifact-root "$ARTROOT"
    --cache-dir "$HF_DATASETS_CACHE"
    --max-pairs-per-transition "$MAX_PAIRS_PER_TRANSITION"
    --min-pairs-per-transition "$MIN_PAIRS_PER_TRANSITION"
    --min-choice-gap "$MIN_CHOICE_GAP"
    --num-shards 2
    --seed "$SEED"
    --out-dir "$SUITE_DIR"
  )
  if [[ -s "$EXCLUDE_PAIRS_JSONL" ]]; then
    args+=(--exclude-pairs-jsonl "$EXCLUDE_PAIRS_JSONL")
  fi
  run_logged build_suite "${args[@]}"
}

run_shard() {
  local shard_index="$1"
  local gpu="$2"
  local score_batch_size="$3"
  local episode_file="$SUITE_DIR/episodes_shard_${shard_index}.jsonl"
  local shard_dir="$OUT_ROOT/behavior_shard_${shard_index}"
  if [[ "$SKIP_EXISTING" == "1" && -s "$shard_dir/manifest.json" ]]; then
    echo "skip existing behavior shard $shard_index -> $shard_dir"
    return 0
  fi
  local args=(
    "$PYTHON" -m aisafety.scripts.run_judge_criterion_switch_behavior
    --workspace-root "$WORKDIR"
    --episodes-jsonl "$episode_file"
    --run-label "$RUN_LABEL"
    --model-id "$MODEL_ID"
    --cache-dir "$HF_HOME"
    --prompt-style chat_template
    --labels A,B,C
    --branches-per-episode "$BRANCHES_PER_EPISODE"
    --phase1-tokens "$PHASE1_TOKENS"
    --phase2-tokens "$PHASE2_TOKENS"
    --max-prompt-length "$MAX_PROMPT_LENGTH"
    --max-score-length "$MAX_SCORE_LENGTH"
    --score-batch-size "$score_batch_size"
    --seed "$SEED"
    --out-dir "$shard_dir"
  )
  if [[ "$USE_4BIT" == "1" ]]; then args+=(--use-4bit); fi
  if [[ -s "$shard_dir/switch_traces.jsonl" ]]; then args+=(--resume); fi
  echo "launch behavior shard=$shard_index gpu=$gpu"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '  CUDA_VISIBLE_DEVICES=%q ' "$gpu"
    printf '%q ' "${args[@]}"
    printf '\n'
    return 0
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "${args[@]}" \
    >"$LOG_DIR/behavior_shard_${shard_index}.out" \
    2>"$LOG_DIR/behavior_shard_${shard_index}.err"
}

run_analysis() {
  if [[ "$SKIP_EXISTING" == "1" && -s "$ANALYSIS_DIR/manifest.json" ]]; then
    echo "skip existing behavior analysis -> $ANALYSIS_DIR"
    return 0
  fi
  run_logged behavior_analysis \
    "$PYTHON" -m aisafety.scripts.analyze_judge_criterion_switch_behavior \
    --workspace-root "$WORKDIR" \
    --run-dir "$BEHAVIOR_0" \
    --run-dir "$BEHAVIOR_1" \
    --out-dir "$ANALYSIS_DIR"
}

if [[ "$ANALYZE_ONLY" != "1" ]]; then
  build_suite
  if [[ "$DRY_RUN" == "1" ]]; then
    run_shard 0 "$GPU_0" "$SCORE_BATCH_SIZE_GPU_0"
    run_shard 1 "$GPU_1" "$SCORE_BATCH_SIZE_GPU_1"
  else
    run_shard 0 "$GPU_0" "$SCORE_BATCH_SIZE_GPU_0" &
    pid_0=$!
    run_shard 1 "$GPU_1" "$SCORE_BATCH_SIZE_GPU_1" &
    pid_1=$!
    status=0
    wait "$pid_0" || status=1
    wait "$pid_1" || status=1
    if [[ "$status" != "0" ]]; then
      echo "At least one behavior shard failed. Check $LOG_DIR." >&2
      exit 1
    fi
  fi
fi
run_analysis
echo "COMPLETE"
echo "out_root=$OUT_ROOT"
echo "analysis_dir=$ANALYSIS_DIR"
