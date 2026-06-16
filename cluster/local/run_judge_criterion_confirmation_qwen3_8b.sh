#!/bin/bash
# Build and run the locked HelpSteer2 criterion confirmation on ipe-monster.

set -euo pipefail

WORKDIR="${WORKDIR:-$(pwd)}"
ARTROOT="${ARTROOT:-$WORKDIR}"
PYTHON="${PYTHON:-$WORKDIR/.venv/bin/python}"
RUN_TAG="${RUN_TAG:-judge_criterion_confirmation_qwen3_8b_v1}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
RUN_LABEL="${RUN_LABEL:-qwen3_8b_it}"
SOURCE_SPLIT="${SOURCE_SPLIT:-train}"
GPU="${GPU:-7}"

SUITE_DIR="${SUITE_DIR:-$ARTROOT/data/derived/helpsteer2_criterion_confirmation_$RUN_TAG}"
OUT_ROOT="${OUT_ROOT:-$ARTROOT/artifacts/mechanistic/$RUN_TAG}"
BEHAVIOR_DIR="${BEHAVIOR_DIR:-$OUT_ROOT/behavior}"
LOG_DIR="${LOG_DIR:-$OUT_ROOT/logs}"

CANDIDATE_POOL_PER_TRANSITION="${CANDIDATE_POOL_PER_TRANSITION:-24}"
MIN_CHOICE_GAP="${MIN_CHOICE_GAP:-1.0}"
MAIN_BRANCHES="${MAIN_BRANCHES:-2}"
CEILING_PAIRS_PER_CONFLICT="${CEILING_PAIRS_PER_CONFLICT:-6}"
PHASE1_TOKENS="${PHASE1_TOKENS:-128}"
PHASE2_TOKENS="${PHASE2_TOKENS:-384}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-4}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_SCORE_LENGTH="${MAX_SCORE_LENGTH:-8192}"
USE_4BIT="${USE_4BIT:-0}"
BUILD_ONLY="${BUILD_ONLY:-0}"
RUN_ONLY="${RUN_ONLY:-0}"
FORCE_SUITE_REBUILD="${FORCE_SUITE_REBUILD:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
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
if [[ "$BUILD_ONLY" == "1" && "$RUN_ONLY" == "1" ]]; then
  echo "BUILD_ONLY and RUN_ONLY cannot both be 1." >&2
  exit 1
fi

echo "Running locked criterion confirmation"
echo "  run_tag=$RUN_TAG"
echo "  model=$RUN_LABEL :: $MODEL_ID"
echo "  gpu=$GPU"
echo "  suite_dir=$SUITE_DIR"
echo "  audit_prompt_dir=$SUITE_DIR/human_audit/prompts"
echo "  behavior_dir=$BEHAVIOR_DIR"
echo "  planned_pairs=24"
echo "  planned_episodes=216"
echo "  planned_traces=408"

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
        && -s "$SUITE_DIR/episodes.jsonl" \
        && -d "$SUITE_DIR/human_audit/prompts" ]]; then
    echo "skip existing locked confirmation suite -> $SUITE_DIR"
    return 0
  fi
  local args=(
    "$PYTHON" -m aisafety.scripts.build_helpsteer2_criterion_confirmation
    --workspace-root "$WORKDIR"
    --artifact-root "$ARTROOT"
    --cache-dir "$HF_DATASETS_CACHE"
    --split "$SOURCE_SPLIT"
    --candidate-pool-per-transition "$CANDIDATE_POOL_PER_TRANSITION"
    --min-choice-gap "$MIN_CHOICE_GAP"
    --main-branches "$MAIN_BRANCHES"
    --ceiling-pairs-per-conflict-transition "$CEILING_PAIRS_PER_CONFLICT"
    --seed "$SEED"
    --out-dir "$SUITE_DIR"
  )
  local exclusions=(
    "$ARTROOT/data/derived/helpsteer2_matched_criterion_suite_helpsteer2_matched_criterion_qwen3_8b_scout_v1/pairs.jsonl"
    "$ARTROOT/data/derived/helpsteer2_criterion_switch_suite_judge_criterion_switch_qwen3_8b_scout_v1/pairs.jsonl"
    "$ARTROOT/data/derived/helpsteer2_criterion_switch_suite_judge_criterion_switch_qwen3_8b_extension_v1/pairs.jsonl"
  )
  local path
  for path in "${exclusions[@]}"; do
    if [[ -s "$path" ]]; then
      args+=(--exclude-pairs-jsonl "$path")
    fi
  done
  run_logged build_confirmation_suite "${args[@]}"
}

run_behavior() {
  if [[ ! -s "$SUITE_DIR/manifest.json" || ! -s "$SUITE_DIR/episodes.jsonl" ]]; then
    echo "Locked suite is missing. Run with BUILD_ONLY=1 first." >&2
    exit 1
  fi
  if [[ "$SKIP_EXISTING" == "1" && -s "$BEHAVIOR_DIR/manifest.json" ]]; then
    echo "skip existing confirmation behavior -> $BEHAVIOR_DIR"
    return 0
  fi
  local args=(
    "$PYTHON" -m aisafety.scripts.run_judge_criterion_switch_behavior
    --workspace-root "$WORKDIR"
    --episodes-jsonl "$SUITE_DIR/episodes.jsonl"
    --run-label "$RUN_LABEL"
    --model-id "$MODEL_ID"
    --cache-dir "$HF_HOME"
    --prompt-style chat_template
    --labels A,B,C
    --branches-per-episode "$MAIN_BRANCHES"
    --phase1-tokens "$PHASE1_TOKENS"
    --phase2-tokens "$PHASE2_TOKENS"
    --max-prompt-length "$MAX_PROMPT_LENGTH"
    --max-score-length "$MAX_SCORE_LENGTH"
    --score-batch-size "$SCORE_BATCH_SIZE"
    --skip-direct
    --seed "$SEED"
    --out-dir "$BEHAVIOR_DIR"
  )
  if [[ "$USE_4BIT" == "1" ]]; then args+=(--use-4bit); fi
  if [[ -s "$BEHAVIOR_DIR/switch_traces.jsonl" ]]; then args+=(--resume); fi
  echo "launch confirmation behavior gpu=$GPU"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '  CUDA_VISIBLE_DEVICES=%q ' "$GPU"
    printf '%q ' "${args[@]}"
    printf '\n'
    return 0
  fi
  CUDA_VISIBLE_DEVICES="$GPU" "${args[@]}" \
    >"$LOG_DIR/behavior.out" \
    2>"$LOG_DIR/behavior.err"
}

if [[ "$RUN_ONLY" != "1" ]]; then
  build_suite
fi
if [[ "$BUILD_ONLY" != "1" ]]; then
  run_behavior
fi

echo "COMPLETE"
echo "suite_dir=$SUITE_DIR"
echo "audit_prompt_dir=$SUITE_DIR/human_audit/prompts"
echo "audit_response_csv=$SUITE_DIR/human_audit/audit_responses.csv"
echo "behavior_dir=$BEHAVIOR_DIR"
