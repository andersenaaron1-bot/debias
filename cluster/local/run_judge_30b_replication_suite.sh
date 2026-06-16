#!/bin/bash
# Run the locked 30B-scale judge-deliberation replication suite on ipe-monster.

set -euo pipefail

WORKDIR="${WORKDIR:-$(pwd)}"
ARTROOT="${ARTROOT:-$WORKDIR}"
PYTHON="${PYTHON:-$WORKDIR/.venv/bin/python}"
RUN_TAG="${RUN_TAG:-judge_30b_replication_qwen3_30b_a3b_v1}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-30B-A3B}"
RUN_LABEL="${RUN_LABEL:-qwen3_30b_a3b_it}"
GPU="${GPU:-7}"

OUT_ROOT="${OUT_ROOT:-$ARTROOT/artifacts/mechanistic/$RUN_TAG}"
LOG_DIR="${LOG_DIR:-$OUT_ROOT/logs}"

HELPSTEER_SOURCE_TAG="${HELPSTEER_SOURCE_TAG:-judge_criterion_confirmation_qwen3_8b_v1}"
HELPSTEER_SUITE_DIR="${HELPSTEER_SUITE_DIR:-$ARTROOT/data/derived/helpsteer2_criterion_confirmation_$HELPSTEER_SOURCE_TAG}"
HELPSTEER_RUN_TAG="${HELPSTEER_RUN_TAG:-${RUN_TAG}_helpsteer_confirmation}"
HELPSTEER_OUT_ROOT="${HELPSTEER_OUT_ROOT:-$OUT_ROOT/helpsteer_confirmation}"
HELPSTEER_BEHAVIOR_DIR="${HELPSTEER_BEHAVIOR_DIR:-$HELPSTEER_OUT_ROOT/behavior}"
HELPSTEER_ANALYSIS_DIR="${HELPSTEER_ANALYSIS_DIR:-$HELPSTEER_OUT_ROOT/analysis}"

SUMMEVAL_SOURCE_TAG="${SUMMEVAL_SOURCE_TAG:-judge_summeval_structured_cot_qwen3_8b_v1}"
SUMMEVAL_SUITE_DIR="${SUMMEVAL_SUITE_DIR:-$ARTROOT/data/derived/summeval_criterion_suite_$SUMMEVAL_SOURCE_TAG}"
SUMMEVAL_RUN_TAG="${SUMMEVAL_RUN_TAG:-${RUN_TAG}_summeval}"
SUMMEVAL_OUT_ROOT="${SUMMEVAL_OUT_ROOT:-$OUT_ROOT/summeval}"
SUMMEVAL_BEHAVIOR_DIR="${SUMMEVAL_BEHAVIOR_DIR:-$SUMMEVAL_OUT_ROOT/behavior}"
SUMMEVAL_ANALYSIS_DIR="${SUMMEVAL_ANALYSIS_DIR:-$SUMMEVAL_OUT_ROOT/analysis}"
SUMMEVAL_REPLAY_DIR="${SUMMEVAL_REPLAY_DIR:-$SUMMEVAL_OUT_ROOT/rationale_replay}"
SUMMEVAL_ACTIVATION_DIR="${SUMMEVAL_ACTIVATION_DIR:-$SUMMEVAL_OUT_ROOT/activations}"
SUMMEVAL_ACTIVATION_ANALYSIS_DIR="${SUMMEVAL_ACTIVATION_ANALYSIS_DIR:-$SUMMEVAL_OUT_ROOT/activation_analysis}"

RUN_HELPSTEER="${RUN_HELPSTEER:-1}"
RUN_SUMMEVAL="${RUN_SUMMEVAL:-1}"
RUN_SUMMEVAL_REPLAY="${RUN_SUMMEVAL_REPLAY:-1}"
RUN_SUMMEVAL_ACTIVATIONS="${RUN_SUMMEVAL_ACTIVATIONS:-0}"

MAIN_BRANCHES="${MAIN_BRANCHES:-2}"
CEILING_BRANCHES="${CEILING_BRANCHES:-1}"
PHASE1_TOKENS="${PHASE1_TOKENS:-128}"
PHASE2_TOKENS="${PHASE2_TOKENS:-384}"
DIRECT_TOKENS="${DIRECT_TOKENS:-16}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-6144}"
MAX_SCORE_LENGTH="${MAX_SCORE_LENGTH:-8192}"
BOOTSTRAP="${BOOTSTRAP:-5000}"
USE_4BIT="${USE_4BIT:-1}"
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

echo "Running 30B judge-deliberation replication suite"
echo "  run_tag=$RUN_TAG"
echo "  model=$RUN_LABEL :: $MODEL_ID"
echo "  gpu=$GPU"
echo "  use_4bit=$USE_4BIT score_batch_size=$SCORE_BATCH_SIZE"
echo "  out_root=$OUT_ROOT"
echo "  helpsteer_suite=$HELPSTEER_SUITE_DIR"
echo "  summeval_suite=$SUMMEVAL_SUITE_DIR"
echo "  run_helpsteer=$RUN_HELPSTEER"
echo "  run_summeval=$RUN_SUMMEVAL"
echo "  run_summeval_replay=$RUN_SUMMEVAL_REPLAY"
echo "  run_summeval_activations=$RUN_SUMMEVAL_ACTIVATIONS"

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
  if "$@" >"$LOG_DIR/${name}.out" 2>"$LOG_DIR/${name}.err"; then
    return 0
  fi
  local status=$?
  echo "FAILED $name status=$status" >&2
  echo "--- $LOG_DIR/${name}.out tail ---" >&2
  tail -n 80 "$LOG_DIR/${name}.out" >&2 || true
  echo "--- $LOG_DIR/${name}.err tail ---" >&2
  tail -n 80 "$LOG_DIR/${name}.err" >&2 || true
  exit "$status"
}

require_suite() {
  local name="$1"
  local suite_dir="$2"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  if [[ ! -s "$suite_dir/manifest.json" || ! -s "$suite_dir/episodes.jsonl" ]]; then
    echo "Missing locked $name suite: $suite_dir" >&2
    echo "This replication runner expects the frozen 8B suite, not newly sampled pairs." >&2
    exit 1
  fi
}

select_helpsteer_audit_csv() {
  local candidates=(
    "$HELPSTEER_SUITE_DIR/human_audit/audit_prompts_for_judging_completed.csv"
    "$HELPSTEER_SUITE_DIR/human_audit/audit_responses.csv"
  )
  local path
  for path in "${candidates[@]}"; do
    if [[ -s "$path" ]]; then
      printf '%s\n' "$path"
      return 0
    fi
  done
}

if [[ "$RUN_HELPSTEER" == "1" ]]; then
  require_suite "HelpSteer2 confirmation" "$HELPSTEER_SUITE_DIR"
  run_logged helpsteer_confirmation_behavior \
    env \
    WORKDIR="$WORKDIR" \
    ARTROOT="$ARTROOT" \
    PYTHON="$PYTHON" \
    RUN_TAG="$HELPSTEER_RUN_TAG" \
    MODEL_ID="$MODEL_ID" \
    RUN_LABEL="$RUN_LABEL" \
    GPU="$GPU" \
    SUITE_DIR="$HELPSTEER_SUITE_DIR" \
    OUT_ROOT="$HELPSTEER_OUT_ROOT" \
    BEHAVIOR_DIR="$HELPSTEER_BEHAVIOR_DIR" \
    LOG_DIR="$HELPSTEER_OUT_ROOT/logs" \
    RUN_ONLY=1 \
    SKIP_EXISTING="$SKIP_EXISTING" \
    FORCE_SUITE_REBUILD=0 \
    MAIN_BRANCHES="$MAIN_BRANCHES" \
    PHASE1_TOKENS="$PHASE1_TOKENS" \
    PHASE2_TOKENS="$PHASE2_TOKENS" \
    SCORE_BATCH_SIZE="$SCORE_BATCH_SIZE" \
    MAX_PROMPT_LENGTH="$MAX_PROMPT_LENGTH" \
    MAX_SCORE_LENGTH="$MAX_SCORE_LENGTH" \
    USE_4BIT="$USE_4BIT" \
    SEED="$SEED" \
    bash "$WORKDIR/cluster/local/run_judge_criterion_confirmation_qwen3_8b.sh"

  if [[ "$SKIP_EXISTING" != "1" || ! -s "$HELPSTEER_ANALYSIS_DIR/manifest.json" ]]; then
    helpsteer_analysis_args=(
      "$PYTHON" -m aisafety.scripts.analyze_judge_criterion_confirmation
      --workspace-root "$WORKDIR"
      --run-dir "$HELPSTEER_BEHAVIOR_DIR"
      --suite-dir "$HELPSTEER_SUITE_DIR"
      --endpoint-budget "$PHASE2_TOKENS"
      --bootstrap "$BOOTSTRAP"
      --seed "$SEED"
      --out-dir "$HELPSTEER_ANALYSIS_DIR"
    )
    audit_csv="$(select_helpsteer_audit_csv || true)"
    if [[ -n "${audit_csv:-}" ]]; then
      helpsteer_analysis_args+=(--audit-csv "$audit_csv")
    fi
    run_logged helpsteer_confirmation_analysis "${helpsteer_analysis_args[@]}"
  else
    echo "skip existing HelpSteer2 confirmation analysis -> $HELPSTEER_ANALYSIS_DIR"
  fi

  if [[ "$DRY_RUN" != "1" ]]; then
    "$PYTHON" -m aisafety.scripts.read_judge_criterion_confirmation \
      --workspace-root "$WORKDIR" \
      --analysis-dir "$HELPSTEER_ANALYSIS_DIR" \
      >"$HELPSTEER_OUT_ROOT/readout.txt"
  fi
fi

if [[ "$RUN_SUMMEVAL" == "1" ]]; then
  require_suite "SummEval" "$SUMMEVAL_SUITE_DIR"
  run_logged summeval_structured_cot_and_replay \
    env \
    WORKDIR="$WORKDIR" \
    ARTROOT="$ARTROOT" \
    PYTHON="$PYTHON" \
    RUN_TAG="$SUMMEVAL_RUN_TAG" \
    MODEL_ID="$MODEL_ID" \
    RUN_LABEL="$RUN_LABEL" \
    GPU="$GPU" \
    SUITE_DIR="$SUMMEVAL_SUITE_DIR" \
    OUT_ROOT="$SUMMEVAL_OUT_ROOT" \
    BEHAVIOR_DIR="$SUMMEVAL_BEHAVIOR_DIR" \
    ANALYSIS_DIR="$SUMMEVAL_ANALYSIS_DIR" \
    REPLAY_DIR="$SUMMEVAL_REPLAY_DIR" \
    ACTIVATION_DIR="$SUMMEVAL_ACTIVATION_DIR" \
    ACTIVATION_ANALYSIS_DIR="$SUMMEVAL_ACTIVATION_ANALYSIS_DIR" \
    LOG_DIR="$SUMMEVAL_OUT_ROOT/logs" \
    SKIP_EXISTING="$SKIP_EXISTING" \
    FORCE_SUITE_REBUILD=0 \
    RUN_REPLAY="$RUN_SUMMEVAL_REPLAY" \
    RUN_ACTIVATIONS="$RUN_SUMMEVAL_ACTIVATIONS" \
    MAIN_BRANCHES="$MAIN_BRANCHES" \
    CEILING_BRANCHES="$CEILING_BRANCHES" \
    PHASE1_TOKENS="$PHASE1_TOKENS" \
    PHASE2_TOKENS="$PHASE2_TOKENS" \
    DIRECT_TOKENS="$DIRECT_TOKENS" \
    SCORE_BATCH_SIZE="$SCORE_BATCH_SIZE" \
    MAX_PROMPT_LENGTH="$MAX_PROMPT_LENGTH" \
    MAX_SCORE_LENGTH="$MAX_SCORE_LENGTH" \
    BOOTSTRAP="$BOOTSTRAP" \
    USE_4BIT="$USE_4BIT" \
    SEED="$SEED" \
    bash "$WORKDIR/cluster/local/run_summeval_structured_cot_qwen3_8b_overnight.sh"
fi

if [[ "$DRY_RUN" != "1" ]]; then
  {
    echo "=== 30B REPLICATION SUITE ==="
    echo "run_tag=$RUN_TAG"
    echo "model=$RUN_LABEL :: $MODEL_ID"
    echo
    if [[ -s "$HELPSTEER_OUT_ROOT/readout.txt" ]]; then
      echo "=== HELPSTEER2 CONFIRMATION ==="
      cat "$HELPSTEER_OUT_ROOT/readout.txt"
      echo
    fi
    if [[ -s "$SUMMEVAL_OUT_ROOT/readout.txt" ]]; then
      echo "=== SUMMEVAL STRUCTURED COT ==="
      cat "$SUMMEVAL_OUT_ROOT/readout.txt"
      echo
    fi
    if [[ -s "$SUMMEVAL_OUT_ROOT/rationale_replay_readout.txt" ]]; then
      echo "=== SUMMEVAL RATIONALE REPLAY ==="
      cat "$SUMMEVAL_OUT_ROOT/rationale_replay_readout.txt"
      echo
    fi
  } >"$OUT_ROOT/readout.txt"
  cat "$OUT_ROOT/readout.txt"
fi

echo "COMPLETE"
echo "out_root=$OUT_ROOT"
echo "helpsteer_suite_dir=$HELPSTEER_SUITE_DIR"
echo "helpsteer_behavior_dir=$HELPSTEER_BEHAVIOR_DIR"
echo "helpsteer_analysis_dir=$HELPSTEER_ANALYSIS_DIR"
echo "summeval_suite_dir=$SUMMEVAL_SUITE_DIR"
echo "summeval_behavior_dir=$SUMMEVAL_BEHAVIOR_DIR"
echo "summeval_analysis_dir=$SUMMEVAL_ANALYSIS_DIR"
if [[ "$RUN_SUMMEVAL_REPLAY" == "1" ]]; then
  echo "summeval_replay_dir=$SUMMEVAL_REPLAY_DIR"
fi
if [[ "$RUN_SUMMEVAL_ACTIVATIONS" == "1" ]]; then
  echo "summeval_activation_dir=$SUMMEVAL_ACTIVATION_DIR"
  echo "summeval_activation_analysis_dir=$SUMMEVAL_ACTIVATION_ANALYSIS_DIR"
fi
