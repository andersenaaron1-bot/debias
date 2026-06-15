#!/bin/bash
# Run the matched long and computationally enforced CoT follow-up.

set -euo pipefail

WORKDIR="${WORKDIR:-$(pwd)}"
ARTROOT="${ARTROOT:-$WORKDIR}"
PYTHON="${PYTHON:-$WORKDIR/.venv/bin/python}"
RUN_TAG="${RUN_TAG:-judge_structured_cot_enforced_qwen3_8b_v1}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
RUN_LABEL="${RUN_LABEL:-qwen3_8b_it}"
GPU="${GPU:-7}"

SOURCE_TAG="${SOURCE_TAG:-judge_criterion_confirmation_qwen3_8b_v1}"
SOURCE_SUITE_DIR="${SOURCE_SUITE_DIR:-$ARTROOT/data/derived/helpsteer2_criterion_confirmation_$SOURCE_TAG}"
SUITE_DIR="${SUITE_DIR:-$ARTROOT/data/derived/helpsteer2_enforced_structure_$RUN_TAG}"
OUT_ROOT="${OUT_ROOT:-$ARTROOT/artifacts/mechanistic/$RUN_TAG}"
BEHAVIOR_DIR="${BEHAVIOR_DIR:-$OUT_ROOT/behavior}"
ANALYSIS_DIR="${ANALYSIS_DIR:-$OUT_ROOT/analysis}"
LOG_DIR="${LOG_DIR:-$OUT_ROOT/logs}"

BRANCHES="${BRANCHES:-1}"
ANALYSIS_TOKENS="${ANALYSIS_TOKENS:-1536}"
STAGE_TOKENS="${STAGE_TOKENS:-384}"
VERDICT_TOKENS="${VERDICT_TOKENS:-128}"
ANALYSIS_THINKING="${ANALYSIS_THINKING:-1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
MAX_SCORE_LENGTH="${MAX_SCORE_LENGTH:-12288}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-4}"
BOOTSTRAP="${BOOTSTRAP:-5000}"
USE_4BIT="${USE_4BIT:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
FORCE_SUITE_REBUILD="${FORCE_SUITE_REBUILD:-0}"
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
if [[ ! -s "$SOURCE_SUITE_DIR/manifest.json" \
      || ! -s "$SOURCE_SUITE_DIR/pairs.jsonl" ]]; then
  echo "Missing locked confirmation suite: $SOURCE_SUITE_DIR" >&2
  exit 1
fi

echo "Running Qwen3-8B enforced structured-CoT follow-up"
echo "  run_tag=$RUN_TAG"
echo "  model=$RUN_LABEL :: $MODEL_ID"
echo "  gpu=$GPU"
echo "  source_suite=$SOURCE_SUITE_DIR"
echo "  suite_dir=$SUITE_DIR"
echo "  behavior_dir=$BEHAVIOR_DIR"
echo "  conditions=free_long,prompted_long,enforced_generic,enforced_criterion"
echo "  planned_pairs=24"
echo "  planned_traces=192"
echo "  analysis_budget=$ANALYSIS_TOKENS"
echo "  staged_budget=4x$STAGE_TOKENS"
echo "  verdict_budget=$VERDICT_TOKENS"

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

if [[ "$FORCE_SUITE_REBUILD" == "1" \
      || ! -s "$SUITE_DIR/manifest.json" \
      || ! -s "$SUITE_DIR/episodes.jsonl" ]]; then
  run_logged build_suite \
    "$PYTHON" \
    -m aisafety.scripts.build_helpsteer2_enforced_structure_suite \
    --workspace-root "$WORKDIR" \
    --source-suite-dir "$SOURCE_SUITE_DIR" \
    --branches "$BRANCHES" \
    --seed "$SEED" \
    --out-dir "$SUITE_DIR"
else
  echo "skip existing enforced-structure suite -> $SUITE_DIR"
fi

if [[ "$SKIP_EXISTING" != "1" || ! -s "$BEHAVIOR_DIR/manifest.json" ]]; then
  behavior_args=(
    "$PYTHON" -m aisafety.scripts.run_judge_structured_cot_enforced
    --workspace-root "$WORKDIR"
    --episodes-jsonl "$SUITE_DIR/episodes.jsonl"
    --run-label "$RUN_LABEL"
    --model-id "$MODEL_ID"
    --cache-dir "$HF_HOME"
    --prompt-style chat_template
    --labels A,B,C
    --branches-per-episode "$BRANCHES"
    --analysis-tokens "$ANALYSIS_TOKENS"
    --stage-tokens "$STAGE_TOKENS"
    --verdict-tokens "$VERDICT_TOKENS"
    --max-prompt-length "$MAX_PROMPT_LENGTH"
    --max-score-length "$MAX_SCORE_LENGTH"
    --score-batch-size "$SCORE_BATCH_SIZE"
    --seed "$SEED"
    --out-dir "$BEHAVIOR_DIR"
  )
  if [[ "$ANALYSIS_THINKING" == "1" ]]; then
    behavior_args+=(--analysis-thinking)
  else
    behavior_args+=(--no-analysis-thinking)
  fi
  behavior_args+=(--no-verdict-thinking)
  if [[ "$USE_4BIT" == "1" ]]; then behavior_args+=(--use-4bit); fi
  if [[ -s "$BEHAVIOR_DIR/traces.jsonl" ]]; then
    behavior_args+=(--resume)
  fi
  echo "launch enforced structured-CoT behavior gpu=$GPU"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '  CUDA_VISIBLE_DEVICES=%q ' "$GPU"
    printf '%q ' "${behavior_args[@]}"
    printf '\n'
  else
    CUDA_VISIBLE_DEVICES="$GPU" "${behavior_args[@]}" \
      >"$LOG_DIR/behavior.out" 2>"$LOG_DIR/behavior.err"
  fi
else
  echo "skip existing behavior -> $BEHAVIOR_DIR"
fi

if [[ "$SKIP_EXISTING" != "1" || ! -s "$ANALYSIS_DIR/manifest.json" ]]; then
  analysis_args=(
    "$PYTHON" -m aisafety.scripts.analyze_judge_structured_cot_enforced
    --workspace-root "$WORKDIR"
    --run-dir "$BEHAVIOR_DIR"
    --suite-dir "$SUITE_DIR"
    --source-suite-dir "$SOURCE_SUITE_DIR"
    --bootstrap "$BOOTSTRAP"
    --seed "$SEED"
    --out-dir "$ANALYSIS_DIR"
  )
  audit_csv="$SOURCE_SUITE_DIR/human_audit/audit_prompts_for_judging_completed.csv"
  if [[ -s "$audit_csv" ]]; then
    analysis_args+=(--audit-csv "$audit_csv")
  fi
  run_logged analysis "${analysis_args[@]}"
else
  echo "skip existing analysis -> $ANALYSIS_DIR"
fi

if [[ "$DRY_RUN" != "1" ]]; then
  "$PYTHON" -m aisafety.scripts.read_judge_structured_cot_enforced \
    --workspace-root "$WORKDIR" \
    --analysis-dir "$ANALYSIS_DIR" \
    >"$OUT_ROOT/readout.txt"
  cat "$OUT_ROOT/readout.txt"
fi

echo "COMPLETE"
echo "out_root=$OUT_ROOT"
echo "suite_dir=$SUITE_DIR"
echo "behavior_dir=$BEHAVIOR_DIR"
echo "analysis_dir=$ANALYSIS_DIR"
