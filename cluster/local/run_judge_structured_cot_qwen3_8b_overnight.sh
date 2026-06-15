#!/bin/bash
# Run the matched HelpSteer2 structured-CoT overnight suite on ipe-monster.

set -euo pipefail

WORKDIR="${WORKDIR:-$(pwd)}"
ARTROOT="${ARTROOT:-$WORKDIR}"
PYTHON="${PYTHON:-$WORKDIR/.venv/bin/python}"
RUN_TAG="${RUN_TAG:-judge_structured_cot_qwen3_8b_v1}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
RUN_LABEL="${RUN_LABEL:-qwen3_8b_it}"
GPU="${GPU:-7}"

SOURCE_TAG="${SOURCE_TAG:-judge_criterion_confirmation_qwen3_8b_v1}"
SOURCE_SUITE_DIR="${SOURCE_SUITE_DIR:-$ARTROOT/data/derived/helpsteer2_criterion_confirmation_$SOURCE_TAG}"
SUITE_DIR="${SUITE_DIR:-$ARTROOT/data/derived/helpsteer2_structured_cot_$RUN_TAG}"
OUT_ROOT="${OUT_ROOT:-$ARTROOT/artifacts/mechanistic/$RUN_TAG}"
BEHAVIOR_DIR="${BEHAVIOR_DIR:-$OUT_ROOT/behavior}"
ANALYSIS_DIR="${ANALYSIS_DIR:-$OUT_ROOT/analysis}"
ACTIVATION_DIR="${ACTIVATION_DIR:-$OUT_ROOT/activations}"
ACTIVATION_ANALYSIS_DIR="${ACTIVATION_ANALYSIS_DIR:-$OUT_ROOT/activation_analysis}"
LOG_DIR="${LOG_DIR:-$OUT_ROOT/logs}"

MAIN_BRANCHES="${MAIN_BRANCHES:-2}"
CEILING_BRANCHES="${CEILING_BRANCHES:-1}"
PHASE1_TOKENS="${PHASE1_TOKENS:-128}"
PHASE2_TOKENS="${PHASE2_TOKENS:-384}"
DIRECT_TOKENS="${DIRECT_TOKENS:-16}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-4}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_SCORE_LENGTH="${MAX_SCORE_LENGTH:-8192}"
BOOTSTRAP="${BOOTSTRAP:-5000}"
RUN_ACTIVATIONS="${RUN_ACTIVATIONS:-1}"
ACTIVATION_CONDITIONS="${ACTIVATION_CONDITIONS:-free_cot,generic_scaffold,criterion_scaffold,score_evidence}"
ACTIVATION_BRANCHES="${ACTIVATION_BRANCHES:-0}"
SELECTED_LAYERS="${SELECTED_LAYERS:-4,8,12,16,20,24,28,32}"
TARGET_LAYERS="${TARGET_LAYERS:-active_criterion:20,criterion_target:32,current_choice:28,final_choice:32,presentation_order:12}"
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

echo "Running Qwen3-8B structured-CoT overnight suite"
echo "  run_tag=$RUN_TAG"
echo "  model=$RUN_LABEL :: $MODEL_ID"
echo "  gpu=$GPU"
echo "  source_suite=$SOURCE_SUITE_DIR"
echo "  suite_dir=$SUITE_DIR"
echo "  behavior_dir=$BEHAVIOR_DIR"
echo "  conditions=free_cot,generic_scaffold,criterion_scaffold,score_evidence,explicit_target"
echo "  planned_pairs=24"
echo "  planned_cot_traces=408"
echo "  planned_direct_rows=48"
echo "  planned_activation_traces=192"

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
    "$PYTHON" -m aisafety.scripts.build_helpsteer2_structured_cot_suite \
    --workspace-root "$WORKDIR" \
    --source-suite-dir "$SOURCE_SUITE_DIR" \
    --main-branches "$MAIN_BRANCHES" \
    --ceiling-branches "$CEILING_BRANCHES" \
    --seed "$SEED" \
    --out-dir "$SUITE_DIR"
else
  echo "skip existing structured-CoT suite -> $SUITE_DIR"
fi

if [[ "$SKIP_EXISTING" != "1" || ! -s "$BEHAVIOR_DIR/manifest.json" ]]; then
  behavior_args=(
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
    --max-new-tokens-direct "$DIRECT_TOKENS"
    --max-prompt-length "$MAX_PROMPT_LENGTH"
    --max-score-length "$MAX_SCORE_LENGTH"
    --score-batch-size "$SCORE_BATCH_SIZE"
    --seed "$SEED"
    --out-dir "$BEHAVIOR_DIR"
  )
  if [[ "$USE_4BIT" == "1" ]]; then behavior_args+=(--use-4bit); fi
  if [[ -s "$BEHAVIOR_DIR/switch_traces.jsonl" ]]; then
    behavior_args+=(--resume)
  fi
  echo "launch structured-CoT behavior gpu=$GPU"
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
    "$PYTHON" -m aisafety.scripts.analyze_judge_structured_cot
    --workspace-root "$WORKDIR"
    --run-dir "$BEHAVIOR_DIR"
    --suite-dir "$SUITE_DIR"
    --source-suite-dir "$SOURCE_SUITE_DIR"
    --endpoint-budget "$PHASE2_TOKENS"
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

if [[ "$RUN_ACTIVATIONS" == "1" ]]; then
  if [[ "$SKIP_EXISTING" != "1" || ! -s "$ACTIVATION_DIR/manifest.json" ]]; then
    activation_args=(
      "$PYTHON" -m aisafety.scripts.run_judge_criterion_switch_activations
      --workspace-root "$WORKDIR"
      --behavior-dir "$BEHAVIOR_DIR"
      --model-id "$MODEL_ID"
      --run-label "$RUN_LABEL"
      --cache-dir "$HF_HOME"
      --include-conditions "$ACTIVATION_CONDITIONS"
      --include-branches "$ACTIVATION_BRANCHES"
      --selected-layers "$SELECTED_LAYERS"
      --point-mode readout
      --max-score-length "$MAX_SCORE_LENGTH"
      --shard-size 32
      --compress-shards
      --out-dir "$ACTIVATION_DIR"
    )
    if [[ "$USE_4BIT" == "1" ]]; then activation_args+=(--use-4bit); fi
    if [[ -s "$ACTIVATION_DIR/traces.jsonl" ]]; then
      activation_args+=(--resume)
    fi
    echo "launch structured-CoT activation capture gpu=$GPU"
    if [[ "$DRY_RUN" == "1" ]]; then
      printf '  CUDA_VISIBLE_DEVICES=%q ' "$GPU"
      printf '%q ' "${activation_args[@]}"
      printf '\n'
    else
      CUDA_VISIBLE_DEVICES="$GPU" "${activation_args[@]}" \
        >"$LOG_DIR/activations.out" 2>"$LOG_DIR/activations.err"
    fi
  else
    echo "skip existing activation capture -> $ACTIVATION_DIR"
  fi

  if [[ "$SKIP_EXISTING" != "1" \
        || ! -s "$ACTIVATION_ANALYSIS_DIR/manifest.json" ]]; then
    run_logged activation_analysis \
      "$PYTHON" \
      -m aisafety.scripts.analyze_judge_criterion_confirmation_activations \
      --workspace-root "$WORKDIR" \
      --trace-dir "$ACTIVATION_DIR" \
      --target-layers "$TARGET_LAYERS" \
      --c-value 1.0 \
      --cv-folds 5 \
      --bootstrap "$BOOTSTRAP" \
      --seed "$SEED" \
      --out-dir "$ACTIVATION_ANALYSIS_DIR"
  else
    echo "skip existing activation analysis -> $ACTIVATION_ANALYSIS_DIR"
  fi
fi

if [[ "$DRY_RUN" != "1" ]]; then
  readout_args=(
    "$PYTHON" -m aisafety.scripts.read_judge_structured_cot
    --workspace-root "$WORKDIR"
    --analysis-dir "$ANALYSIS_DIR"
  )
  if [[ "$RUN_ACTIVATIONS" == "1" ]]; then
    readout_args+=(--activation-analysis-dir "$ACTIVATION_ANALYSIS_DIR")
  fi
  "${readout_args[@]}" >"$OUT_ROOT/readout.txt"
  cat "$OUT_ROOT/readout.txt"
fi

echo "COMPLETE"
echo "out_root=$OUT_ROOT"
echo "suite_dir=$SUITE_DIR"
echo "behavior_dir=$BEHAVIOR_DIR"
echo "analysis_dir=$ANALYSIS_DIR"
if [[ "$RUN_ACTIVATIONS" == "1" ]]; then
  echo "activation_dir=$ACTIVATION_DIR"
  echo "activation_analysis_dir=$ACTIVATION_ANALYSIS_DIR"
fi
