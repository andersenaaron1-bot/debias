#!/usr/bin/env bash
# Run evaluative prose/decision and factual mediator probing plus patching.

set -euo pipefail

WORKDIR="${WORKDIR:-$(pwd)}"
ARTROOT="${ARTROOT:-$WORKDIR}"
PYTHON="${PYTHON:-$WORKDIR/.venv/bin/python}"
RUN_TAG="${RUN_TAG:-judge_summeval_structured_cot_qwen3_8b_v1}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
RUN_LABEL="${RUN_LABEL:-}"
GPU="${GPU:-0}"

OUT_ROOT="${OUT_ROOT:-$ARTROOT/artifacts/mechanistic/$RUN_TAG}"
BEHAVIOR_DIR="${BEHAVIOR_DIR:-$OUT_ROOT/behavior}"
TRACE_DIR="${TRACE_DIR:-$OUT_ROOT/activations}"
DISSOCIATION_DIR="${DISSOCIATION_DIR:-$OUT_ROOT/prose_decision_dissociation}"
PATCH_DIR="${PATCH_DIR:-$OUT_ROOT/prose_decision_direction_patching}"
LOG_DIR="${LOG_DIR:-$OUT_ROOT/logs}"

FACTUAL_RUN_TAG="${FACTUAL_RUN_TAG:-judge_factual_cot_effect_qwen3_8b_v1}"
FACTUAL_OUT_ROOT="${FACTUAL_OUT_ROOT:-$ARTROOT/artifacts/mechanistic/$FACTUAL_RUN_TAG}"
FACTUAL_BUDGET_DIR="${FACTUAL_BUDGET_DIR:-$FACTUAL_OUT_ROOT/budget_sweep}"
FACTUAL_COT_ANALYSIS_DIR="${FACTUAL_COT_ANALYSIS_DIR:-$FACTUAL_OUT_ROOT/factual_cot_effect}"
FACTUAL_ACTIVATION_DIR="${FACTUAL_ACTIVATION_DIR:-$FACTUAL_OUT_ROOT/readout_activations}"
FACTUAL_DISSOCIATION_DIR="${FACTUAL_DISSOCIATION_DIR:-$FACTUAL_OUT_ROOT/factual_mediator_dissociation}"
FACTUAL_PATCH_DIR="${FACTUAL_PATCH_DIR:-$FACTUAL_OUT_ROOT/factual_mediator_direction_patching}"
FACTUAL_LOG_DIR="${FACTUAL_LOG_DIR:-$FACTUAL_OUT_ROOT/logs}"

POINT_NAME="${POINT_NAME:-phase2_readout_384}"
CONDITIONS="${CONDITIONS:-free_cot,criterion_scaffold,generic_scaffold,score_evidence}"
PROBE_TARGETS="${PROBE_TARGETS:-criterion_prose,target_grounded_prose,verdict_binding,criterion_target,final_choice}"
PATCH_TRANSITION_TYPES="${PATCH_TRANSITION_TYPES:-criterion_flip}"
PATCH_BRANCH_INDEX="${PATCH_BRANCH_INDEX:-0}"
PATCH_MAX_PAIRS="${PATCH_MAX_PAIRS:-16}"
PATCH_ALPHAS="${PATCH_ALPHAS:--2.0,-1.0,0.0,1.0,2.0}"
BOOTSTRAP="${BOOTSTRAP:-2000}"
MAX_SCORE_LENGTH="${MAX_SCORE_LENGTH:-8192}"
USE_4BIT="${USE_4BIT:-0}"
RUN_EVALUATIVE="${RUN_EVALUATIVE:-1}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"
RUN_PATCHING="${RUN_PATCHING:-1}"
RUN_FACTUAL="${RUN_FACTUAL:-0}"
RUN_FACTUAL_BEHAVIOR="${RUN_FACTUAL_BEHAVIOR:-1}"
RUN_FACTUAL_ACTIVATIONS="${RUN_FACTUAL_ACTIVATIONS:-1}"
RUN_FACTUAL_ANALYSIS="${RUN_FACTUAL_ANALYSIS:-1}"
RUN_FACTUAL_PATCHING="${RUN_FACTUAL_PATCHING:-1}"
FACTUAL_INCLUDE_DATASETS="${FACTUAL_INCLUDE_DATASETS:-gsm8k_verification,math500_verification,bbh_logical_deduction,arc_challenge,truthfulqa}"
FACTUAL_MAX_PAIRS_PER_DATASET="${FACTUAL_MAX_PAIRS_PER_DATASET:-24}"
FACTUAL_BRANCHES_PER_COMPARISON="${FACTUAL_BRANCHES_PER_COMPARISON:-3}"
FACTUAL_BUDGET_TOKENS="${FACTUAL_BUDGET_TOKENS:-0,128,512,2048}"
FACTUAL_ENDPOINT_BUDGET="${FACTUAL_ENDPOINT_BUDGET:-2048}"
FACTUAL_SELECTED_LAYERS="${FACTUAL_SELECTED_LAYERS:-4,8,12,16,20,24,28,32}"
FACTUAL_PROBE_TARGETS="${FACTUAL_PROBE_TARGETS:-criterion_target,current_choice,final_choice,target_reached}"
FACTUAL_PATCH_PROBE_TARGETS="${FACTUAL_PATCH_PROBE_TARGETS:-criterion_target,current_choice,final_choice}"
FACTUAL_PATCH_BRANCH_INDEX="${FACTUAL_PATCH_BRANCH_INDEX:-0}"
FACTUAL_PATCH_MAX_PAIRS="${FACTUAL_PATCH_MAX_PAIRS:-24}"
FACTUAL_PATCH_ALPHAS="${FACTUAL_PATCH_ALPHAS:--2.0,-1.0,0.0,1.0,2.0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SEED="${SEED:-1234}"

cd "$WORKDIR"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$WORKDIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$ARTROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$LOG_DIR" "$FACTUAL_LOG_DIR"

echo "Running prose/decision dissociation suite"
echo "  run_tag=$RUN_TAG"
echo "  model=$MODEL_ID gpu=$GPU use_4bit=$USE_4BIT"
echo "  behavior_dir=$BEHAVIOR_DIR"
echo "  trace_dir=$TRACE_DIR"
echo "  dissociation_dir=$DISSOCIATION_DIR"
echo "  patch_dir=$PATCH_DIR"
echo "  patch_transition_types=${PATCH_TRANSITION_TYPES:-<all>}"
echo "  run_factual=$RUN_FACTUAL factual_out_root=$FACTUAL_OUT_ROOT"

if [[ "$RUN_EVALUATIVE" == "1" ]]; then
  if [[ "$RUN_ANALYSIS" == "1" ]]; then
    if [[ "$SKIP_EXISTING" == "1" && -s "$DISSOCIATION_DIR/manifest.json" ]]; then
      echo "skip existing prose/decision analysis -> $DISSOCIATION_DIR"
    else
      "$PYTHON" -m aisafety.scripts.analyze_judge_prose_decision_dissociation \
        --workspace-root "$WORKDIR" \
        --trace-dir "$TRACE_DIR" \
        --point-name "$POINT_NAME" \
        --conditions "$CONDITIONS" \
        --bootstrap "$BOOTSTRAP" \
        --seed "$SEED" \
        --out-dir "$DISSOCIATION_DIR" \
        >"$LOG_DIR/prose_decision_analysis.out" \
        2>"$LOG_DIR/prose_decision_analysis.err"
    fi
  fi

  if [[ "$RUN_PATCHING" == "1" ]]; then
    if [[ "$SKIP_EXISTING" == "1" && -s "$PATCH_DIR/manifest.json" ]]; then
      echo "skip existing prose/decision patching -> $PATCH_DIR"
    else
      patch_args=(
        "$PYTHON" -m aisafety.scripts.run_judge_prose_decision_direction_patching
        --workspace-root "$WORKDIR"
        --behavior-dir "$BEHAVIOR_DIR"
        --analysis-dir "$DISSOCIATION_DIR"
        --model-id "$MODEL_ID"
        --labels A,B,C
        --probe-targets "$PROBE_TARGETS"
        --conditions "$CONDITIONS"
        --include-orders original,swapped
        --branch-index "$PATCH_BRANCH_INDEX"
        --stage phase2
        --budget-tokens 384
        --alphas "$PATCH_ALPHAS"
        --max-pairs "$PATCH_MAX_PAIRS"
        --max-score-length "$MAX_SCORE_LENGTH"
        --bootstrap "$BOOTSTRAP"
        --seed "$SEED"
        --out-dir "$PATCH_DIR"
      )
      if [[ -n "$PATCH_TRANSITION_TYPES" ]]; then
        patch_args+=(--transition-types "$PATCH_TRANSITION_TYPES")
      fi
      if [[ "$USE_4BIT" == "1" ]]; then
        patch_args+=(--use-4bit)
      fi
      CUDA_VISIBLE_DEVICES="$GPU" "${patch_args[@]}" \
        >"$LOG_DIR/prose_decision_patching.out" \
        2>"$LOG_DIR/prose_decision_patching.err"
    fi
  fi

  "$PYTHON" -m aisafety.scripts.read_judge_prose_decision_dissociation \
    --workspace-root "$WORKDIR" \
    --analysis-dir "$DISSOCIATION_DIR" \
    --patch-dir "$PATCH_DIR" \
    >"$OUT_ROOT/prose_decision_readout.txt"
  cat "$OUT_ROOT/prose_decision_readout.txt"
fi

if [[ "$RUN_FACTUAL" == "1" ]]; then
  echo "Running factual mediator baseline"
  echo "  factual_budget_dir=$FACTUAL_BUDGET_DIR"
  echo "  factual_activation_dir=$FACTUAL_ACTIVATION_DIR"
  echo "  factual_dissociation_dir=$FACTUAL_DISSOCIATION_DIR"
  echo "  factual_patch_dir=$FACTUAL_PATCH_DIR"

  if [[ "$RUN_FACTUAL_BEHAVIOR" == "1" ]]; then
    if [[ "$SKIP_EXISTING" == "1" && -s "$FACTUAL_COT_ANALYSIS_DIR/manifest.json" ]]; then
      echo "skip existing factual behavior analysis -> $FACTUAL_COT_ANALYSIS_DIR"
    else
      RUN_TAG="$FACTUAL_RUN_TAG" \
      MODEL_ID="$MODEL_ID" \
      RUN_LABEL="$RUN_LABEL" \
      GPU="$GPU" \
      OUT_ROOT="$FACTUAL_OUT_ROOT" \
      BUDGET_DIR="$FACTUAL_BUDGET_DIR" \
      FACTUAL_ANALYSIS_DIR="$FACTUAL_COT_ANALYSIS_DIR" \
      LOG_DIR="$FACTUAL_LOG_DIR" \
      INCLUDE_DATASETS="$FACTUAL_INCLUDE_DATASETS" \
      MAX_PAIRS_PER_DATASET="$FACTUAL_MAX_PAIRS_PER_DATASET" \
      BRANCHES_PER_COMPARISON="$FACTUAL_BRANCHES_PER_COMPARISON" \
      BUDGET_TOKENS="$FACTUAL_BUDGET_TOKENS" \
      ENDPOINT_BUDGET="$FACTUAL_ENDPOINT_BUDGET" \
      BOOTSTRAP="$BOOTSTRAP" \
      SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-4}" \
      USE_4BIT="$USE_4BIT" \
      SKIP_EXISTING="$SKIP_EXISTING" \
      SEED="$SEED" \
      bash "$WORKDIR/cluster/local/run_judge_factual_cot_effect_qwen3_8b.sh" \
        >"$FACTUAL_LOG_DIR/factual_behavior_suite.out" \
        2>"$FACTUAL_LOG_DIR/factual_behavior_suite.err"
    fi
  fi

  if [[ "$RUN_FACTUAL_ACTIVATIONS" == "1" ]]; then
    if [[ "$SKIP_EXISTING" == "1" && -s "$FACTUAL_ACTIVATION_DIR/manifest.json" ]]; then
      echo "skip existing factual activation capture -> $FACTUAL_ACTIVATION_DIR"
    else
      activation_args=(
        "$PYTHON" -m aisafety.scripts.run_judge_factual_readout_activations
        --workspace-root "$WORKDIR"
        --budget-run-dir "$FACTUAL_BUDGET_DIR"
        --model-id "$MODEL_ID"
        --cache-dir "$HF_HOME"
        --include-datasets "$FACTUAL_INCLUDE_DATASETS"
        --budget-tokens "$FACTUAL_BUDGET_TOKENS"
        --selected-layers "$FACTUAL_SELECTED_LAYERS"
        --max-score-length "$MAX_SCORE_LENGTH"
        --shard-size 32
        --compress-shards
        --out-dir "$FACTUAL_ACTIVATION_DIR"
      )
      if [[ "$USE_4BIT" == "1" ]]; then activation_args+=(--use-4bit); fi
      if [[ -s "$FACTUAL_ACTIVATION_DIR/traces.jsonl" ]]; then activation_args+=(--resume); fi
      CUDA_VISIBLE_DEVICES="$GPU" "${activation_args[@]}" \
        >"$FACTUAL_LOG_DIR/factual_activations.out" \
        2>"$FACTUAL_LOG_DIR/factual_activations.err"
    fi
  fi

  if [[ "$RUN_FACTUAL_ANALYSIS" == "1" ]]; then
    if [[ "$SKIP_EXISTING" == "1" && -s "$FACTUAL_DISSOCIATION_DIR/manifest.json" ]]; then
      echo "skip existing factual mediator analysis -> $FACTUAL_DISSOCIATION_DIR"
    else
      "$PYTHON" -m aisafety.scripts.analyze_judge_factual_mediator_dissociation \
        --workspace-root "$WORKDIR" \
        --trace-dir "$FACTUAL_ACTIVATION_DIR" \
        --point-name "readout_$FACTUAL_ENDPOINT_BUDGET" \
        --include-datasets "$FACTUAL_INCLUDE_DATASETS" \
        --probe-targets "$FACTUAL_PROBE_TARGETS" \
        --bootstrap "$BOOTSTRAP" \
        --seed "$SEED" \
        --out-dir "$FACTUAL_DISSOCIATION_DIR" \
        >"$FACTUAL_LOG_DIR/factual_mediator_analysis.out" \
        2>"$FACTUAL_LOG_DIR/factual_mediator_analysis.err"
    fi
  fi

  if [[ "$RUN_FACTUAL_PATCHING" == "1" ]]; then
    if [[ "$SKIP_EXISTING" == "1" && -s "$FACTUAL_PATCH_DIR/manifest.json" ]]; then
      echo "skip existing factual mediator patching -> $FACTUAL_PATCH_DIR"
    else
      factual_patch_args=(
        "$PYTHON" -m aisafety.scripts.run_judge_factual_mediator_direction_patching
        --workspace-root "$WORKDIR"
        --budget-run-dir "$FACTUAL_BUDGET_DIR"
        --analysis-dir "$FACTUAL_DISSOCIATION_DIR"
        --model-id "$MODEL_ID"
        --labels A,B
        --probe-targets "$FACTUAL_PATCH_PROBE_TARGETS"
        --include-datasets "$FACTUAL_INCLUDE_DATASETS"
        --budget-tokens "$FACTUAL_ENDPOINT_BUDGET"
        --branch-index "$FACTUAL_PATCH_BRANCH_INDEX"
        --alphas "$FACTUAL_PATCH_ALPHAS"
        --max-pairs "$FACTUAL_PATCH_MAX_PAIRS"
        --max-score-length "$MAX_SCORE_LENGTH"
        --bootstrap "$BOOTSTRAP"
        --seed "$SEED"
        --out-dir "$FACTUAL_PATCH_DIR"
      )
      if [[ "$USE_4BIT" == "1" ]]; then factual_patch_args+=(--use-4bit); fi
      CUDA_VISIBLE_DEVICES="$GPU" "${factual_patch_args[@]}" \
        >"$FACTUAL_LOG_DIR/factual_mediator_patching.out" \
        2>"$FACTUAL_LOG_DIR/factual_mediator_patching.err"
    fi
  fi

  "$PYTHON" -m aisafety.scripts.read_judge_factual_mediator_dissociation \
    --workspace-root "$WORKDIR" \
    --analysis-dir "$FACTUAL_DISSOCIATION_DIR" \
    --patch-dir "$FACTUAL_PATCH_DIR" \
    >"$FACTUAL_OUT_ROOT/factual_mediator_readout.txt"
  cat "$FACTUAL_OUT_ROOT/factual_mediator_readout.txt"
fi

echo "COMPLETE"
echo "dissociation_dir=$DISSOCIATION_DIR"
echo "patch_dir=$PATCH_DIR"
if [[ "$RUN_FACTUAL" == "1" ]]; then
  echo "factual_dissociation_dir=$FACTUAL_DISSOCIATION_DIR"
  echo "factual_patch_dir=$FACTUAL_PATCH_DIR"
fi
