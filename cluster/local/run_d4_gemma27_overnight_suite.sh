#!/bin/bash
# Run a Gemma 2 base-vs-IT replication suite on a local GPU host.
#
# This wrapper is intended for an unattended overnight run. It covers the
# decisive minimal/standard experiments for:
#   1. human-vs-LLM paired judgments,
#   2. deterministic surface-BT cue judgments,
# with response-likelihood controls for the standard template and with both
# chat-template and plain-prompt Gemma-IT controls. It reuses base-model scores
# between chat/plain summaries because the base model is always run with the
# plain prompt style.

set -euo pipefail

WORKDIR="${WORKDIR:-$(pwd)}"
ARTROOT="${ARTROOT:-$WORKDIR}"
PYTHON="${PYTHON:-python}"
GEMMA_SIZE="${GEMMA_SIZE:-27b}"

case "$GEMMA_SIZE" in
  9b)
    DEFAULT_RUN_TAG="gemma2_9b_overnight_v1"
    DEFAULT_BASE_LABEL="gemma2_9b_base"
    DEFAULT_BASE_MODEL_ID="google/gemma-2-9b"
    DEFAULT_INSTRUCT_LABEL="gemma2_9b_it"
    DEFAULT_INSTRUCT_MODEL_ID="google/gemma-2-9b-it"
    DEFAULT_GPU_A="0"
    DEFAULT_GPU_B="1"
    ;;
  27b)
    DEFAULT_RUN_TAG="gemma2_27b_overnight_v1"
    DEFAULT_BASE_LABEL="gemma2_27b_base"
    DEFAULT_BASE_MODEL_ID="google/gemma-2-27b"
    DEFAULT_INSTRUCT_LABEL="gemma2_27b_it"
    DEFAULT_INSTRUCT_MODEL_ID="google/gemma-2-27b-it"
    DEFAULT_GPU_A="1"
    DEFAULT_GPU_B="7"
    ;;
  *)
    echo "Unsupported GEMMA_SIZE=$GEMMA_SIZE. Use 9b or 27b." >&2
    exit 2
    ;;
esac

RUN_TAG="${RUN_TAG:-$DEFAULT_RUN_TAG}"

BASE_LABEL="${BASE_LABEL:-$DEFAULT_BASE_LABEL}"
BASE_MODEL_ID="${BASE_MODEL_ID:-$DEFAULT_BASE_MODEL_ID}"
BASE_PROMPT_STYLE="${BASE_PROMPT_STYLE:-plain}"
BASE_STAGE="${BASE_STAGE:-base_lm}"
INSTRUCT_LABEL="${INSTRUCT_LABEL:-$DEFAULT_INSTRUCT_LABEL}"
INSTRUCT_MODEL_ID="${INSTRUCT_MODEL_ID:-$DEFAULT_INSTRUCT_MODEL_ID}"
INSTRUCT_PROMPT_STYLE="${INSTRUCT_PROMPT_STYLE:-chat_template}"
PLAIN_INSTRUCT_PROMPT_STYLE="${PLAIN_INSTRUCT_PROMPT_STYLE:-plain}"
INSTRUCT_STAGE="${INSTRUCT_STAGE:-it_lm}"

GPU_A="${GPU_A:-$DEFAULT_GPU_A}"
GPU_B="${GPU_B:-$DEFAULT_GPU_B}"

RUN_HLLM="${RUN_HLLM:-1}"
RUN_SURFACE_BT="${RUN_SURFACE_BT:-1}"
RUN_CHAT_TEMPLATE="${RUN_CHAT_TEMPLATE:-1}"
RUN_PLAIN_CONTROL="${RUN_PLAIN_CONTROL:-1}"
RUN_LIKELIHOOD="${RUN_LIKELIHOOD:-1}"
RUN_BOOTSTRAP="${RUN_BOOTSTRAP:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

HLLM_PAIR_JSONL="${HLLM_PAIR_JSONL:-$ARTROOT/data/derived/d4_human_llm_alignment_pairs_matched_lenlex_relaxed_v1/pairs.jsonl}"
HLLM_PAIR_OUT_DIR="${HLLM_PAIR_OUT_DIR:-$ARTROOT/data/derived/d4_human_llm_stage_contrast_pairs_${RUN_TAG}}"
HLLM_BT_PAIRS_JSONL="${HLLM_BT_PAIRS_JSONL:-$HLLM_PAIR_OUT_DIR/bt_pairs.jsonl}"
HLLM_OUT_ROOT="${HLLM_OUT_ROOT:-$ARTROOT/artifacts/mechanistic/d4_hllm_stage_${RUN_TAG}}"
HLLM_SUMMARY_ROOT="${HLLM_SUMMARY_ROOT:-$ARTROOT/artifacts/mechanistic}"

SURFACE_COUNTERFACTUAL_JSONL="${SURFACE_COUNTERFACTUAL_JSONL:-$ARTROOT/data/derived/d4_surface_counterfactual_pairs_matched_lenlex_relaxed_v1/counterfactuals.jsonl}"
SURFACE_BT_PAIR_OUT_DIR="${SURFACE_BT_PAIR_OUT_DIR:-$ARTROOT/data/derived/d4_bt_surface_stage_pairs_matched_lenlex_relaxed_v1}"
SURFACE_BT_PAIRS_JSONL="${SURFACE_BT_PAIRS_JSONL:-$SURFACE_BT_PAIR_OUT_DIR/bt_pairs.jsonl}"
SURFACE_OUT_ROOT="${SURFACE_OUT_ROOT:-$ARTROOT/artifacts/mechanistic/d4_bt_surface_stage_${RUN_TAG}}"
SURFACE_SUMMARY_ROOT="${SURFACE_SUMMARY_ROOT:-$ARTROOT/artifacts/mechanistic}"
AXES="${AXES:-structured_assistant_packaging,answer_likeness_packaging,formal_institutional_packaging}"
DIRECTIONS="${DIRECTIONS:-}"
ROLES="${ROLES:-}"
MAX_COUNTERFACTUALS="${MAX_COUNTERFACTUALS:-0}"
INCLUDE_ORDER_SWAPS="${INCLUDE_ORDER_SWAPS:-1}"

MAX_SOURCE_PAIRS="${MAX_SOURCE_PAIRS:-0}"
SCORE_MAX_PAIRS="${SCORE_MAX_PAIRS:-0}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-1}"
LIKELIHOOD_BATCH_SIZE="${LIKELIHOOD_BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
USE_4BIT="${USE_4BIT:-0}"
SEED="${SEED:-1234}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-2000}"

LOG_DIR="${LOG_DIR:-$ARTROOT/artifacts/mechanistic/logs/${RUN_TAG}}"

mkdir -p "$LOG_DIR"
cd "$WORKDIR"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$WORKDIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$ARTROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

echo "Running Gemma 2 $GEMMA_SIZE overnight D4 judge-bias suite"
echo "  run_tag=$RUN_TAG"
echo "  workdir=$WORKDIR"
echo "  artroot=$ARTROOT"
echo "  base=$BASE_LABEL :: $BASE_MODEL_ID :: $BASE_PROMPT_STYLE"
echo "  instruct=$INSTRUCT_LABEL :: $INSTRUCT_MODEL_ID :: $INSTRUCT_PROMPT_STYLE"
echo "  plain_control=$RUN_PLAIN_CONTROL :: $PLAIN_INSTRUCT_PROMPT_STYLE"
echo "  run_hllm=$RUN_HLLM"
echo "  run_surface_bt=$RUN_SURFACE_BT"
echo "  run_likelihood=$RUN_LIKELIHOOD"
echo "  run_bootstrap=$RUN_BOOTSTRAP"
echo "  gpus=$GPU_A,$GPU_B"
echo "  score_batch_size=$SCORE_BATCH_SIZE"
echo "  likelihood_batch_size=$LIKELIHOOD_BATCH_SIZE"
echo "  max_length=$MAX_LENGTH"
echo "  use_4bit=$USE_4BIT"
echo "  bootstrap_samples=$BOOTSTRAP_SAMPLES"
echo "  skip_existing=$SKIP_EXISTING"
echo "  logs=$LOG_DIR"

if [[ "$RUN_CHAT_TEMPLATE" != "1" && "$RUN_PLAIN_CONTROL" != "1" ]]; then
  echo "Nothing to run: set RUN_CHAT_TEMPLATE=1 and/or RUN_PLAIN_CONTROL=1." >&2
  exit 2
fi

require_file() {
  local path="$1"
  local name="$2"
  if [[ ! -f "$path" ]]; then
    echo "Missing $name: $path" >&2
    exit 2
  fi
}

run_logged() {
  local name="$1"
  shift
  local stdout="$LOG_DIR/${name}.out"
  local stderr="$LOG_DIR/${name}.err"
  echo "run $name"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '  '
    printf '%q ' "$@"
    printf '\n'
    return 0
  fi
  "$@" >"$stdout" 2>"$stderr"
}

score_done() {
  local out_dir="$1"
  local score_file="$2"
  [[ "$SKIP_EXISTING" == "1" && -f "$out_dir/manifest.json" && -f "$out_dir/$score_file" ]]
}

PIDS=()
launch_hllm() {
  local name="$1"
  local label="$2"
  local model_id="$3"
  local scoring_mode="$4"
  local prompt_style="$5"
  local template="$6"
  local out_dir="$7"
  local batch_size="$8"
  local gpu_id="$9"

  mkdir -p "$out_dir"
  if score_done "$out_dir" "hllm_stage_scores.csv"; then
    echo "skip existing $name -> $out_dir"
    return 0
  fi
  echo "launch $name label=$label gpu=$gpu_id model=$model_id mode=$scoring_mode prompt_style=$prompt_style template=$template"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  (
    export CUDA_VISIBLE_DEVICES="$gpu_id"
    cmd=(
      "$PYTHON" -m aisafety.scripts.run_d4_human_llm_stage_contrast
      --workspace-root "$WORKDIR"
      --bt-pairs-jsonl "$HLLM_BT_PAIRS_JSONL"
      --scoring-mode "$scoring_mode"
      --stage-label "$label"
      --model-id "$model_id"
      --cache-dir "$HF_HOME"
      --prompt-style "$prompt_style"
      --comparison-template "$template"
      --max-pairs "$SCORE_MAX_PAIRS"
      --score-batch-size "$batch_size"
      --max-length "$MAX_LENGTH"
      --seed "$SEED"
      --out-dir "$out_dir"
    )
    if [[ "$USE_4BIT" == "1" ]]; then
      cmd+=(--use-4bit)
    fi
    "${cmd[@]}"
  ) >"$LOG_DIR/${name}.out" 2>"$LOG_DIR/${name}.err" &
  PIDS+=("$!")
}

launch_surface() {
  local name="$1"
  local label="$2"
  local model_id="$3"
  local stage="$4"
  local scoring_mode="$5"
  local prompt_style="$6"
  local template="$7"
  local out_dir="$8"
  local batch_size="$9"
  local gpu_id="${10}"

  mkdir -p "$out_dir"
  if score_done "$out_dir" "bt_stage_scores.csv"; then
    echo "skip existing $name -> $out_dir"
    return 0
  fi
  echo "launch $name label=$label gpu=$gpu_id model=$model_id mode=$scoring_mode prompt_style=$prompt_style template=$template"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  (
    export CUDA_VISIBLE_DEVICES="$gpu_id"
    cmd=(
      "$PYTHON" -m aisafety.scripts.run_d4_bt_stage_contrast
      --workspace-root "$WORKDIR"
      --bt-pairs-jsonl "$SURFACE_BT_PAIRS_JSONL"
      --stage "$stage"
      --scoring-mode "$scoring_mode"
      --stage-label "$label"
      --model-id "$model_id"
      --cache-dir "$HF_HOME"
      --prompt-style "$prompt_style"
      --comparison-template "$template"
      --max-pairs "$SCORE_MAX_PAIRS"
      --score-batch-size "$batch_size"
      --max-length "$MAX_LENGTH"
      --seed "$SEED"
      --out-dir "$out_dir"
    )
    if [[ "$USE_4BIT" == "1" ]]; then
      cmd+=(--use-4bit)
    fi
    "${cmd[@]}"
  ) >"$LOG_DIR/${name}.out" 2>"$LOG_DIR/${name}.err" &
  PIDS+=("$!")
}

wait_wave() {
  local failed=0
  local pid
  if [[ "${#PIDS[@]}" -eq 0 ]]; then
    return 0
  fi
  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  PIDS=()
  if [[ "$failed" != "0" ]]; then
    echo "At least one scoring stage failed. Check logs in $LOG_DIR" >&2
    exit 1
  fi
}

hllm_base_forced_dir() { echo "$HLLM_OUT_ROOT/base/$1/${BASE_LABEL}_forced"; }
hllm_base_like_dir() { echo "$HLLM_OUT_ROOT/base/$1/${BASE_LABEL}_response_likelihood"; }
hllm_chat_forced_dir() { echo "$HLLM_OUT_ROOT/chat/$1/${INSTRUCT_LABEL}_forced"; }
hllm_chat_like_dir() { echo "$HLLM_OUT_ROOT/chat/$1/${INSTRUCT_LABEL}_response_likelihood"; }
hllm_plain_forced_dir() { echo "$HLLM_OUT_ROOT/plain/$1/${INSTRUCT_LABEL}_forced"; }
hllm_plain_like_dir() { echo "$HLLM_OUT_ROOT/plain/$1/${INSTRUCT_LABEL}_response_likelihood"; }

surface_base_forced_dir() { echo "$SURFACE_OUT_ROOT/base/$1/${BASE_LABEL}_forced"; }
surface_base_like_dir() { echo "$SURFACE_OUT_ROOT/base/$1/${BASE_LABEL}_response_likelihood"; }
surface_chat_forced_dir() { echo "$SURFACE_OUT_ROOT/chat/$1/${INSTRUCT_LABEL}_forced"; }
surface_chat_like_dir() { echo "$SURFACE_OUT_ROOT/chat/$1/${INSTRUCT_LABEL}_response_likelihood"; }
surface_plain_forced_dir() { echo "$SURFACE_OUT_ROOT/plain/$1/${INSTRUCT_LABEL}_forced"; }
surface_plain_like_dir() { echo "$SURFACE_OUT_ROOT/plain/$1/${INSTRUCT_LABEL}_response_likelihood"; }

summarize_hllm_template() {
  local style="$1"
  local template="$2"
  local instruct_forced="$3"
  local instruct_like="$4"
  local out_dir="$HLLM_SUMMARY_ROOT/d4_human_llm_stage_contrast_summary_${RUN_TAG}_hllm_${style}_${template}"

  args=(
    "$PYTHON" -m aisafety.scripts.summarize_d4_human_llm_stage_contrasts
    --workspace-root "$WORKDIR"
    --run "$BASE_LABEL=$(hllm_base_forced_dir "$template")"
    --run "$INSTRUCT_LABEL=$instruct_forced"
    --contrast "${INSTRUCT_LABEL}_minus_${BASE_LABEL}=${INSTRUCT_LABEL}-${BASE_LABEL}"
  )
  if [[ "$template" == "standard" && "$RUN_LIKELIHOOD" == "1" ]]; then
    args+=(
      --run "${BASE_LABEL}_like=$(hllm_base_like_dir "$template")"
      --run "${INSTRUCT_LABEL}_like=$instruct_like"
      --contrast "${INSTRUCT_LABEL}_like_minus_${BASE_LABEL}_like=${INSTRUCT_LABEL}_like-${BASE_LABEL}_like"
    )
  fi
  args+=(--out-dir "$out_dir")
  run_logged "summarize_hllm_${style}_${template}" "${args[@]}"
}

combine_hllm_templates() {
  local style="$1"
  local out_dir="$HLLM_SUMMARY_ROOT/d4_human_llm_template_sensitivity_${RUN_TAG}_hllm_${style}"
  run_logged "combine_hllm_${style}" \
    "$PYTHON" -m aisafety.scripts.summarize_d4_human_llm_template_sensitivity \
    --workspace-root "$WORKDIR" \
    --summary "minimal=$HLLM_SUMMARY_ROOT/d4_human_llm_stage_contrast_summary_${RUN_TAG}_hllm_${style}_minimal" \
    --summary "standard=$HLLM_SUMMARY_ROOT/d4_human_llm_stage_contrast_summary_${RUN_TAG}_hllm_${style}_standard" \
    --template-contrast "standard_minus_minimal=standard-minimal" \
    --stage-contrast "${INSTRUCT_LABEL}_minus_${BASE_LABEL}=${INSTRUCT_LABEL}-${BASE_LABEL}" \
    --out-dir "$out_dir"
  if [[ "$RUN_BOOTSTRAP" == "1" ]]; then
    run_logged "bootstrap_hllm_${style}" \
      "$PYTHON" -m aisafety.scripts.bootstrap_d4_mean_effects \
      --workspace-root "$WORKDIR" \
      --input "$out_dir/template_stage_interaction_pair_deltas.csv" \
      --value-col stage_template_interaction_llm_margin \
      --unit-col pair_id \
      --group-cols stage_contrast,template_contrast \
      --bootstrap "$BOOTSTRAP_SAMPLES" \
      --seed "$SEED" \
      --metric-name hllm_stage_template_interaction \
      --out-csv "$out_dir/bootstrap_stage_template_interactions.csv"
  fi
}

combine_surface_templates() {
  local style="$1"
  local instruct_minimal="$2"
  local instruct_standard="$3"
  local instruct_like="$4"
  local out_dir="$SURFACE_SUMMARY_ROOT/d4_bt_surface_stage_template_summary_${RUN_TAG}_${style}"

  args=(
    "$PYTHON" -m aisafety.scripts.summarize_d4_bt_stage_templates
    --workspace-root "$WORKDIR"
    --run "minimal:$BASE_LABEL=$(surface_base_forced_dir minimal)"
    --run "minimal:$INSTRUCT_LABEL=$instruct_minimal"
    --run "standard:$BASE_LABEL=$(surface_base_forced_dir standard)"
    --run "standard:$INSTRUCT_LABEL=$instruct_standard"
    --template-contrast "standard_minus_minimal=standard-minimal"
    --stage-contrast "${INSTRUCT_LABEL}_minus_${BASE_LABEL}=${INSTRUCT_LABEL}-${BASE_LABEL}"
  )
  if [[ "$RUN_LIKELIHOOD" == "1" ]]; then
    args+=(
      --run "standard:${BASE_LABEL}_like=$(surface_base_like_dir standard)"
      --run "standard:${INSTRUCT_LABEL}_like=$instruct_like"
      --stage-contrast "${INSTRUCT_LABEL}_like_minus_${BASE_LABEL}_like=${INSTRUCT_LABEL}_like-${BASE_LABEL}_like"
    )
  fi
  args+=(--out-dir "$out_dir")
  run_logged "combine_surface_${style}" "${args[@]}"
  if [[ "$RUN_BOOTSTRAP" == "1" ]]; then
    run_logged "bootstrap_surface_${style}" \
      "$PYTHON" -m aisafety.scripts.bootstrap_d4_mean_effects \
      --workspace-root "$WORKDIR" \
      --input "$out_dir/template_stage_interaction_pair_deltas.csv" \
      --value-col stage_template_interaction_cue_plus_margin \
      --unit-col counterfactual_id \
      --group-cols stage_contrast,template_contrast \
      --bootstrap "$BOOTSTRAP_SAMPLES" \
      --seed "$SEED" \
      --metric-name surface_stage_template_interaction \
      --out-csv "$out_dir/bootstrap_stage_template_interactions.csv"
  fi
}

if [[ "$RUN_HLLM" == "1" ]]; then
  require_file "$HLLM_PAIR_JSONL" "HLLM_PAIR_JSONL"
  if [[ "$SKIP_EXISTING" == "1" && -f "$HLLM_BT_PAIRS_JSONL" ]]; then
    echo "skip existing HLLM BT pairs -> $HLLM_BT_PAIRS_JSONL"
  else
    run_logged "build_hllm_bt_pairs" \
      "$PYTHON" -m aisafety.scripts.build_d4_human_llm_stage_contrast_pairs \
      --workspace-root "$WORKDIR" \
      --pair-jsonl "$HLLM_PAIR_JSONL" \
      --out-dir "$HLLM_PAIR_OUT_DIR" \
      --max-pairs "$MAX_SOURCE_PAIRS" \
      --include-order-swaps
  fi

  for template in minimal standard; do
    PIDS=()
    launch_hllm "hllm_base_${template}_forced" "$BASE_LABEL" "$BASE_MODEL_ID" forced_choice "$BASE_PROMPT_STYLE" "$template" "$(hllm_base_forced_dir "$template")" "$SCORE_BATCH_SIZE" "$GPU_A"
    if [[ "$RUN_CHAT_TEMPLATE" == "1" ]]; then
      launch_hllm "hllm_chat_${template}_forced" "$INSTRUCT_LABEL" "$INSTRUCT_MODEL_ID" forced_choice "$INSTRUCT_PROMPT_STYLE" "$template" "$(hllm_chat_forced_dir "$template")" "$SCORE_BATCH_SIZE" "$GPU_B"
    fi
    wait_wave

    if [[ "$RUN_PLAIN_CONTROL" == "1" ]]; then
      PIDS=()
      launch_hllm "hllm_plain_${template}_forced" "$INSTRUCT_LABEL" "$INSTRUCT_MODEL_ID" forced_choice "$PLAIN_INSTRUCT_PROMPT_STYLE" "$template" "$(hllm_plain_forced_dir "$template")" "$SCORE_BATCH_SIZE" "$GPU_B"
      wait_wave
    fi

    if [[ "$template" == "standard" && "$RUN_LIKELIHOOD" == "1" ]]; then
      PIDS=()
      launch_hllm "hllm_base_${template}_like" "${BASE_LABEL}_like" "$BASE_MODEL_ID" response_likelihood "$BASE_PROMPT_STYLE" "$template" "$(hllm_base_like_dir "$template")" "$LIKELIHOOD_BATCH_SIZE" "$GPU_A"
      if [[ "$RUN_CHAT_TEMPLATE" == "1" ]]; then
        launch_hllm "hllm_chat_${template}_like" "${INSTRUCT_LABEL}_like" "$INSTRUCT_MODEL_ID" response_likelihood "$INSTRUCT_PROMPT_STYLE" "$template" "$(hllm_chat_like_dir "$template")" "$LIKELIHOOD_BATCH_SIZE" "$GPU_B"
      fi
      wait_wave

      if [[ "$RUN_PLAIN_CONTROL" == "1" ]]; then
        PIDS=()
        launch_hllm "hllm_plain_${template}_like" "${INSTRUCT_LABEL}_like" "$INSTRUCT_MODEL_ID" response_likelihood "$PLAIN_INSTRUCT_PROMPT_STYLE" "$template" "$(hllm_plain_like_dir "$template")" "$LIKELIHOOD_BATCH_SIZE" "$GPU_B"
        wait_wave
      fi
    fi

    if [[ "$RUN_CHAT_TEMPLATE" == "1" ]]; then
      summarize_hllm_template "chat" "$template" "$(hllm_chat_forced_dir "$template")" "$(hllm_chat_like_dir "$template")"
    fi
    if [[ "$RUN_PLAIN_CONTROL" == "1" ]]; then
      summarize_hllm_template "plain" "$template" "$(hllm_plain_forced_dir "$template")" "$(hllm_plain_like_dir "$template")"
    fi
  done

  if [[ "$RUN_CHAT_TEMPLATE" == "1" ]]; then
    combine_hllm_templates "chat"
  fi
  if [[ "$RUN_PLAIN_CONTROL" == "1" ]]; then
    combine_hllm_templates "plain"
  fi
fi

if [[ "$RUN_SURFACE_BT" == "1" ]]; then
  require_file "$SURFACE_COUNTERFACTUAL_JSONL" "SURFACE_COUNTERFACTUAL_JSONL"
  if [[ "$SKIP_EXISTING" == "1" && -f "$SURFACE_BT_PAIRS_JSONL" ]]; then
    echo "skip existing surface BT pairs -> $SURFACE_BT_PAIRS_JSONL"
  else
    pair_cmd=(
      "$PYTHON" -m aisafety.scripts.build_d4_bt_stage_contrast_pairs
      --workspace-root "$WORKDIR"
      --counterfactual-jsonl "$SURFACE_COUNTERFACTUAL_JSONL"
      --out-dir "$SURFACE_BT_PAIR_OUT_DIR"
      --axes "$AXES"
      --directions "$DIRECTIONS"
      --roles "$ROLES"
      --max-counterfactuals "$MAX_COUNTERFACTUALS"
    )
    if [[ "$INCLUDE_ORDER_SWAPS" == "1" ]]; then
      pair_cmd+=(--include-order-swaps)
    else
      pair_cmd+=(--no-include-order-swaps)
    fi
    run_logged "build_surface_bt_pairs" "${pair_cmd[@]}"
  fi

  for template in minimal standard; do
    PIDS=()
    launch_surface "surface_base_${template}_forced" "$BASE_LABEL" "$BASE_MODEL_ID" "$BASE_STAGE" forced_choice "$BASE_PROMPT_STYLE" "$template" "$(surface_base_forced_dir "$template")" "$SCORE_BATCH_SIZE" "$GPU_A"
    if [[ "$RUN_CHAT_TEMPLATE" == "1" ]]; then
      launch_surface "surface_chat_${template}_forced" "$INSTRUCT_LABEL" "$INSTRUCT_MODEL_ID" "$INSTRUCT_STAGE" forced_choice "$INSTRUCT_PROMPT_STYLE" "$template" "$(surface_chat_forced_dir "$template")" "$SCORE_BATCH_SIZE" "$GPU_B"
    fi
    wait_wave

    if [[ "$RUN_PLAIN_CONTROL" == "1" ]]; then
      PIDS=()
      launch_surface "surface_plain_${template}_forced" "$INSTRUCT_LABEL" "$INSTRUCT_MODEL_ID" "$INSTRUCT_STAGE" forced_choice "$PLAIN_INSTRUCT_PROMPT_STYLE" "$template" "$(surface_plain_forced_dir "$template")" "$SCORE_BATCH_SIZE" "$GPU_B"
      wait_wave
    fi

    if [[ "$template" == "standard" && "$RUN_LIKELIHOOD" == "1" ]]; then
      PIDS=()
      launch_surface "surface_base_${template}_like" "${BASE_LABEL}_like" "$BASE_MODEL_ID" "$BASE_STAGE" response_likelihood "$BASE_PROMPT_STYLE" "$template" "$(surface_base_like_dir "$template")" "$LIKELIHOOD_BATCH_SIZE" "$GPU_A"
      if [[ "$RUN_CHAT_TEMPLATE" == "1" ]]; then
        launch_surface "surface_chat_${template}_like" "${INSTRUCT_LABEL}_like" "$INSTRUCT_MODEL_ID" "$INSTRUCT_STAGE" response_likelihood "$INSTRUCT_PROMPT_STYLE" "$template" "$(surface_chat_like_dir "$template")" "$LIKELIHOOD_BATCH_SIZE" "$GPU_B"
      fi
      wait_wave

      if [[ "$RUN_PLAIN_CONTROL" == "1" ]]; then
        PIDS=()
        launch_surface "surface_plain_${template}_like" "${INSTRUCT_LABEL}_like" "$INSTRUCT_MODEL_ID" "$INSTRUCT_STAGE" response_likelihood "$PLAIN_INSTRUCT_PROMPT_STYLE" "$template" "$(surface_plain_like_dir "$template")" "$LIKELIHOOD_BATCH_SIZE" "$GPU_B"
        wait_wave
      fi
    fi
  done

  if [[ "$RUN_CHAT_TEMPLATE" == "1" ]]; then
    combine_surface_templates "chat" "$(surface_chat_forced_dir minimal)" "$(surface_chat_forced_dir standard)" "$(surface_chat_like_dir standard)"
  fi
  if [[ "$RUN_PLAIN_CONTROL" == "1" ]]; then
    combine_surface_templates "plain" "$(surface_plain_forced_dir minimal)" "$(surface_plain_forced_dir standard)" "$(surface_plain_like_dir standard)"
  fi
fi

echo "Gemma 2 27B overnight suite complete"
if [[ "$RUN_HLLM" == "1" ]]; then
  if [[ "$RUN_CHAT_TEMPLATE" == "1" ]]; then
    echo "hllm_chat_summary=$HLLM_SUMMARY_ROOT/d4_human_llm_template_sensitivity_${RUN_TAG}_hllm_chat"
  fi
  if [[ "$RUN_PLAIN_CONTROL" == "1" ]]; then
    echo "hllm_plain_summary=$HLLM_SUMMARY_ROOT/d4_human_llm_template_sensitivity_${RUN_TAG}_hllm_plain"
  fi
fi
if [[ "$RUN_SURFACE_BT" == "1" ]]; then
  if [[ "$RUN_CHAT_TEMPLATE" == "1" ]]; then
    echo "surface_chat_summary=$SURFACE_SUMMARY_ROOT/d4_bt_surface_stage_template_summary_${RUN_TAG}_chat"
  fi
  if [[ "$RUN_PLAIN_CONTROL" == "1" ]]; then
    echo "surface_plain_summary=$SURFACE_SUMMARY_ROOT/d4_bt_surface_stage_template_summary_${RUN_TAG}_plain"
  fi
fi
echo "logs=$LOG_DIR"
