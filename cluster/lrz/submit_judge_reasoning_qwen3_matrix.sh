#!/bin/bash
# Submit a Qwen3 base/post-training judge-reasoning trajectory scout on LRZ.

set -euo pipefail

WORKDIR="${WORKDIR:-$HOME/debias}"
ARTROOT="${ARTROOT:-$WORKDIR}"
IMAGE="${IMAGE:-ghcr.io#andersenaaron1-bot/debias:sae-mech-v1}"
PARTS="${PARTS:-lrz-hgx-h100-94x4,lrz-dgx-a100-80x8,lrz-hgx-a100-80x4}"
HF_HOME="${HF_HOME:-$ARTROOT/.cache/huggingface}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
LOG_DIR="${LOG_DIR:-$ARTROOT/slurm_logs}"

RUN_TAG="${RUN_TAG:-judge_reasoning_qwen3_scout_v1}"
CONFIG="${CONFIG:-$WORKDIR/configs/datasets/judge_reasoning_suite_v1.json}"
PAIR_OUT_DIR="${PAIR_OUT_DIR:-$ARTROOT/data/derived/judge_reasoning_suite_${RUN_TAG}}"
COMPARISONS_JSONL="${COMPARISONS_JSONL:-$PAIR_OUT_DIR/comparisons.jsonl}"
OUT_ROOT="${OUT_ROOT:-$ARTROOT/artifacts/mechanistic/$RUN_TAG}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B-Base}"
IT_MODEL="${IT_MODEL:-Qwen/Qwen3-8B}"
SELECTED_LAYERS="${SELECTED_LAYERS:-4:8:12:16:20:24:28:32}"
MAX_PAIRS_PER_DATASET="${MAX_PAIRS_PER_DATASET:-80}"
MAX_PAIRS="${MAX_PAIRS:-60}"
BRANCHES_PER_COMPARISON="${BRANCHES_PER_COMPARISON:-3}"
TRAJECTORY_POINTS="${TRAJECTORY_POINTS:-17}"
MAX_NEW_TOKENS_THINKING="${MAX_NEW_TOKENS_THINKING:-256}"
MAX_NEW_TOKENS_DIRECT="${MAX_NEW_TOKENS_DIRECT:-32}"
USE_4BIT="${USE_4BIT:-0}"
COMPRESS_SHARDS="${COMPRESS_SHARDS:-1}"
PROBE_TARGETS="${PROBE_TARGETS:-final_choice:target_option:target_selected:condition_label:presentation_order}"
GROUP_COLUMNS="${GROUP_COLUMNS:-comparison_dimension:source_dataset:validity_type:difficulty_tier:analysis_split}"
CHOICE_CONFIDENCE_THRESHOLD="${CHOICE_CONFIDENCE_THRESHOLD:-0.8}"
TARGET_CONFIDENCE_THRESHOLD="${TARGET_CONFIDENCE_THRESHOLD:-0.8}"

mkdir -p "$LOG_DIR"

common_container_args=(
  --container-image="$IMAGE"
  --container-mounts="$WORKDIR:$WORKDIR,$ARTROOT:$ARTROOT,$ARTROOT:/workspace"
  --container-workdir="$WORKDIR"
)

pair_job="$(
  sbatch --parsable \
    --partition=lrz-cpu \
    --qos=cpu \
    --job-name=judge-reason-pairs \
    --cpus-per-task=2 \
    --mem=32G \
    --time=00:30:00 \
    --chdir="$WORKDIR" \
    --output="$LOG_DIR/%x-%j.out" \
    --error="$LOG_DIR/%x-%j.err" \
    "${common_container_args[@]}" \
    --container-env=PYTHONPATH \
    --export=ALL,WORKDIR="$WORKDIR",ARTROOT="$ARTROOT",PYTHONPATH="$WORKDIR/src",CONFIG="$CONFIG",OUT_DIR="$PAIR_OUT_DIR",MAX_PAIRS_PER_DATASET="$MAX_PAIRS_PER_DATASET",SKIP_MISSING=1 \
    cluster/lrz/judge_reasoning_suite_pairs.sbatch
)"
echo "pair_job=$pair_job"

trace_jobs=()
analysis_jobs=()
analysis_specs=()

submit_trace_and_analysis() {
  local label="$1"
  local model_id="$2"
  local prompt_style="$3"
  local reasoning_modes="$4"
  local trace_dir="$OUT_ROOT/trajectories_$label"
  local analysis_dir="$OUT_ROOT/analysis_$label"

  local trace_job
  trace_job="$(
    sbatch --parsable \
      --dependency=afterok:"$pair_job" \
      --partition="$PARTS" \
      --job-name="judge-trace-$label" \
      --gres=gpu:1 \
      --cpus-per-task=8 \
      --mem=160G \
      --time=12:00:00 \
      --chdir="$WORKDIR" \
      --output="$LOG_DIR/%x-%j.out" \
      --error="$LOG_DIR/%x-%j.err" \
      "${common_container_args[@]}" \
      --container-env=PYTHONPATH,HF_HOME,TRANSFORMERS_CACHE,HF_DATASETS_CACHE,HF_TOKEN,HUGGING_FACE_HUB_TOKEN \
      --export=ALL,WORKDIR="$WORKDIR",ARTROOT="$ARTROOT",PYTHONPATH="$WORKDIR/src",HF_HOME="$HF_HOME",TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE",HF_DATASETS_CACHE="$HF_DATASETS_CACHE",COMPARISONS_JSONL="$COMPARISONS_JSONL",MODEL_ID="$model_id",RUN_LABEL="$label",PROMPT_STYLE="$prompt_style",REASONING_MODES="$reasoning_modes",SELECTED_LAYERS="$SELECTED_LAYERS",BRANCHES_PER_COMPARISON="$BRANCHES_PER_COMPARISON",MAX_PAIRS="$MAX_PAIRS",TRAJECTORY_POINTS="$TRAJECTORY_POINTS",MAX_NEW_TOKENS_THINKING="$MAX_NEW_TOKENS_THINKING",MAX_NEW_TOKENS_DIRECT="$MAX_NEW_TOKENS_DIRECT",USE_4BIT="$USE_4BIT",COMPRESS_SHARDS="$COMPRESS_SHARDS",OUT_DIR="$trace_dir" \
      cluster/lrz/judge_reasoning_trajectories.sbatch
  )"
  trace_jobs+=("$trace_job")
  echo "trace_$label=$trace_job"

  local analysis_job
  analysis_job="$(
    sbatch --parsable \
      --dependency=afterok:"$trace_job" \
      --partition=lrz-cpu \
      --qos=cpu \
      --job-name="judge-analyze-$label" \
      --cpus-per-task=8 \
      --mem=96G \
      --time=04:00:00 \
      --chdir="$WORKDIR" \
      --output="$LOG_DIR/%x-%j.out" \
      --error="$LOG_DIR/%x-%j.err" \
      "${common_container_args[@]}" \
      --container-env=PYTHONPATH \
      --export=ALL,WORKDIR="$WORKDIR",ARTROOT="$ARTROOT",PYTHONPATH="$WORKDIR/src",TRACE_DIR="$trace_dir",OUT_DIR="$analysis_dir",PROBE_TARGETS="$PROBE_TARGETS",GROUP_COLUMNS="$GROUP_COLUMNS",CHOICE_CONFIDENCE_THRESHOLD="$CHOICE_CONFIDENCE_THRESHOLD",TARGET_CONFIDENCE_THRESHOLD="$TARGET_CONFIDENCE_THRESHOLD" \
      cluster/lrz/judge_reasoning_analysis.sbatch
  )"
  analysis_jobs+=("$analysis_job")
  analysis_specs+=("$label=$analysis_dir")
  echo "analysis_$label=$analysis_job"
}

submit_trace_and_analysis "qwen3_8b_base" "$BASE_MODEL" "plain" "free_reasoning:direct"
submit_trace_and_analysis "qwen3_8b_it" "$IT_MODEL" "chat_template" "thinking:direct"

analysis_dependency="$(IFS=:; echo "${analysis_jobs[*]}")"
analysis_arg="$(IFS='|'; echo "${analysis_specs[*]}")"
summary_job="$(
  sbatch --parsable \
    --dependency=afterok:"$analysis_dependency" \
    --partition=lrz-cpu \
    --qos=cpu \
    --job-name=judge-reason-summary \
    --cpus-per-task=2 \
    --mem=16G \
    --time=00:30:00 \
    --chdir="$WORKDIR" \
    --output="$LOG_DIR/%x-%j.out" \
    --error="$LOG_DIR/%x-%j.err" \
    "${common_container_args[@]}" \
    --container-env=PYTHONPATH \
    --export=ALL,WORKDIR="$WORKDIR",ARTROOT="$ARTROOT",PYTHONPATH="$WORKDIR/src",ANALYSIS_SPECS="$analysis_arg",OUT_DIR="$OUT_ROOT/summary" \
    cluster/lrz/judge_reasoning_summary.sbatch
)"

echo "summary_job=$summary_job"
echo "out_root=$OUT_ROOT"
