#!/bin/bash
# Submit the Qwen3-8B deliberation source, suite, budget, and analysis jobs.

set -euo pipefail

WORKDIR="${WORKDIR:-$HOME/debias}"
ARTROOT="${ARTROOT:-$WORKDIR}"
IMAGE="${IMAGE:-ghcr.io#andersenaaron1-bot/debias:sae-mech-v1}"
PARTS="${PARTS:-lrz-hgx-h100-94x4,lrz-dgx-a100-80x8,lrz-hgx-a100-80x4}"
HF_HOME="${HF_HOME:-$ARTROOT/.cache/huggingface}"
LOG_DIR="${LOG_DIR:-$ARTROOT/slurm_logs}"
RUN_TAG="${RUN_TAG:-judge_deliberation_qwen3_8b_budget_scout_v1}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
BASE_SOURCE_DIR="${BASE_SOURCE_DIR:-$ARTROOT/data/derived/judge_reasoning_sources_v1}"
SOURCE_DIR="${SOURCE_DIR:-$ARTROOT/data/derived/judge_deliberation_sources_v1}"
SUITE_DIR="${SUITE_DIR:-$ARTROOT/data/derived/judge_deliberation_suite_$RUN_TAG}"
RUN_DIR="${RUN_DIR:-$ARTROOT/artifacts/mechanistic/$RUN_TAG}"
CONFIG="${CONFIG:-$WORKDIR/configs/datasets/judge_deliberation_progression_v1.json}"
MAX_SOURCE_PAIRS="${MAX_SOURCE_PAIRS:-200}"
MAX_PAIRS_PER_DATASET="${MAX_PAIRS_PER_DATASET:-30}"
BRANCHES_PER_COMPARISON="${BRANCHES_PER_COMPARISON:-3}"
BUDGET_TOKENS="${BUDGET_TOKENS:-0:128:256:512:1024:2048}"
BOOTSTRAP="${BOOTSTRAP:-5000}"
USE_4BIT="${USE_4BIT:-0}"
RESUME="${RESUME:-0}"
SEED="${SEED:-1234}"

mkdir -p "$LOG_DIR"

container_args=(
  --container-image="$IMAGE"
  --container-mounts="$WORKDIR:$WORKDIR,$ARTROOT:$ARTROOT,$ARTROOT:/workspace"
  --container-workdir="$WORKDIR"
)

source_job="$(
  sbatch --parsable \
    --partition=lrz-cpu \
    --qos=cpu \
    --job-name=judge-delib-sources \
    --cpus-per-task=4 \
    --mem=64G \
    --time=02:00:00 \
    --chdir="$WORKDIR" \
    --output="$LOG_DIR/%x-%j.out" \
    --error="$LOG_DIR/%x-%j.err" \
    "${container_args[@]}" \
    --container-env=PYTHONPATH,HF_HOME,HF_DATASETS_CACHE,HF_TOKEN,HUGGING_FACE_HUB_TOKEN \
    --export=ALL,WORKDIR="$WORKDIR",ARTROOT="$ARTROOT",PYTHONPATH="$WORKDIR/src",HF_HOME="$HF_HOME",HF_DATASETS_CACHE="$HF_HOME/datasets",BASE_SOURCE_DIR="$BASE_SOURCE_DIR",OUT_DIR="$SOURCE_DIR",MAX_PAIRS_PER_DATASET="$MAX_SOURCE_PAIRS",SEED="$SEED" \
    cluster/lrz/judge_deliberation_source_pack.sbatch
)"
echo "source_job=$source_job"

pair_job="$(
  sbatch --parsable \
    --dependency=afterok:"$source_job" \
    --partition=lrz-cpu \
    --qos=cpu \
    --job-name=judge-delib-pairs \
    --cpus-per-task=2 \
    --mem=32G \
    --time=00:30:00 \
    --chdir="$WORKDIR" \
    --output="$LOG_DIR/%x-%j.out" \
    --error="$LOG_DIR/%x-%j.err" \
    "${container_args[@]}" \
    --container-env=PYTHONPATH \
    --export=ALL,WORKDIR="$WORKDIR",ARTROOT="$ARTROOT",PYTHONPATH="$WORKDIR/src",CONFIG="$CONFIG",OUT_DIR="$SUITE_DIR",MAX_PAIRS_PER_DATASET="$MAX_PAIRS_PER_DATASET",SEED="$SEED" \
    cluster/lrz/judge_reasoning_suite_pairs.sbatch
)"
echo "pair_job=$pair_job"

budget_job="$(
  sbatch --parsable \
    --dependency=afterok:"$pair_job" \
    --partition="$PARTS" \
    --job-name=judge-delib-budget \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=160G \
    --time=24:00:00 \
    --chdir="$WORKDIR" \
    --output="$LOG_DIR/%x-%j.out" \
    --error="$LOG_DIR/%x-%j.err" \
    "${container_args[@]}" \
    --container-env=PYTHONPATH,HF_HOME,TRANSFORMERS_CACHE,HF_DATASETS_CACHE,HF_TOKEN,HUGGING_FACE_HUB_TOKEN \
    --export=ALL,WORKDIR="$WORKDIR",ARTROOT="$ARTROOT",PYTHONPATH="$WORKDIR/src",HF_HOME="$HF_HOME",TRANSFORMERS_CACHE="$HF_HOME/transformers",HF_DATASETS_CACHE="$HF_HOME/datasets",COMPARISONS_JSONL="$SUITE_DIR/comparisons.jsonl",OUT_DIR="$RUN_DIR/budget_sweep",MODEL_ID="$MODEL_ID",RUN_LABEL=qwen3_8b_it,BUDGET_TOKENS="$BUDGET_TOKENS",BRANCHES_PER_COMPARISON="$BRANCHES_PER_COMPARISON",MAX_PAIRS=0,USE_4BIT="$USE_4BIT",RESUME="$RESUME",SEED="$SEED" \
    cluster/lrz/judge_reasoning_budget_sweep.sbatch
)"
echo "budget_job=$budget_job"

analysis_job="$(
  sbatch --parsable \
    --dependency=afterok:"$budget_job" \
    --partition=lrz-cpu \
    --qos=cpu \
    --job-name=judge-delib-budget-analysis \
    --cpus-per-task=4 \
    --mem=64G \
    --time=03:00:00 \
    --chdir="$WORKDIR" \
    --output="$LOG_DIR/%x-%j.out" \
    --error="$LOG_DIR/%x-%j.err" \
    "${container_args[@]}" \
    --container-env=PYTHONPATH \
    --export=ALL,WORKDIR="$WORKDIR",ARTROOT="$ARTROOT",PYTHONPATH="$WORKDIR/src",RUN_DIR="$RUN_DIR/budget_sweep",OUT_DIR="$RUN_DIR/budget_analysis",BOOTSTRAP="$BOOTSTRAP",SEED="$SEED" \
    cluster/lrz/judge_reasoning_budget_analysis.sbatch
)"
echo "analysis_job=$analysis_job"
echo "run_dir=$RUN_DIR"
