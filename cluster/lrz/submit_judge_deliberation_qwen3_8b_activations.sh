#!/bin/bash
# Submit the post-budget Qwen3-8B activation capture and fixed-decoder analysis.

set -euo pipefail

WORKDIR="${WORKDIR:-$HOME/debias}"
ARTROOT="${ARTROOT:-$WORKDIR}"
IMAGE="${IMAGE:-ghcr.io#andersenaaron1-bot/debias:sae-mech-v1}"
PARTS="${PARTS:-lrz-hgx-h100-94x4,lrz-dgx-a100-80x8,lrz-hgx-a100-80x4}"
HF_HOME="${HF_HOME:-$ARTROOT/.cache/huggingface}"
LOG_DIR="${LOG_DIR:-$ARTROOT/slurm_logs}"
RUN_TAG="${RUN_TAG:-judge_deliberation_qwen3_8b_activations_v1}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
SUITE_DIR="${SUITE_DIR:?Set SUITE_DIR to the frozen 8B comparison suite}"
OUT_ROOT="${OUT_ROOT:-$ARTROOT/artifacts/mechanistic/$RUN_TAG}"
SELECTED_LAYERS="${SELECTED_LAYERS:-4:8:12:16:20:24:28:32}"
BRANCHES_PER_COMPARISON="${BRANCHES_PER_COMPARISON:-3}"
MAX_PAIRS="${MAX_PAIRS:-0}"
MAX_NEW_TOKENS_THINKING="${MAX_NEW_TOKENS_THINKING:-2048}"
TRAJECTORY_POINTS="${TRAJECTORY_POINTS:-33}"
SEED="${SEED:-1234}"

mkdir -p "$LOG_DIR"
container_args=(
  --container-image="$IMAGE"
  --container-mounts="$WORKDIR:$WORKDIR,$ARTROOT:$ARTROOT,$ARTROOT:/workspace"
  --container-workdir="$WORKDIR"
)

trace_job="$(
  sbatch --parsable \
    --partition="$PARTS" \
    --job-name=judge-delib-trace \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=160G \
    --time=24:00:00 \
    --chdir="$WORKDIR" \
    --output="$LOG_DIR/%x-%j.out" \
    --error="$LOG_DIR/%x-%j.err" \
    "${container_args[@]}" \
    --container-env=PYTHONPATH,HF_HOME,TRANSFORMERS_CACHE,HF_DATASETS_CACHE,HF_TOKEN,HUGGING_FACE_HUB_TOKEN \
    --export=ALL,WORKDIR="$WORKDIR",ARTROOT="$ARTROOT",PYTHONPATH="$WORKDIR/src",HF_HOME="$HF_HOME",TRANSFORMERS_CACHE="$HF_HOME/transformers",HF_DATASETS_CACHE="$HF_HOME/datasets",COMPARISONS_JSONL="$SUITE_DIR/comparisons.jsonl",OUT_DIR="$OUT_ROOT/trajectories_qwen3_8b_it",MODEL_ID="$MODEL_ID",RUN_LABEL=qwen3_8b_it,PROMPT_STYLE=chat_template,REASONING_MODES=thinking,SELECTED_LAYERS="$SELECTED_LAYERS",BRANCHES_PER_COMPARISON="$BRANCHES_PER_COMPARISON",MAX_PAIRS="$MAX_PAIRS",TRAJECTORY_POINTS="$TRAJECTORY_POINTS",MAX_NEW_TOKENS_THINKING="$MAX_NEW_TOKENS_THINKING",COMPRESS_SHARDS=1,SEED="$SEED" \
    cluster/lrz/judge_reasoning_trajectories.sbatch
)"
echo "trace_job=$trace_job"

decoder_job="$(
  sbatch --parsable \
    --dependency=afterok:"$trace_job" \
    --partition=lrz-cpu \
    --qos=cpu \
    --job-name=judge-delib-fixed \
    --cpus-per-task=8 \
    --mem=160G \
    --time=08:00:00 \
    --chdir="$WORKDIR" \
    --output="$LOG_DIR/%x-%j.out" \
    --error="$LOG_DIR/%x-%j.err" \
    "${container_args[@]}" \
    --container-env=PYTHONPATH \
    --export=ALL,WORKDIR="$WORKDIR",ARTROOT="$ARTROOT",PYTHONPATH="$WORKDIR/src",TRACE_DIR="$OUT_ROOT/trajectories_qwen3_8b_it",OUT_DIR="$OUT_ROOT/fixed_decoders",REASONING_MODE=thinking,ANCHOR_TARGET=final_choice,PROBE_TARGETS=final_choice:target_option:presentation_order,SEED="$SEED" \
    cluster/lrz/judge_reasoning_fixed_decoders.sbatch
)"
echo "decoder_job=$decoder_job"
echo "out_root=$OUT_ROOT"
