#!/bin/bash
#SBATCH --job-name=judge-reason-trace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
ARTROOT="${ARTROOT:-$WORKDIR}"
HF_HOME="${HF_HOME:-$ARTROOT/.cache/huggingface}"
CACHE_DIR="${CACHE_DIR:-$HF_HOME}"
COMPARISONS_JSONL="${COMPARISONS_JSONL:-$ARTROOT/data/derived/judge_reasoning_suite_v1/comparisons.jsonl}"
OUT_DIR="${OUT_DIR:-$ARTROOT/artifacts/mechanistic/judge_reasoning_trajectories_v1}"
MODEL_ID="${MODEL_ID:?Set MODEL_ID}"
RUN_LABEL="${RUN_LABEL:-$MODEL_ID}"
PROMPT_STYLE="${PROMPT_STYLE:-chat_template}"
REASONING_MODES="${REASONING_MODES:-thinking,direct}"
SELECTED_LAYERS="${SELECTED_LAYERS:-}"
LAYER_STRIDE="${LAYER_STRIDE:-6}"
TAIL_LAYERS="${TAIL_LAYERS:-2}"
BRANCHES_PER_COMPARISON="${BRANCHES_PER_COMPARISON:-4}"
MAX_PAIRS="${MAX_PAIRS:-200}"
CAP_STRATEGY="${CAP_STRATEGY:-source_round_robin}"
TRAJECTORY_POINTS="${TRAJECTORY_POINTS:-17}"
SHARD_SIZE="${SHARD_SIZE:-32}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_NEW_TOKENS_THINKING="${MAX_NEW_TOKENS_THINKING:-512}"
MAX_NEW_TOKENS_DIRECT="${MAX_NEW_TOKENS_DIRECT:-32}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
SEED="${SEED:-1234}"
REASONING_MODES="${REASONING_MODES//:/,}"
SELECTED_LAYERS="${SELECTED_LAYERS//:/,}"

cd "$WORKDIR"
mkdir -p "$HF_HOME"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:-$WORKDIR/src}"
export HF_HOME
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

CMD=(
  python -m aisafety.scripts.run_judge_reasoning_trajectories
  --workspace-root "$WORKDIR"
  --comparisons-jsonl "$COMPARISONS_JSONL"
  --run-label "$RUN_LABEL"
  --model-id "$MODEL_ID"
  --cache-dir "$CACHE_DIR"
  --prompt-style "$PROMPT_STYLE"
  --reasoning-modes "$REASONING_MODES"
  --branches-per-comparison "$BRANCHES_PER_COMPARISON"
  --max-pairs "$MAX_PAIRS"
  --cap-strategy "$CAP_STRATEGY"
  --selected-layers "$SELECTED_LAYERS"
  --layer-stride "$LAYER_STRIDE"
  --tail-layers "$TAIL_LAYERS"
  --trajectory-points "$TRAJECTORY_POINTS"
  --shard-size "$SHARD_SIZE"
  --max-prompt-length "$MAX_PROMPT_LENGTH"
  --max-new-tokens-thinking "$MAX_NEW_TOKENS_THINKING"
  --max-new-tokens-direct "$MAX_NEW_TOKENS_DIRECT"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --top-k "$TOP_K"
  --seed "$SEED"
  --out-dir "$OUT_DIR"
)

if [[ "${USE_4BIT:-0}" == "1" ]]; then
  CMD+=(--use-4bit)
fi
if [[ "${COMPRESS_SHARDS:-0}" == "1" ]]; then
  CMD+=(--compress-shards)
fi
if [[ "${RESUME:-0}" == "1" ]]; then
  CMD+=(--resume)
fi

"${CMD[@]}"
