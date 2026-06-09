#!/bin/bash
#SBATCH --job-name=judge-budget
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
COMPARISONS_JSONL="${COMPARISONS_JSONL:-$ARTROOT/data/derived/judge_deliberation_suite_v1/comparisons.jsonl}"
OUT_DIR="${OUT_DIR:-$ARTROOT/artifacts/mechanistic/judge_deliberation_budget_sweep_v1}"
MODEL_ID="${MODEL_ID:?Set MODEL_ID}"
RUN_LABEL="${RUN_LABEL:-$MODEL_ID}"
PROMPT_STYLE="${PROMPT_STYLE:-chat_template}"
BUDGET_TOKENS="${BUDGET_TOKENS:-0:128:256:512:1024:2048}"
BRANCHES_PER_COMPARISON="${BRANCHES_PER_COMPARISON:-3}"
MAX_PAIRS="${MAX_PAIRS:-0}"
CAP_STRATEGY="${CAP_STRATEGY:-source_round_robin}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_SCORE_LENGTH="${MAX_SCORE_LENGTH:-8192}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-4}"
MAX_NEW_TOKENS_DIRECT="${MAX_NEW_TOKENS_DIRECT:-16}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
SEED="${SEED:-1234}"
BUDGET_TOKENS="${BUDGET_TOKENS//:/,}"

cd "$WORKDIR"
mkdir -p "$HF_HOME" "$HF_HOME/transformers" "$HF_HOME/datasets"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:-$WORKDIR/src}"
export HF_HOME
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CMD=(
  python -m aisafety.scripts.run_judge_reasoning_budget_sweep
  --workspace-root "$WORKDIR"
  --comparisons-jsonl "$COMPARISONS_JSONL"
  --run-label "$RUN_LABEL"
  --model-id "$MODEL_ID"
  --cache-dir "$CACHE_DIR"
  --prompt-style "$PROMPT_STYLE"
  --budget-tokens "$BUDGET_TOKENS"
  --branches-per-comparison "$BRANCHES_PER_COMPARISON"
  --max-pairs "$MAX_PAIRS"
  --cap-strategy "$CAP_STRATEGY"
  --max-prompt-length "$MAX_PROMPT_LENGTH"
  --max-score-length "$MAX_SCORE_LENGTH"
  --score-batch-size "$SCORE_BATCH_SIZE"
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
if [[ "${RESUME:-0}" == "1" ]]; then
  CMD+=(--resume)
fi

"${CMD[@]}"
