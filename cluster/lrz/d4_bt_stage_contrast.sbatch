#!/bin/bash
#SBATCH --job-name=d4-bt-stage
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
BT_PAIRS_JSONL="${BT_PAIRS_JSONL:-$ARTROOT/data/derived/d4_bt_stage_contrast_pairs_v1/bt_pairs.jsonl}"
OUT_DIR="${OUT_DIR:-$ARTROOT/artifacts/mechanistic/d4_bt_stage_contrast_v1}"
STAGE="${STAGE:-it_lm}"
MODEL_ID="${MODEL_ID:-google/gemma-2-9b-it}"
REWARD_RUN_DIR="${REWARD_RUN_DIR:-$ARTROOT/artifacts/reward/j0_anchor_v1_h100compact}"
PROMPT_STYLE="${PROMPT_STYLE:-plain}"
MAX_PAIRS="${MAX_PAIRS:-0}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-4}"
MAX_LENGTH="${MAX_LENGTH:-2048}"

cd "$WORKDIR"
mkdir -p "$HF_HOME"

export PYTHONUNBUFFERED=1
export HF_HOME
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

CMD=(
  python -m aisafety.scripts.run_d4_bt_stage_contrast
  --workspace-root "$WORKDIR"
  --bt-pairs-jsonl "$BT_PAIRS_JSONL"
  --stage "$STAGE"
  --model-id "$MODEL_ID"
  --reward-run-dir "$REWARD_RUN_DIR"
  --prompt-style "$PROMPT_STYLE"
  --max-pairs "$MAX_PAIRS"
  --score-batch-size "$SCORE_BATCH_SIZE"
  --max-length "$MAX_LENGTH"
  --out-dir "$OUT_DIR"
)

if [[ "${USE_4BIT:-0}" == "1" ]]; then
  CMD+=(--use-4bit)
fi

"${CMD[@]}"
