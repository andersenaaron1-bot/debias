#!/bin/bash
#SBATCH --job-name=d4-readout-null
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
COUNTERFACTUAL_JSONL="${COUNTERFACTUAL_JSONL:-$ARTROOT/data/derived/d4_surface_counterfactual_pairs_v1/counterfactuals.jsonl}"
PAIR_JSONL="${PAIR_JSONL:-$ARTROOT/data/derived/d4_human_llm_alignment_pairs_strat10k_v3/pairs.jsonl}"
PREF_VAL_JSONL="${PREF_VAL_JSONL:-$ARTROOT/data/derived/pref_pairs_shp2/pref_pairs_val.jsonl}"
REWARD_RUN_DIR="${REWARD_RUN_DIR:-$ARTROOT/artifacts/reward/j0_anchor_v1_h100compact}"
OUT_DIR="${OUT_DIR:-$ARTROOT/artifacts/mechanistic/d4_j0_readout_surface_nulling_v1}"
AXES="${AXES:-structured_assistant_packaging,formal_institutional_packaging,benefit_value_framing}"
FIT_FRAC="${FIT_FRAC:-0.5}"
MIN_DIRECTION_ROWS="${MIN_DIRECTION_ROWS:-20}"
MAX_COUNTERFACTUALS="${MAX_COUNTERFACTUALS:-0}"
MAX_PAIRS="${MAX_PAIRS:-3000}"
MAX_PREF_PAIRS="${MAX_PREF_PAIRS:-1000}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-4}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-4}"
MAX_LENGTH="${MAX_LENGTH:-512}"

cd "$WORKDIR"
mkdir -p "$HF_HOME"

export PYTHONUNBUFFERED=1
export HF_HOME
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

CMD=(
  python -m aisafety.scripts.run_d4_readout_surface_nulling
  --workspace-root "$WORKDIR"
  --counterfactual-jsonl "$COUNTERFACTUAL_JSONL"
  --pair-jsonl "$PAIR_JSONL"
  --pref-val-jsonl "$PREF_VAL_JSONL"
  --axes "$AXES"
  --reward-run-dir "$REWARD_RUN_DIR"
  --fit-frac "$FIT_FRAC"
  --min-direction-rows "$MIN_DIRECTION_ROWS"
  --max-counterfactuals "$MAX_COUNTERFACTUALS"
  --max-pairs "$MAX_PAIRS"
  --max-pref-pairs "$MAX_PREF_PAIRS"
  --score-batch-size "$SCORE_BATCH_SIZE"
  --encode-batch-size "$ENCODE_BATCH_SIZE"
  --max-length "$MAX_LENGTH"
  --out-dir "$OUT_DIR"
)

if [[ "${SKIP_PREF_RETENTION:-0}" == "1" ]]; then
  CMD+=(--skip-pref-retention)
fi
if [[ "${USE_4BIT:-0}" == "1" ]]; then
  CMD+=(--use-4bit)
fi

"${CMD[@]}"
