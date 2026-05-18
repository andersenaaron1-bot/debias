#!/bin/bash
#SBATCH --job-name=d4-hllm-stage-pairs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
ARTROOT="${ARTROOT:-$WORKDIR}"
PAIR_JSONL="${PAIR_JSONL:-$ARTROOT/data/derived/d4_human_llm_alignment_pairs_v1/pairs.jsonl}"
OUT_DIR="${OUT_DIR:-$ARTROOT/data/derived/d4_human_llm_stage_contrast_pairs_v1}"
INCLUDE_DATASETS="${INCLUDE_DATASETS:-}"
EXCLUDE_DATASETS="${EXCLUDE_DATASETS:-}"
INCLUDE_SUBSETS="${INCLUDE_SUBSETS:-}"
INCLUDE_ITEM_TYPES="${INCLUDE_ITEM_TYPES:-}"
MAX_PAIRS="${MAX_PAIRS:-0}"
INCLUDE_ORDER_SWAPS="${INCLUDE_ORDER_SWAPS:-1}"
SEED="${SEED:-1234}"

cd "$WORKDIR"
export PYTHONUNBUFFERED=1

CMD=(
  python -m aisafety.scripts.build_d4_human_llm_stage_contrast_pairs
  --workspace-root "$WORKDIR"
  --pair-jsonl "$PAIR_JSONL"
  --out-dir "$OUT_DIR"
  --include-datasets "$INCLUDE_DATASETS"
  --exclude-datasets "$EXCLUDE_DATASETS"
  --include-subsets "$INCLUDE_SUBSETS"
  --include-item-types "$INCLUDE_ITEM_TYPES"
  --max-pairs "$MAX_PAIRS"
  --seed "$SEED"
)

if [[ "$INCLUDE_ORDER_SWAPS" == "1" ]]; then
  CMD+=(--include-order-swaps)
else
  CMD+=(--no-include-order-swaps)
fi

"${CMD[@]}"
