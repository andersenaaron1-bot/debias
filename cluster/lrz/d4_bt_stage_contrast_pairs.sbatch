#!/bin/bash
#SBATCH --job-name=d4-bt-pairs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
ARTROOT="${ARTROOT:-$WORKDIR}"
COUNTERFACTUAL_JSONL="${COUNTERFACTUAL_JSONL:-$ARTROOT/data/derived/d4_surface_counterfactual_pairs_v1/counterfactuals.jsonl}"
OUT_DIR="${OUT_DIR:-$ARTROOT/data/derived/d4_bt_stage_contrast_pairs_v1}"
AXES="${AXES:-}"
DIRECTIONS="${DIRECTIONS:-}"
ROLES="${ROLES:-}"
MAX_COUNTERFACTUALS="${MAX_COUNTERFACTUALS:-0}"
INCLUDE_ORDER_SWAPS="${INCLUDE_ORDER_SWAPS:-1}"

cd "$WORKDIR"
export PYTHONUNBUFFERED=1

CMD=(
  python -m aisafety.scripts.build_d4_bt_stage_contrast_pairs
  --workspace-root "$WORKDIR"
  --counterfactual-jsonl "$COUNTERFACTUAL_JSONL"
  --out-dir "$OUT_DIR"
  --axes "$AXES"
  --directions "$DIRECTIONS"
  --roles "$ROLES"
  --max-counterfactuals "$MAX_COUNTERFACTUALS"
)

if [[ "$INCLUDE_ORDER_SWAPS" == "1" ]]; then
  CMD+=(--include-order-swaps)
else
  CMD+=(--no-include-order-swaps)
fi

"${CMD[@]}"
