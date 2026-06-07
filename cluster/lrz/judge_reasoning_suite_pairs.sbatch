#!/bin/bash
#SBATCH --job-name=judge-reason-pairs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
ARTROOT="${ARTROOT:-$WORKDIR}"
CONFIG="${CONFIG:-$WORKDIR/configs/datasets/judge_reasoning_suite_v1.json}"
OUT_DIR="${OUT_DIR:-$ARTROOT/data/derived/judge_reasoning_suite_v1}"
MAX_PAIRS_PER_DATASET="${MAX_PAIRS_PER_DATASET:-0}"
SEED="${SEED:-1234}"

cd "$WORKDIR"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:-$WORKDIR/src}"

CMD=(
  python -m aisafety.scripts.build_judge_reasoning_suite
  --workspace-root "$WORKDIR"
  --input-root "$ARTROOT"
  --config "$CONFIG"
  --max-pairs-per-dataset "$MAX_PAIRS_PER_DATASET"
  --seed "$SEED"
  --out-dir "$OUT_DIR"
)

if [[ "${SKIP_MISSING:-0}" == "1" ]]; then
  CMD+=(--skip-missing)
fi

"${CMD[@]}"
