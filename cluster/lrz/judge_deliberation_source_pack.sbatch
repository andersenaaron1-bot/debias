#!/bin/bash
#SBATCH --job-name=judge-delib-sources
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
ARTROOT="${ARTROOT:-$WORKDIR}"
HF_HOME="${HF_HOME:-$ARTROOT/.cache/huggingface}"
BASE_SOURCE_DIR="${BASE_SOURCE_DIR:-$ARTROOT/data/derived/judge_reasoning_sources_v1}"
OUT_DIR="${OUT_DIR:-$ARTROOT/data/derived/judge_deliberation_sources_v1}"
MAX_PAIRS_PER_DATASET="${MAX_PAIRS_PER_DATASET:-200}"
SEED="${SEED:-1234}"

cd "$WORKDIR"
mkdir -p "$HF_HOME" "$HF_HOME/datasets"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:-$WORKDIR/src}"
export HF_HOME
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

python -m aisafety.scripts.build_judge_deliberation_source_pack \
  --workspace-root "$WORKDIR" \
  --artifact-root "$ARTROOT" \
  --base-source-dir "$BASE_SOURCE_DIR" \
  --cache-dir "$HF_DATASETS_CACHE" \
  --max-pairs-per-dataset "$MAX_PAIRS_PER_DATASET" \
  --seed "$SEED" \
  --out-dir "$OUT_DIR"
