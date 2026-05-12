#!/bin/bash
#SBATCH --job-name=d4-surf-cf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
ARTROOT="${ARTROOT:-$WORKDIR}"
PAIR_JSONL="${PAIR_JSONL:-$ARTROOT/data/derived/d4_human_llm_alignment_pairs_strat10k_v3/pairs.jsonl}"
OUT_DIR="${OUT_DIR:-$ARTROOT/data/derived/d4_surface_counterfactual_pairs_v1}"
AXES="${AXES:-structured_assistant_packaging,answer_likeness_packaging,formal_institutional_packaging,benefit_value_framing}"
MAX_PAIRS="${MAX_PAIRS:-0}"
MIN_TOKENS="${MIN_TOKENS:-20}"
MIN_LENGTH_RATIO="${MIN_LENGTH_RATIO:-0.7}"
MAX_LENGTH_RATIO="${MAX_LENGTH_RATIO:-1.3}"

cd "$WORKDIR"
export PYTHONUNBUFFERED=1

python -m aisafety.scripts.build_d4_surface_counterfactual_pairs \
  --workspace-root "$WORKDIR" \
  --pair-jsonl "$PAIR_JSONL" \
  --out-dir "$OUT_DIR" \
  --axes "$AXES" \
  --max-pairs "$MAX_PAIRS" \
  --min-tokens "$MIN_TOKENS" \
  --min-length-ratio "$MIN_LENGTH_RATIO" \
  --max-length-ratio "$MAX_LENGTH_RATIO"
