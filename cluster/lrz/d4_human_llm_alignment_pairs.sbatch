#!/bin/bash
#SBATCH --job-name=d4-hllm-pairs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
ARTROOT="${ARTROOT:-$WORKDIR}"
RECORDS_JSONL="${RECORDS_JSONL:-$ARTROOT/data/derived/bundle_creation_corpus_v1/all_records.jsonl}"
OUT_DIR="${OUT_DIR:-$ARTROOT/data/derived/d4_human_llm_alignment_pairs_v1}"
INCLUDE_ROLES="${INCLUDE_ROLES:-discovery_core,controlled_confirmation,heldout_transfer}"
INCLUDE_DATASETS="${INCLUDE_DATASETS:-}"
EXCLUDE_DATASETS="${EXCLUDE_DATASETS:-}"
REQUIRE_DATASETS="${REQUIRE_DATASETS:-}"
CAP_STRATEGY="${CAP_STRATEGY:-dataset}"
MAX_PAIRS_PER_DATASET="${MAX_PAIRS_PER_DATASET:-5000}"
MAX_TOTAL_PAIRS="${MAX_TOTAL_PAIRS:-0}"
MAX_LLM_PER_GROUP="${MAX_LLM_PER_GROUP:-2}"
MIN_TOKENS="${MIN_TOKENS:-20}"
MAX_TOKENS="${MAX_TOKENS:-900}"
SEED="${SEED:-1234}"

cd "$WORKDIR"
export PYTHONUNBUFFERED=1

python -m aisafety.scripts.build_d4_human_llm_alignment_pairs \
  --workspace-root "$WORKDIR" \
  --records-jsonl "$RECORDS_JSONL" \
  --out-dir "$OUT_DIR" \
  --include-roles "$INCLUDE_ROLES" \
  --include-datasets "$INCLUDE_DATASETS" \
  --exclude-datasets "$EXCLUDE_DATASETS" \
  --require-datasets "$REQUIRE_DATASETS" \
  --cap-strategy "$CAP_STRATEGY" \
  --max-pairs-per-dataset "$MAX_PAIRS_PER_DATASET" \
  --max-total-pairs "$MAX_TOTAL_PAIRS" \
  --max-llm-per-group "$MAX_LLM_PER_GROUP" \
  --min-tokens "$MIN_TOKENS" \
  --max-tokens "$MAX_TOKENS" \
  --seed "$SEED"
