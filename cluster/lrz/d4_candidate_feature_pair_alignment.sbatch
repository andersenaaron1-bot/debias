#!/bin/bash
#SBATCH --job-name=d4-hllm-align
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
PAIR_JSONL="${PAIR_JSONL:-$ARTROOT/data/derived/d4_human_llm_alignment_pairs_v1/pairs.jsonl}"
CANDIDATE_SOURCE_DIR="${CANDIDATE_SOURCE_DIR:-$ARTROOT/artifacts/mechanistic/d4_j0_sae_merged_ontology_discovery_v1}"
REWARD_RUN_DIR="${REWARD_RUN_DIR:-$ARTROOT/artifacts/reward/j0_anchor_v1_h100compact}"
OUT_DIR="${OUT_DIR:-$ARTROOT/artifacts/mechanistic/d4_j0_human_llm_candidate_alignment_v1}"
SELECTED_LAYERS="${SELECTED_LAYERS:-}"
AGGREGATION="${AGGREGATION:-max}"
MAX_PAIRS="${MAX_PAIRS:-0}"
MAX_CANDIDATES="${MAX_CANDIDATES:-900}"
MAX_FEATURES_PER_LAYER="${MAX_FEATURES_PER_LAYER:-90}"
RANDOM_CONTROLS_PER_LAYER="${RANDOM_CONTROLS_PER_LAYER:-10}"
SOURCE_MIN_PAIRS="${SOURCE_MIN_PAIRS:-25}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-4}"
SAE_BATCH_SIZE="${SAE_BATCH_SIZE:-4}"
SAE_TOKEN_CHUNK_SIZE="${SAE_TOKEN_CHUNK_SIZE:-1024}"
MAX_LENGTH="${MAX_LENGTH:-512}"

cd "$WORKDIR"
mkdir -p "$HF_HOME"

export PYTHONUNBUFFERED=1
export HF_HOME
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

CMD=(
  python -m aisafety.scripts.run_d4_candidate_feature_pair_alignment
  --workspace-root "$WORKDIR"
  --pair-jsonl "$PAIR_JSONL"
  --candidate-source-dir "$CANDIDATE_SOURCE_DIR"
  --reward-run-dir "$REWARD_RUN_DIR"
  --aggregation "$AGGREGATION"
  --max-pairs "$MAX_PAIRS"
  --max-candidates "$MAX_CANDIDATES"
  --max-features-per-layer "$MAX_FEATURES_PER_LAYER"
  --random-controls-per-layer "$RANDOM_CONTROLS_PER_LAYER"
  --source-min-pairs "$SOURCE_MIN_PAIRS"
  --score-batch-size "$SCORE_BATCH_SIZE"
  --sae-batch-size "$SAE_BATCH_SIZE"
  --sae-token-chunk-size "$SAE_TOKEN_CHUNK_SIZE"
  --max-length "$MAX_LENGTH"
  --out-dir "$OUT_DIR"
)

if [[ -n "$SELECTED_LAYERS" ]]; then
  CMD+=(--selected-layers "$SELECTED_LAYERS")
fi
if [[ "${USE_4BIT:-0}" == "1" ]]; then
  CMD+=(--use-4bit)
fi
if [[ "${RESCORE_PAIRS:-0}" == "1" ]]; then
  CMD+=(--rescore-pairs)
fi
if [[ "${SKIP_MISSING_SAE:-0}" == "1" ]]; then
  CMD+=(--skip-missing-sae)
fi
if [[ -n "${ATOMS:-}" ]]; then
  CMD+=(--atoms "$ATOMS")
fi

"${CMD[@]}"
