#!/bin/bash
#SBATCH --job-name=d4-feat-pert
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
PAIR_JSONL="${PAIR_JSONL:-$ARTROOT/data/derived/d4_human_llm_alignment_pairs_strat10k_v3/pairs.jsonl}"
BUNDLE_REGISTRY_DIR="${BUNDLE_REGISTRY_DIR:-$ARTROOT/artifacts/mechanistic/d4_j0_bundle_candidate_registry_v1}"
REWARD_RUN_DIR="${REWARD_RUN_DIR:-$ARTROOT/artifacts/reward/j0_anchor_v1_h100compact}"
OUT_DIR="${OUT_DIR:-$ARTROOT/artifacts/mechanistic/d4_j0_formal_bundle_feature_perturbation_scout_v1}"
BUNDLE_ID="${BUNDLE_ID:-formal_institutional_packaging}"
MAX_PAIRS="${MAX_PAIRS:-1200}"
MAX_FEATURES="${MAX_FEATURES:-0}"
DAMPING_STRENGTH="${DAMPING_STRENGTH:-0.5}"
HIGH_LOW_FRAC="${HIGH_LOW_FRAC:-0.25}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-4}"
SAE_BATCH_SIZE="${SAE_BATCH_SIZE:-4}"
SAE_TOKEN_CHUNK_SIZE="${SAE_TOKEN_CHUNK_SIZE:-1024}"
MAX_LENGTH="${MAX_LENGTH:-512}"
RANDOM_CONTROL_RANK="${RANDOM_CONTROL_RANK:-1}"

cd "$WORKDIR"
mkdir -p "$HF_HOME"

export PYTHONUNBUFFERED=1
export HF_HOME
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

CMD=(
  python -m aisafety.scripts.run_d4_feature_perturbation
  --workspace-root "$WORKDIR"
  --pair-jsonl "$PAIR_JSONL"
  --bundle-registry-dir "$BUNDLE_REGISTRY_DIR"
  --bundle-id "$BUNDLE_ID"
  --reward-run-dir "$REWARD_RUN_DIR"
  --max-pairs "$MAX_PAIRS"
  --max-features "$MAX_FEATURES"
  --damping-strength "$DAMPING_STRENGTH"
  --high-low-frac "$HIGH_LOW_FRAC"
  --score-batch-size "$SCORE_BATCH_SIZE"
  --sae-batch-size "$SAE_BATCH_SIZE"
  --sae-token-chunk-size "$SAE_TOKEN_CHUNK_SIZE"
  --max-length "$MAX_LENGTH"
  --random-control-rank "$RANDOM_CONTROL_RANK"
  --out-dir "$OUT_DIR"
)

if [[ "${INCLUDE_SOURCE_SENSITIVE:-0}" == "1" ]]; then
  CMD+=(--include-source-sensitive)
fi
if [[ "${SKIP_RANDOM_CONTROLS:-0}" == "1" ]]; then
  CMD+=(--skip-random-controls)
fi
if [[ "${USE_4BIT:-0}" == "1" ]]; then
  CMD+=(--use-4bit)
fi

"${CMD[@]}"
