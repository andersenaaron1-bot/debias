#!/bin/bash
#SBATCH --job-name=d4-surf-audit
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
BUNDLE_REGISTRY_DIR="${BUNDLE_REGISTRY_DIR:-$ARTROOT/artifacts/mechanistic/d4_j0_bundle_candidate_registry_v1}"
REWARD_RUN_DIR="${REWARD_RUN_DIR:-$ARTROOT/artifacts/reward/j0_anchor_v1_h100compact}"
OUT_DIR="${OUT_DIR:-$ARTROOT/artifacts/mechanistic/d4_j0_surface_counterfactual_audit_v1}"
BUNDLE_IDS="${BUNDLE_IDS:-structured_assistant_packaging,answer_likeness_packaging,formal_institutional_packaging,benefit_value_framing}"
MAX_COUNTERFACTUALS="${MAX_COUNTERFACTUALS:-0}"
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
  python -m aisafety.scripts.run_d4_surface_counterfactual_audit
  --workspace-root "$WORKDIR"
  --counterfactual-jsonl "$COUNTERFACTUAL_JSONL"
  --bundle-registry-dir "$BUNDLE_REGISTRY_DIR"
  --bundle-ids "$BUNDLE_IDS"
  --reward-run-dir "$REWARD_RUN_DIR"
  --max-counterfactuals "$MAX_COUNTERFACTUALS"
  --score-batch-size "$SCORE_BATCH_SIZE"
  --sae-batch-size "$SAE_BATCH_SIZE"
  --sae-token-chunk-size "$SAE_TOKEN_CHUNK_SIZE"
  --max-length "$MAX_LENGTH"
  --out-dir "$OUT_DIR"
)

if [[ "${SKIP_SAE:-0}" == "1" ]]; then
  CMD+=(--skip-sae)
fi
if [[ "${USE_4BIT:-0}" == "1" ]]; then
  CMD+=(--use-4bit)
fi

"${CMD[@]}"
