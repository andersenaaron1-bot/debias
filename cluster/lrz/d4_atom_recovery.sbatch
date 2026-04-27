#!/bin/bash
#SBATCH --job-name=aisafety-d4-recovery
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
HF_HOME="${HF_HOME:-$WORKDIR/.cache/huggingface}"
MANIFEST_JSON="${MANIFEST_JSON:-data/derived/d4_dataset_pack_v1/manifest.json}"
REWARD_RUN_DIR="${REWARD_RUN_DIR:-artifacts/reward/j0_anchor_v1_h100compact}"
OUT_DIR="${OUT_DIR:-artifacts/mechanistic/d4_j0_atom_recovery_v1_resplit}"
CONTENT_MAX_PAIRS="${CONTENT_MAX_PAIRS:-4000}"
MODE="${MODE:-content-anchor-only}"

cd "$WORKDIR"
mkdir -p "$HF_HOME"

export PYTHONUNBUFFERED=1
export HF_HOME
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

CMD=(
  python -m aisafety.scripts.run_d4_atom_recovery
  --workspace-root "$WORKDIR"
  --manifest-json "$MANIFEST_JSON"
  --reward-run-dir "$REWARD_RUN_DIR"
  --content-max-pairs "$CONTENT_MAX_PAIRS"
  --out-dir "$OUT_DIR"
)

if [[ "$MODE" == "content-anchor-only" ]]; then
  CMD+=(--content-anchor-only)
fi
if [[ "${USE_4BIT:-0}" == "1" ]]; then
  CMD+=(--use-4bit)
fi

"${CMD[@]}"

