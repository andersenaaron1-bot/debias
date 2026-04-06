#!/bin/bash
#SBATCH --job-name=aisafety-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

CONFIG_PATH="${1:?usage: sbatch ... train_from_config.sbatch /workspace/configs/experiments/m2_full_v1.json}"
WORKDIR="${WORKDIR:-/workspace}"
HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

cd "$WORKDIR"
mkdir -p "$HF_HOME"

export PYTHONUNBUFFERED=1
export HF_HOME
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

# Force a clean single-process distributed setup for Accelerate on LRZ/Pyxis.
# Some Slurm environments expose partial rendezvous variables, which causes
# torch.distributed env:// initialization to fail when WORLD_SIZE is missing.
export WORLD_SIZE="${WORLD_SIZE:-1}"
export RANK="${RANK:-0}"
export LOCAL_RANK="${LOCAL_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT_DEFAULT=29500
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  MASTER_PORT_DEFAULT=$((20000 + (SLURM_JOB_ID % 20000)))
fi
export MASTER_PORT="${MASTER_PORT:-$MASTER_PORT_DEFAULT}"

python -m aisafety.scripts.run_experiment_config \
  --config "$CONFIG_PATH" \
  --workspace-root "$WORKDIR"
