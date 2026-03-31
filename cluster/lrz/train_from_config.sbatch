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

python -m aisafety.scripts.run_experiment_config \
  --config "$CONFIG_PATH" \
  --workspace-root "$WORKDIR"
