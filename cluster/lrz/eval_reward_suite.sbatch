#!/bin/bash
#SBATCH --job-name=aisafety-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

RUN_DIR="${1:?usage: sbatch ... eval_reward_suite.sbatch /workspace/outputs/m2_full_v1}"
OUTPUT_DIR="${2:-$RUN_DIR/eval}"
WORKDIR="${WORKDIR:-/workspace}"
HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

cd "$WORKDIR"
mkdir -p "$HF_HOME"

export PYTHONUNBUFFERED=1
export HF_HOME
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

bash "$WORKDIR/cluster/lrz/eval_reward_suite.sh" "$RUN_DIR" "$OUTPUT_DIR"
