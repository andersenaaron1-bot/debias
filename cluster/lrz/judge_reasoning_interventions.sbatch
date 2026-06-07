#!/bin/bash
#SBATCH --job-name=judge-reason-intervene
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
CACHE_DIR="${CACHE_DIR:-$HF_HOME}"
TRACE_DIR="${TRACE_DIR:-$ARTROOT/artifacts/mechanistic/judge_reasoning_trajectories_v1}"
ANALYSIS_DIR="${ANALYSIS_DIR:-$ARTROOT/artifacts/mechanistic/judge_reasoning_analysis_v1}"
OUT_DIR="${OUT_DIR:-$ARTROOT/artifacts/mechanistic/judge_reasoning_interventions_v1}"
MODEL_ID="${MODEL_ID:?Set MODEL_ID}"
REASONING_MODE="${REASONING_MODE:-thinking}"
ANALYSIS_GROUP="${ANALYSIS_GROUP:-all}"
PROBE_TARGET="${PROBE_TARGET:-final_choice}"
HIDDEN_LAYER="${HIDDEN_LAYER:-0}"
POINT_INDEX="${POINT_INDEX:--1}"
ALPHAS="${ALPHAS:-0.5,1.0,2.0}"
RANDOM_CONTROL_SEEDS="${RANDOM_CONTROL_SEEDS:-2101,2102,2103}"
MAX_TRACES="${MAX_TRACES:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
GENERATION_MODE="${GENERATION_MODE:-greedy}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
SEED="${SEED:-1234}"
ALPHAS="${ALPHAS//:/,}"
RANDOM_CONTROL_SEEDS="${RANDOM_CONTROL_SEEDS//:/,}"

cd "$WORKDIR"
mkdir -p "$HF_HOME"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:-$WORKDIR/src}"
export HF_HOME
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

CMD=(
  python -m aisafety.scripts.run_judge_reasoning_interventions
  --workspace-root "$WORKDIR"
  --trace-dir "$TRACE_DIR"
  --analysis-dir "$ANALYSIS_DIR"
  --model-id "$MODEL_ID"
  --cache-dir "$CACHE_DIR"
  --reasoning-mode "$REASONING_MODE"
  --analysis-group "$ANALYSIS_GROUP"
  --probe-target "$PROBE_TARGET"
  --hidden-layer "$HIDDEN_LAYER"
  --point-index "$POINT_INDEX"
  --alphas "$ALPHAS"
  --random-control-seeds "$RANDOM_CONTROL_SEEDS"
  --max-traces "$MAX_TRACES"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --generation-mode "$GENERATION_MODE"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --top-k "$TOP_K"
  --seed "$SEED"
  --out-dir "$OUT_DIR"
)

if [[ "${USE_4BIT:-0}" == "1" ]]; then
  CMD+=(--use-4bit)
fi
if [[ "${INCLUDE_NEGATIVE:-1}" == "0" ]]; then
  CMD+=(--no-include-negative)
fi
if [[ "${ALLOW_ARTIFACT_MISMATCH:-0}" == "1" ]]; then
  CMD+=(--allow-artifact-mismatch)
fi

"${CMD[@]}"
