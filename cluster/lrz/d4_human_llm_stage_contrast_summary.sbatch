#!/bin/bash
#SBATCH --job-name=d4-hllm-stage-sum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
ARTROOT="${ARTROOT:-$WORKDIR}"
RUNS="${RUNS:-}"
CONTRASTS="${CONTRASTS:-}"
OUT_DIR="${OUT_DIR:-$ARTROOT/artifacts/mechanistic/d4_human_llm_stage_contrast_summary_v1}"

cd "$WORKDIR"
export PYTHONUNBUFFERED=1

if [[ -z "$RUNS" ]]; then
  echo "RUNS must be a colon-separated list of LABEL=DIR_OR_CSV entries." >&2
  exit 2
fi

RUN_ARGS=()
IFS=':' read -ra RUN_ITEMS <<< "$RUNS"
for item in "${RUN_ITEMS[@]}"; do
  if [[ -n "$item" ]]; then
    RUN_ARGS+=(--run "$item")
  fi
done

CONTRAST_ARGS=()
if [[ -n "$CONTRASTS" ]]; then
  IFS=':' read -ra CONTRAST_ITEMS <<< "$CONTRASTS"
  for item in "${CONTRAST_ITEMS[@]}"; do
    if [[ -n "$item" ]]; then
      CONTRAST_ARGS+=(--contrast "$item")
    fi
  done
fi

python -m aisafety.scripts.summarize_d4_human_llm_stage_contrasts \
  --workspace-root "$WORKDIR" \
  "${RUN_ARGS[@]}" \
  "${CONTRAST_ARGS[@]}" \
  --out-dir "$OUT_DIR"
