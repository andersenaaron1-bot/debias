#!/bin/bash
#SBATCH --job-name=judge-budget-analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
ARTROOT="${ARTROOT:-$WORKDIR}"
RUN_DIR="${RUN_DIR:-$ARTROOT/artifacts/mechanistic/judge_deliberation_budget_sweep_v1}"
OUT_DIR="${OUT_DIR:-$ARTROOT/artifacts/mechanistic/judge_deliberation_budget_analysis_v1}"
BOOTSTRAP="${BOOTSTRAP:-5000}"
SEED="${SEED:-1234}"

cd "$WORKDIR"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:-$WORKDIR/src}"

python -m aisafety.scripts.analyze_judge_reasoning_budget_sweep \
  --workspace-root "$WORKDIR" \
  --run-dir "$RUN_DIR" \
  --bootstrap "$BOOTSTRAP" \
  --seed "$SEED" \
  --out-dir "$OUT_DIR"
