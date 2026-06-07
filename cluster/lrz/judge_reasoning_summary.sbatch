#!/bin/bash
#SBATCH --job-name=judge-reason-summary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
ARTROOT="${ARTROOT:-$WORKDIR}"
OUT_DIR="${OUT_DIR:-$ARTROOT/artifacts/mechanistic/judge_reasoning_suite_summary_v1}"
ANALYSIS_SPECS="${ANALYSIS_SPECS:?Set ANALYSIS_SPECS as label=path|label=path}"
INTERVENTION_SPECS="${INTERVENTION_SPECS:-}"

cd "$WORKDIR"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:-$WORKDIR/src}"

CMD=(
  python -m aisafety.scripts.summarize_judge_reasoning_suite
  --workspace-root "$WORKDIR"
  --out-dir "$OUT_DIR"
)

IFS='|' read -r -a ANALYSES <<< "$ANALYSIS_SPECS"
for spec in "${ANALYSES[@]}"; do
  [[ -n "$spec" ]] && CMD+=(--analysis "$spec")
done
if [[ -n "$INTERVENTION_SPECS" ]]; then
  IFS='|' read -r -a INTERVENTIONS <<< "$INTERVENTION_SPECS"
  for spec in "${INTERVENTIONS[@]}"; do
    [[ -n "$spec" ]] && CMD+=(--intervention "$spec")
  done
fi

"${CMD[@]}"
