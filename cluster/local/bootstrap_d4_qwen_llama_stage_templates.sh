#!/bin/bash
# Bootstrap existing Qwen/Tulu/Llama D4 stage-by-template interaction summaries.
#
# This is CPU-only. It scans artifact directories for
# template_stage_interaction_pair_deltas.csv, skips empty single-template
# summaries, and writes bootstrap_stage_template_interactions.csv next to each
# valid combined summary.

set -euo pipefail

WORKDIR="${WORKDIR:-$(pwd)}"
ARTROOT="${ARTROOT:-$WORKDIR}"
PYTHON="${PYTHON:-python}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$ARTROOT/artifacts/mechanistic}"
INCLUDE_NAME="${INCLUDE_NAME:-qwen,tulu,llama}"
EXCLUDE_NAME="${EXCLUDE_NAME:-}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-2000}"
SEED="${SEED:-1234}"
OVERWRITE="${OVERWRITE:-1}"
DRY_RUN="${DRY_RUN:-0}"
MAX_FILES="${MAX_FILES:-0}"

cd "$WORKDIR"
export PYTHONPATH="$WORKDIR/src${PYTHONPATH:+:$PYTHONPATH}"

cmd=(
  "$PYTHON" -m aisafety.scripts.bootstrap_d4_stage_template_summaries
  --workspace-root "$WORKDIR"
  --artifact-root "$ARTIFACT_ROOT"
  --include-name "$INCLUDE_NAME"
  --exclude-name "$EXCLUDE_NAME"
  --bootstrap "$BOOTSTRAP_SAMPLES"
  --seed "$SEED"
  --max-files "$MAX_FILES"
)
if [[ "$OVERWRITE" == "1" ]]; then
  cmd+=(--overwrite)
fi
if [[ "$DRY_RUN" == "1" ]]; then
  cmd+=(--dry-run)
fi

echo "Running D4 stage-template bootstrap scan"
echo "  artifact_root=$ARTIFACT_ROOT"
echo "  include_name=$INCLUDE_NAME"
echo "  exclude_name=$EXCLUDE_NAME"
echo "  bootstrap_samples=$BOOTSTRAP_SAMPLES"
echo "  seed=$SEED"
echo "  overwrite=$OVERWRITE"

"${cmd[@]}"
