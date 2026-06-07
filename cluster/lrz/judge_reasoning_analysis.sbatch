#!/bin/bash
#SBATCH --job-name=judge-reason-analyze
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
ARTROOT="${ARTROOT:-$WORKDIR}"
TRACE_DIR="${TRACE_DIR:-$ARTROOT/artifacts/mechanistic/judge_reasoning_trajectories_v1}"
OUT_DIR="${OUT_DIR:-$ARTROOT/artifacts/mechanistic/judge_reasoning_analysis_v1}"
PROBE_TARGETS="${PROBE_TARGETS:-final_choice,target_option,target_selected,condition_label,presentation_order}"
GROUP_COLUMNS="${GROUP_COLUMNS:-comparison_dimension,source_dataset,validity_type,difficulty_tier,analysis_split}"
POSITIVE_CONDITION_LABEL="${POSITIVE_CONDITION_LABEL:-}"
CV_FOLDS="${CV_FOLDS:-5}"
PROBE_C="${PROBE_C:-0.1}"
MIN_PROBE_ROWS="${MIN_PROBE_ROWS:-40}"
MIN_DIMENSION_ROWS="${MIN_DIMENSION_ROWS:-80}"
COMMITMENT_AUC="${COMMITMENT_AUC:-0.75}"
COMMITMENT_PERSISTENCE="${COMMITMENT_PERSISTENCE:-3}"
CHOICE_CONFIDENCE_THRESHOLD="${CHOICE_CONFIDENCE_THRESHOLD:-0.8}"
TARGET_CONFIDENCE_THRESHOLD="${TARGET_CONFIDENCE_THRESHOLD:-0.8}"
SEED="${SEED:-1234}"
PROBE_TARGETS="${PROBE_TARGETS//:/,}"
GROUP_COLUMNS="${GROUP_COLUMNS//:/,}"

cd "$WORKDIR"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:-$WORKDIR/src}"

python -m aisafety.scripts.analyze_judge_reasoning_trajectories \
  --workspace-root "$WORKDIR" \
  --trace-dir "$TRACE_DIR" \
  --probe-targets "$PROBE_TARGETS" \
  --group-columns "$GROUP_COLUMNS" \
  --positive-condition-label "$POSITIVE_CONDITION_LABEL" \
  --cv-folds "$CV_FOLDS" \
  --probe-c "$PROBE_C" \
  --min-probe-rows "$MIN_PROBE_ROWS" \
  --min-dimension-rows "$MIN_DIMENSION_ROWS" \
  --commitment-auc "$COMMITMENT_AUC" \
  --commitment-persistence "$COMMITMENT_PERSISTENCE" \
  --choice-confidence-threshold "$CHOICE_CONFIDENCE_THRESHOLD" \
  --target-confidence-threshold "$TARGET_CONFIDENCE_THRESHOLD" \
  --seed "$SEED" \
  --out-dir "$OUT_DIR"
