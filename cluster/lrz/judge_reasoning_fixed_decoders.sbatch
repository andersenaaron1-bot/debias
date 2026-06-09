#!/bin/bash
#SBATCH --job-name=judge-fixed-decoder
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
ARTROOT="${ARTROOT:-$WORKDIR}"
TRACE_DIR="${TRACE_DIR:-$ARTROOT/artifacts/mechanistic/judge_reasoning_trajectories_v1}"
OUT_DIR="${OUT_DIR:-$ARTROOT/artifacts/mechanistic/judge_reasoning_fixed_decoders_v1}"
REASONING_MODE="${REASONING_MODE:-thinking}"
ANCHOR_TARGET="${ANCHOR_TARGET:-final_choice}"
PROBE_TARGETS="${PROBE_TARGETS:-final_choice:target_option:presentation_order}"
C_VALUES="${C_VALUES:-0.01:0.03:0.1:0.3:1.0}"
MIN_FIT_ROWS="${MIN_FIT_ROWS:-40}"
MIN_SELECTION_ROWS="${MIN_SELECTION_ROWS:-20}"
MIN_EVALUATION_ROWS="${MIN_EVALUATION_ROWS:-20}"
CHOICE_CONFIDENCE_THRESHOLD="${CHOICE_CONFIDENCE_THRESHOLD:-0.6}"
TARGET_CONFIDENCE_THRESHOLD="${TARGET_CONFIDENCE_THRESHOLD:-0.6}"
COMMITMENT_PERSISTENCE="${COMMITMENT_PERSISTENCE:-2}"
SEED="${SEED:-1234}"
PROBE_TARGETS="${PROBE_TARGETS//:/,}"
C_VALUES="${C_VALUES//:/,}"

cd "$WORKDIR"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:-$WORKDIR/src}"

python -m aisafety.scripts.analyze_judge_reasoning_fixed_decoders \
  --workspace-root "$WORKDIR" \
  --trace-dir "$TRACE_DIR" \
  --reasoning-mode "$REASONING_MODE" \
  --anchor-target "$ANCHOR_TARGET" \
  --probe-targets "$PROBE_TARGETS" \
  --c-values "$C_VALUES" \
  --min-fit-rows "$MIN_FIT_ROWS" \
  --min-selection-rows "$MIN_SELECTION_ROWS" \
  --min-evaluation-rows "$MIN_EVALUATION_ROWS" \
  --choice-confidence-threshold "$CHOICE_CONFIDENCE_THRESHOLD" \
  --target-confidence-threshold "$TARGET_CONFIDENCE_THRESHOLD" \
  --commitment-persistence "$COMMITMENT_PERSISTENCE" \
  --seed "$SEED" \
  --out-dir "$OUT_DIR"
