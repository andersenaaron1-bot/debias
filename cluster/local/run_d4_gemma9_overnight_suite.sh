#!/bin/bash
# Convenience wrapper for the full Gemma 2 9B base-vs-IT suite on GPUs 0 and 1.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export GEMMA_SIZE="${GEMMA_SIZE:-9b}"
export RUN_TAG="${RUN_TAG:-gemma2_9b_overnight_v1}"
export GPU_A="${GPU_A:-0}"
export GPU_B="${GPU_B:-1}"
export SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-1}"
export LIKELIHOOD_BATCH_SIZE="${LIKELIHOOD_BATCH_SIZE:-1}"
export MAX_LENGTH="${MAX_LENGTH:-2048}"

bash "$SCRIPT_DIR/run_d4_gemma27_overnight_suite.sh"
