#!/bin/bash
# Resume an IT direct/thinking trace to five branches, then run both analyses.

set -euo pipefail

WORKDIR="${WORKDIR:-$(pwd)}"
ARTROOT="${ARTROOT:-$WORKDIR}"
PYTHON="${PYTHON:-$WORKDIR/.venv/bin/python}"
RUN_TAG="${RUN_TAG:-judge_reasoning_qwen3_it_corrected_v2}"
PAIRDIR="${PAIRDIR:-$ARTROOT/data/derived/judge_reasoning_suite_$RUN_TAG}"
ROOT="${ROOT:-$ARTROOT/artifacts/mechanistic/$RUN_TAG}"
TRACE_DIR="${TRACE_DIR:-$ROOT/trajectories_qwen3_8b_it}"
ANALYSIS_DIR="${ANALYSIS_DIR:-$ROOT/analysis_qwen3_8b_it_five_branch}"
CONTRAST_DIR="${CONTRAST_DIR:-$ROOT/mode_contrasts_five_branch}"
LOG_DIR="${LOG_DIR:-$ROOT/logs_clarity}"

GPU="${GPU:-7}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
BRANCHES_PER_COMPARISON="${BRANCHES_PER_COMPARISON:-5}"
MAX_PAIRS="${MAX_PAIRS:-210}"
MAX_NEW_TOKENS_THINKING="${MAX_NEW_TOKENS_THINKING:-1024}"
MAX_NEW_TOKENS_DIRECT="${MAX_NEW_TOKENS_DIRECT:-16}"
BOOTSTRAP="${BOOTSTRAP:-5000}"
SEED="${SEED:-1234}"

cd "$WORKDIR"
export PYTHONPATH="$WORKDIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$ARTROOT/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$LOG_DIR"

if [[ ! -s "$PAIRDIR/comparisons.jsonl" ]]; then
  echo "Missing comparison suite: $PAIRDIR/comparisons.jsonl" >&2
  exit 1
fi
if [[ ! -s "$TRACE_DIR/traces.jsonl" ]]; then
  echo "Missing existing corrected trace: $TRACE_DIR/traces.jsonl" >&2
  exit 1
fi

echo "Resuming IT clarity run"
echo "  trace_dir=$TRACE_DIR"
echo "  branches=$BRANCHES_PER_COMPARISON"
echo "  max_thinking_tokens=$MAX_NEW_TOKENS_THINKING"
echo "  bootstrap=$BOOTSTRAP"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" \
  -m aisafety.scripts.run_judge_reasoning_trajectories \
  --workspace-root "$WORKDIR" \
  --comparisons-jsonl "$PAIRDIR/comparisons.jsonl" \
  --run-label qwen3_8b_it \
  --model-id "$MODEL_ID" \
  --cache-dir "$HF_HOME" \
  --prompt-style chat_template \
  --reasoning-modes thinking,direct \
  --branches-per-comparison "$BRANCHES_PER_COMPARISON" \
  --max-pairs "$MAX_PAIRS" \
  --cap-strategy source_round_robin \
  --selected-layers 4,8,12,16,20,24,28,32 \
  --trajectory-points 17 \
  --shard-size 32 \
  --compress-shards \
  --resume \
  --max-prompt-length 4096 \
  --max-new-tokens-thinking "$MAX_NEW_TOKENS_THINKING" \
  --max-new-tokens-direct "$MAX_NEW_TOKENS_DIRECT" \
  --temperature 0.6 \
  --top-p 0.95 \
  --top-k 20 \
  --seed "$SEED" \
  --out-dir "$TRACE_DIR" \
  >"$LOG_DIR/trace_resume.out" 2>"$LOG_DIR/trace_resume.err"

PYTHONWARNINGS="ignore::FutureWarning" "$PYTHON" \
  -m aisafety.scripts.analyze_judge_reasoning_trajectories \
  --workspace-root "$WORKDIR" \
  --trace-dir "$TRACE_DIR" \
  --choice-confidence-threshold 0.6 \
  --target-confidence-threshold 0.6 \
  --commitment-persistence 2 \
  --out-dir "$ANALYSIS_DIR" \
  >"$LOG_DIR/trajectory_analysis.out" 2>"$LOG_DIR/trajectory_analysis.err"

"$PYTHON" \
  -m aisafety.scripts.analyze_judge_reasoning_mode_contrasts \
  --workspace-root "$WORKDIR" \
  --trace-dir "$TRACE_DIR" \
  --direct-mode direct \
  --deliberative-mode thinking \
  --trajectory-analysis-dir "$ANALYSIS_DIR" \
  --confidence-thresholds 0.4,0.5,0.6,0.7,0.8 \
  --commitment-persistence 2 \
  --bootstrap "$BOOTSTRAP" \
  --seed "$SEED" \
  --out-dir "$CONTRAST_DIR" \
  >"$LOG_DIR/mode_contrasts.out" 2>"$LOG_DIR/mode_contrasts.err"

echo "COMPLETE"
echo "trace_dir=$TRACE_DIR"
echo "analysis_dir=$ANALYSIS_DIR"
echo "contrast_dir=$CONTRAST_DIR"
echo "logs=$LOG_DIR"
