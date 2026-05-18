#!/bin/bash
# Submit a first-pass Tulu/Llama human-vs-LLM training-stage contrast matrix on LRZ.
#
# Run this from the LRZ login node after exporting WORKDIR, ARTROOT, IMAGE, and
# cache variables as described in AGENTS.md. The script queues one CPU pair-build
# job, several dependent GPU scoring jobs, and one dependent CPU summary job.

set -euo pipefail

WORKDIR="${WORKDIR:-$HOME/debias}"
ARTROOT="${ARTROOT:-$WORKDIR}"
IMAGE="${IMAGE:-ghcr.io#andersenaaron1-bot/debias:sae-mech-v1}"
PARTS="${PARTS:-lrz-hgx-h100-94x4,lrz-dgx-a100-80x8,lrz-hgx-a100-80x4}"
HF_HOME="${HF_HOME:-$ARTROOT/.cache/huggingface}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
LOG_DIR="${LOG_DIR:-$ARTROOT/slurm_logs}"

RUN_TAG="${RUN_TAG:-tulu_stage_scout_v1}"
PAIR_JSONL="${PAIR_JSONL:-$ARTROOT/data/derived/d4_human_llm_alignment_pairs_strat10k_v3/pairs.jsonl}"
PAIR_OUT_DIR="${PAIR_OUT_DIR:-$ARTROOT/data/derived/d4_human_llm_stage_contrast_pairs_${RUN_TAG}}"
BT_PAIRS_JSONL="${BT_PAIRS_JSONL:-$PAIR_OUT_DIR/bt_pairs.jsonl}"
OUT_ROOT="${OUT_ROOT:-$ARTROOT/artifacts/mechanistic/d4_hllm_stage_${RUN_TAG}}"
SUMMARY_OUT_DIR="${SUMMARY_OUT_DIR:-$ARTROOT/artifacts/mechanistic/d4_human_llm_stage_contrast_summary_${RUN_TAG}}"

# Keep the default as a scout. Set MAX_SOURCE_PAIRS=0 for the full broad file.
MAX_SOURCE_PAIRS="${MAX_SOURCE_PAIRS:-1000}"
SCORE_MAX_PAIRS="${SCORE_MAX_PAIRS:-0}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-2}"
LIKELIHOOD_BATCH_SIZE="${LIKELIHOOD_BATCH_SIZE:-2}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
USE_4BIT="${USE_4BIT:-0}"
INCLUDE_META_INSTRUCT="${INCLUDE_META_INSTRUCT:-1}"
INCLUDE_RESPONSE_LIKELIHOOD="${INCLUDE_RESPONSE_LIKELIHOOD:-1}"
COMPARISON_TEMPLATE="${COMPARISON_TEMPLATE:-standard}"

mkdir -p "$LOG_DIR"

common_container_args=(
  --container-image="$IMAGE"
  --container-mounts="$WORKDIR:$WORKDIR,$ARTROOT:$ARTROOT,$ARTROOT:/workspace"
  --container-workdir="$WORKDIR"
)

echo "Submitting Tulu/Llama stage matrix"
echo "  run_tag=$RUN_TAG"
echo "  pair_jsonl=$PAIR_JSONL"
echo "  max_source_pairs=$MAX_SOURCE_PAIRS"
echo "  out_root=$OUT_ROOT"
echo "  comparison_template=$COMPARISON_TEMPLATE"

pair_job="$(
  sbatch --parsable \
    --partition=lrz-cpu \
    --qos=cpu \
    --job-name=hllm-stage-pairs \
    --cpus-per-task=2 \
    --mem=16G \
    --time=00:20:00 \
    --chdir="$WORKDIR" \
    --output="$LOG_DIR/%x-%j.out" \
    --error="$LOG_DIR/%x-%j.err" \
    "${common_container_args[@]}" \
    --container-env=PYTHONPATH \
    --export=ALL,WORKDIR="$WORKDIR",ARTROOT="$ARTROOT",PYTHONPATH="$WORKDIR/src",PAIR_JSONL="$PAIR_JSONL",OUT_DIR="$PAIR_OUT_DIR",MAX_PAIRS="$MAX_SOURCE_PAIRS",INCLUDE_ORDER_SWAPS=1 \
    cluster/lrz/d4_human_llm_stage_contrast_pairs.sbatch
)"
echo "pair_job=$pair_job"

score_jobs=()
run_entries=()

submit_score() {
  local label="$1"
  local model_id="$2"
  local scoring_mode="$3"
  local prompt_style="$4"
  local out_dir="$5"
  local batch_size="$6"
  local job_name="$7"

  local job_id
  job_id="$(
    sbatch --parsable \
      --dependency=afterok:"$pair_job" \
      --partition="$PARTS" \
      --job-name="$job_name" \
      --gres=gpu:1 \
      --cpus-per-task=8 \
      --mem=160G \
      --time=04:00:00 \
      --chdir="$WORKDIR" \
      --output="$LOG_DIR/%x-%j.out" \
      --error="$LOG_DIR/%x-%j.err" \
      "${common_container_args[@]}" \
      --container-env=PYTHONPATH,HF_HOME,TRANSFORMERS_CACHE,HF_DATASETS_CACHE,HF_TOKEN,HUGGING_FACE_HUB_TOKEN \
      --export=ALL,WORKDIR="$WORKDIR",ARTROOT="$ARTROOT",PYTHONPATH="$WORKDIR/src",HF_HOME="$HF_HOME",TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE",HF_DATASETS_CACHE="$HF_DATASETS_CACHE",BT_PAIRS_JSONL="$BT_PAIRS_JSONL",SCORING_MODE="$scoring_mode",STAGE_LABEL="$label",MODEL_ID="$model_id",PROMPT_STYLE="$prompt_style",COMPARISON_TEMPLATE="$COMPARISON_TEMPLATE",OUT_DIR="$out_dir",MAX_PAIRS="$SCORE_MAX_PAIRS",SCORE_BATCH_SIZE="$batch_size",MAX_LENGTH="$MAX_LENGTH",USE_4BIT="$USE_4BIT" \
      cluster/lrz/d4_human_llm_stage_contrast.sbatch
  )"
  score_jobs+=("$job_id")
  run_entries+=("$label=$out_dir")
  echo "$label=$job_id"
}

submit_score \
  "llama31_base" \
  "meta-llama/Llama-3.1-8B" \
  "forced_choice" \
  "plain" \
  "$OUT_ROOT/llama31_base_forced_plain" \
  "$SCORE_BATCH_SIZE" \
  "hllm-l31-base"

submit_score \
  "tulu3_sft" \
  "allenai/Llama-3.1-Tulu-3-8B-SFT" \
  "forced_choice" \
  "chat_template" \
  "$OUT_ROOT/tulu3_sft_forced_chat" \
  "$SCORE_BATCH_SIZE" \
  "hllm-tulu-sft"

submit_score \
  "tulu3_dpo" \
  "allenai/Llama-3.1-Tulu-3-8B-DPO" \
  "forced_choice" \
  "chat_template" \
  "$OUT_ROOT/tulu3_dpo_forced_chat" \
  "$SCORE_BATCH_SIZE" \
  "hllm-tulu-dpo"

submit_score \
  "tulu3_final" \
  "allenai/Llama-3.1-Tulu-3-8B" \
  "forced_choice" \
  "chat_template" \
  "$OUT_ROOT/tulu3_final_forced_chat" \
  "$SCORE_BATCH_SIZE" \
  "hllm-tulu-final"

if [[ "$INCLUDE_META_INSTRUCT" == "1" ]]; then
  submit_score \
    "llama31_instruct" \
    "meta-llama/Llama-3.1-8B-Instruct" \
    "forced_choice" \
    "chat_template" \
    "$OUT_ROOT/llama31_instruct_forced_chat" \
    "$SCORE_BATCH_SIZE" \
    "hllm-l31-inst"
fi

if [[ "$INCLUDE_RESPONSE_LIKELIHOOD" == "1" ]]; then
  submit_score \
    "llama31_base_like" \
    "meta-llama/Llama-3.1-8B" \
    "response_likelihood" \
    "plain" \
    "$OUT_ROOT/llama31_base_response_likelihood" \
    "$LIKELIHOOD_BATCH_SIZE" \
    "hllm-l31-like"

  submit_score \
    "tulu3_sft_like" \
    "allenai/Llama-3.1-Tulu-3-8B-SFT" \
    "response_likelihood" \
    "chat_template" \
    "$OUT_ROOT/tulu3_sft_response_likelihood" \
    "$LIKELIHOOD_BATCH_SIZE" \
    "hllm-sft-like"

  submit_score \
    "tulu3_dpo_like" \
    "allenai/Llama-3.1-Tulu-3-8B-DPO" \
    "response_likelihood" \
    "chat_template" \
    "$OUT_ROOT/tulu3_dpo_response_likelihood" \
    "$LIKELIHOOD_BATCH_SIZE" \
    "hllm-dpo-like"
fi

score_dependency="$(IFS=:; echo "${score_jobs[*]}")"
runs_arg="$(IFS=:; echo "${run_entries[*]}")"

contrasts_arg="tulu3_sft_minus_base=tulu3_sft-llama31_base:tulu3_dpo_minus_sft=tulu3_dpo-tulu3_sft:tulu3_final_minus_dpo=tulu3_final-tulu3_dpo:tulu3_final_minus_base=tulu3_final-llama31_base"
if [[ "$INCLUDE_META_INSTRUCT" == "1" ]]; then
  contrasts_arg="$contrasts_arg:llama31_instruct_minus_base=llama31_instruct-llama31_base:llama31_instruct_minus_tulu3_dpo=llama31_instruct-tulu3_dpo"
fi

summary_job="$(
  sbatch --parsable \
    --dependency=afterok:"$score_dependency" \
    --partition=lrz-cpu \
    --qos=cpu \
    --job-name=hllm-stage-sum \
    --cpus-per-task=2 \
    --mem=8G \
    --time=00:15:00 \
    --chdir="$WORKDIR" \
    --output="$LOG_DIR/%x-%j.out" \
    --error="$LOG_DIR/%x-%j.err" \
    "${common_container_args[@]}" \
    --container-env=PYTHONPATH \
    --export=ALL,WORKDIR="$WORKDIR",ARTROOT="$ARTROOT",PYTHONPATH="$WORKDIR/src",RUNS="$runs_arg",CONTRASTS="$contrasts_arg",OUT_DIR="$SUMMARY_OUT_DIR" \
    cluster/lrz/d4_human_llm_stage_contrast_summary.sbatch
)"

echo "summary_job=$summary_job"
echo "summary_out_dir=$SUMMARY_OUT_DIR"
