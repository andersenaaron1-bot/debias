#!/bin/bash
# CPU-only lexical artifact suite over the completed post-training judge-bias runs.

set -euo pipefail

WORKDIR="${WORKDIR:-$(pwd)}"
ARTROOT="${ARTROOT:-$WORKDIR}"
HLLM_PAIRS_JSONL="${HLLM_PAIRS_JSONL:-$ARTROOT/data/derived/d4_human_llm_alignment_pairs_matched_lenlex_relaxed_v1/pairs.jsonl}"
SURFACE_COUNTERFACTUAL_JSONL="${SURFACE_COUNTERFACTUAL_JSONL:-$ARTROOT/data/derived/d4_surface_counterfactual_pairs_matched_lenlex_relaxed_v1/counterfactuals.jsonl}"
OUT_DIR="${OUT_DIR:-$ARTROOT/artifacts/mechanistic/d4_lexical_judge_artifact_suite_v1}"
READOUT_TOP_K="${READOUT_TOP_K:-6}"

hllm_specs=()
surface_specs=()
hllm_contrasts=()
declare -A available_hllm=()

append_hllm() {
  local label="$1"
  local path="$2"
  if [[ -s "$path/stage_pair_summary_long.csv" ]]; then
    hllm_specs+=("$label=$path")
    available_hllm["$label"]=1
  else
    echo "skip missing HLLM summary: $path"
  fi
}

append_contrast_if_hllm() {
  local label="$1"
  local contrast="$2"
  if [[ -n "${available_hllm[$label]:-}" ]]; then
    hllm_contrasts+=("$contrast")
  fi
}

append_surface() {
  local label="$1"
  local path="$2"
  if [[ -s "$path/bt_pair_summary_long.csv" ]]; then
    surface_specs+=("$label=$path")
  else
    echo "skip missing surface summary: $path"
  fi
}

join_pipe() {
  local IFS='|'
  echo "$*"
}

MECH="$ARTROOT/artifacts/mechanistic"

append_hllm tulu_standard "$MECH/d4_human_llm_stage_contrast_summary_tulu_stage_matched_lenlex_relaxed_local_v1"
append_hllm tulu_minimal "$MECH/d4_human_llm_stage_contrast_summary_tulu_stage_template_minimal_relaxed_local_v1"
append_hllm tulu_rubric "$MECH/d4_human_llm_stage_contrast_summary_tulu_stage_template_rubric_quality_relaxed_local_v1"
append_hllm tulu_substance "$MECH/d4_human_llm_stage_contrast_summary_tulu_stage_template_substance_only_relaxed_local_v1"
append_hllm qwen_minimal "$MECH/d4_human_llm_stage_contrast_summary_qwen25_hllm_minimal_matched_relaxed_v1"
append_hllm qwen_rubric "$MECH/d4_human_llm_stage_contrast_summary_qwen25_hllm_rubric_quality_matched_relaxed_v1"
append_hllm qwen_standard "$MECH/d4_human_llm_stage_contrast_summary_qwen25_hllm_standard_matched_relaxed_v1"
append_hllm qwen_substance "$MECH/d4_human_llm_stage_contrast_summary_qwen25_hllm_substance_only_matched_relaxed_v1"
append_hllm gemma9_chat_minimal "$MECH/d4_human_llm_stage_contrast_summary_gemma2_9b_overnight_v1_hllm_chat_minimal"
append_hllm gemma9_chat_standard "$MECH/d4_human_llm_stage_contrast_summary_gemma2_9b_overnight_v1_hllm_chat_standard"
append_hllm gemma9_plain_minimal "$MECH/d4_human_llm_stage_contrast_summary_gemma2_9b_overnight_v1_hllm_plain_minimal"
append_hllm gemma9_plain_standard "$MECH/d4_human_llm_stage_contrast_summary_gemma2_9b_overnight_v1_hllm_plain_standard"

append_surface tulu "$MECH/d4_bt_surface_stage_template_summary_matched_relaxed_v1"
append_surface qwen "$MECH/d4_bt_surface_stage_template_summary_qwen25_matched_relaxed_v1"
append_surface gemma9_chat "$MECH/d4_bt_surface_stage_template_summary_gemma2_9b_overnight_v1_chat"
append_surface gemma9_plain "$MECH/d4_bt_surface_stage_template_summary_gemma2_9b_overnight_v1_plain"

append_contrast_if_hllm tulu_standard "tulu_standard::tulu3_sft_minus_base=tulu_standard::tulu3_sft,tulu_standard::llama31_base"
append_contrast_if_hllm tulu_standard "tulu_standard::tulu3_dpo_minus_sft=tulu_standard::tulu3_dpo,tulu_standard::tulu3_sft"
append_contrast_if_hllm tulu_standard "tulu_standard::tulu3_final_minus_base=tulu_standard::tulu3_final,tulu_standard::llama31_base"
append_contrast_if_hllm qwen_standard "qwen_standard::qwen25_instruct_minus_base=qwen_standard::qwen25_instruct,qwen_standard::qwen25_base"
append_contrast_if_hllm gemma9_chat_standard "gemma9_chat_standard::gemma2_9b_it_minus_base=gemma9_chat_standard::gemma2_9b_it,gemma9_chat_standard::gemma2_9b_base"
append_contrast_if_hllm gemma9_plain_standard "gemma9_plain_standard::gemma2_9b_it_minus_base=gemma9_plain_standard::gemma2_9b_it,gemma9_plain_standard::gemma2_9b_base"

if [[ ! -s "$HLLM_PAIRS_JSONL" ]]; then
  echo "Missing HLLM pair text source: $HLLM_PAIRS_JSONL" >&2
  exit 1
fi
if [[ ${#hllm_specs[@]} -eq 0 && ${#surface_specs[@]} -eq 0 ]]; then
  echo "No completed summary inputs were found under $MECH" >&2
  exit 1
fi

export WORKDIR ARTROOT OUT_DIR HLLM_PAIRS_JSONL SURFACE_COUNTERFACTUAL_JSONL
export HLLM_SUMMARY_SPECS="$(join_pipe "${hllm_specs[@]}")"
export SURFACE_SUMMARY_SPECS="$(join_pipe "${surface_specs[@]}")"
export HLLM_TARGET_CONTRASTS="$(join_pipe "${hllm_contrasts[@]}")"

bash cluster/local/run_d4_lexical_judge_artifact_audit.sh
python -m aisafety.scripts.read_d4_lexical_judge_artifacts --input "$OUT_DIR" --top-k "$READOUT_TOP_K"
