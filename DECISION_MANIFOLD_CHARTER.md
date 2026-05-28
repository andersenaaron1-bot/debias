# D4 Decision-Manifold And Circuit-Suppression Charter

Status: working charter, created 2026-05-29.

This charter extends the D4 judge-bias work without replacing
`IMPLEMENTAION_PATH.md`. The immediate goal is to turn the existing
post-training judge-behavior results into a stable behavioral geometry before
committing to SAE, crosscoder, or circuit-suppression claims.

## Motivation

The current D4 judge experiments show that base and instruction-tuned models
do not differ only by a scalar amount of LLM preference. Their decisions change
across prompt frames, ecological judge prompts, deterministic surface-cue
counterfactuals, response-likelihood controls, model families, and training
stages. This suggests that LLM-favoring judge behavior is a structured decision
manifold rather than a single bias knob.

The next project asks:

> What latent decision factors make a model prefer LLM-authored or
> LLM-presented answers, how does post-training reshape those factors, and can
> feature- or circuit-level suppression reduce specific factors without
> damaging the whole evaluator?

## Scientific Target

The primary object is an example-level behavioral vector:

```text
answer_pair_i -> [
  margins across prompt frames,
  stage deltas across base/SFT/DPO/IT/final checkpoints,
  surface-cue counterfactual deltas,
  response-likelihood controls,
  family-specific and interface-specific variants
]
```

This vector is the decision manifold coordinate for the pair. The project
first studies this geometry behaviorally, then tests whether corresponding
representational directions or sparse features mediate the behavior.

## Core Hypotheses

1. Base models already contain structured LLM-preference factors; the bias is
   not created only by instruction tuning.
2. Instruction tuning and related post-training steps amplify, suppress, or
   rotate these factors rather than uniformly increasing all LLM preference.
3. Prompt-frame sensitivity, surface-cue sensitivity, and response-likelihood
   preference are partially separable factors.
4. Mitigation by feature or circuit suppression should target factors, not the
   global LLM-preference margin.
5. Crosscoder or SAE features become viable only after a behavioral factor is
   stable enough to serve as a target.

## Claim Gates

Behavioral manifold analysis may support:

- separability of LLM-preference factors
- stage-wise amplification or rotation of those factors
- family and interface differences in factor loading
- likelihood-vs-forced-choice dissociation

Semantic direction analysis may support:

- hidden-state directions that predict factor scores on held-out examples
- layer/span localization of factor predictiveness
- increased coupling between post-trained models and frame/cue factors

Causal patching or direction intervention may support:

- a factor direction causally participates in the judge margin
- suppressing the direction reduces a target sensitivity factor
- suppression side effects on content preference or likelihood controls are
  bounded

SAE or crosscoder analysis may support:

- sparse or shared latents align with validated behavioral factors
- latents differ between base and post-trained checkpoints in activation,
  decoder strength, or decision coupling
- latent suppression changes the target factor on held-out examples

Do not claim that a latent explains the phenomenon if it only has plausible top
activations. It must predict or causally affect the factor.

## Roadmap

### Phase 1: Behavioral Manifold Assembly

Build a reproducible matrix from existing pair-level outputs:

- human-vs-LLM stage margins
- prompt-template deltas
- stage-by-template interactions
- surface-cue stage/template deltas
- response-likelihood controls where available

Deliverables:

- `manifold_long.csv`: one feature value per example and metric
- `manifold_wide.csv`: one row per example with all available coordinates
- `feature_summary.csv`: coverage and distribution summaries
- manifest recording source directories and value columns

### Phase 2: Factor Discovery

Run factor analysis over the wide matrix:

- PCA for dominant variance directions
- sparse PCA or NMF for more localized factors
- clustering of examples by factor profiles
- source/domain checks
- bootstrap stability of loadings

Factor names are assigned only after inspecting high-loading and low-loading
examples. Candidate labels might include judge-frame reliance,
assistant-packaging reliance, recommendation-domain authority, formal abstract
plausibility, or likelihood-aligned LLM preference, but these are not assumed
up front.

### Phase 3: Semantic Direction Probes

For stable factors, fit per-model and per-layer directions:

- target: factor score, signed delta, or absolute sensitivity
- input: residual stream at decision position, prompt-frame spans, and answer
  spans
- validation: held-out pairs, source splits, and prompt families

This phase asks whether a factor is represented in model states and where it is
most predictive.

### Phase 4: Causal Direction Tests

Intervene on factor directions:

- patch high-factor activations into low-factor examples
- remove or damp the direction
- measure target factor reduction and off-target side effects

This is the first phase that can support mitigation claims.

### Phase 5: Crosscoder Or SAE Follow-Up

Train sparse model-diffing tools only after a factor is behaviorally stable:

- within-family first: Llama base/Tulu SFT, SFT/DPO, Qwen base/IT, Gemma base/IT
- compare feature activation, decoder geometry, and coupling to factor scores
- validate with close negatives and held-out source splits
- test feature/circuit suppression against the factor

Cross-family feature alignment is deferred until within-family results are
stable. Crosscoders should be evaluated as model-diff tools, not as generic
semantic label generators.

## Initial Data Assets

Already available or partly available:

- Tulu/Llama staged human-vs-LLM prompt-template results
- Qwen base/IT prompt-template results
- Gemma 2 9B and partial 27B prompt-template results
- deterministic surface-cue counterfactual results
- response-likelihood controls
- Laurito ecological prompt validation runs

The first implementation step is to assemble these pair-level files into a
single behavioral-manifold dataset without recomputing model scores.

## Non-Goals

- Do not start by training cross-family crosscoders.
- Do not define the semantic set from hand-written style atoms alone.
- Do not treat attention hotspots as causal evidence.
- Do not claim mitigation from prompt wording alone when the target is feature
  or circuit suppression.
- Do not use SAE top examples without held-out behavioral and causal checks.

## First-Step Command Shape

After copying result directories locally or running on the server:

```bash
python -m aisafety.scripts.build_d4_decision_manifold_matrix \
  --workspace-root "$WORKDIR" \
  --input tulu_hllm="$ARTROOT/artifacts/mechanistic/d4_human_llm_template_sensitivity_tulu_relaxed_v1" \
  --input qwen_hllm="$ARTROOT/artifacts/mechanistic/d4_human_llm_template_sensitivity_qwen25_matched_relaxed_v1" \
  --input surface_tulu="$ARTROOT/artifacts/mechanistic/d4_bt_surface_stage_template_summary_matched_relaxed_v1" \
  --out-dir "$ARTROOT/artifacts/mechanistic/d4_decision_manifold_matrix_v1"
```

Additional inputs can be appended as more validation summaries are copied.
