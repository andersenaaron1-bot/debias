# Judge Experiment Matrix V1

## Purpose

This document separates the project into two distinct programs:

1. `ontology and tracing`
2. `repair interventions`

The repair matrix should not define the ontology by itself.

Instead, the repair matrix is used to test whether specific intervention channels
change the same cue pathways that are later traced mechanistically.

It is a `diagnostic intervention suite`, not yet the final ideal intervention
suite for the paper.

## Canonical Model Roles

### `J0`

The neutral anchor judge.

- config: [j0_anchor_v1.json](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/configs/experiments/j0_anchor_v1.json)
- fallback config: [j0_anchor_v1_h100safe.json](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/configs/experiments/j0_anchor_v1_h100safe.json)
- trained on:
  - SHP preference supervision
  - HelpSteer2 anchor supervision
- not trained on:
  - paired invariance
  - cue-adversarial removal

This is the main ecological and mechanistic anchor.

`j0_anchor_v1_h100safe` should be treated as a runtime-equivalent fallback, not
as a new experimental condition. It keeps the same objective family and overall
schedule, while reducing train/eval microbatches to avoid marginal OOM on a
single H100.

### `Jrepair-all`

The full repair judge.

- config: [jrepair_all_v1.json](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/configs/experiments/jrepair_all_v1.json)
- trained on:
  - SHP preference supervision
  - HelpSteer2 anchor supervision
  - paired invariance
  - cue-adversarial removal

This is the main repaired comparison model.

## What This Matrix Is And Is Not

This matrix is:

- a source of controlled behavioral perturbations
- a necessity-test suite for current repair families
- a bridge into mechanistic tracing

This matrix is not:

- the full ontology
- the final intervention design
- a one-to-one reflection of the D4 reduced ontology

The final intervention design should be revisited only after the first
mechanistic pass on `J0`, `Jrepair-all`, and the first-wave ablations.

## Why `J0` Is Cleaner Than Using Only `M2`/`M3`

`M2` and `M3` are both repaired judges.

They are useful for intervention sensitivity, but they are not a clean anchor for
the question:

`Which cues does a generic aligned judge rely on in the first place?`

`J0` answers that.

Then `Jrepair-all` and the ablations answer:

`Which interventions change those cue pathways, and how?`

## Repair Matrix Structure

The current matrix has three levels.

### 1. Cue-family leave-one-out

These runs remove one cue-adversarial family at a time while keeping the rest of
the repair active.

Generated configs:

- `jrepair_loo_cue_academic_formality_v1.json`
- `jrepair_loo_cue_safety_corporate_tone_v1.json`
- `jrepair_loo_cue_promotional_sales_tone_v1.json`
- `jrepair_loo_cue_narrative_packaging_v1.json`
- `jrepair_loo_cue_template_boilerplate_v1.json`
- `jrepair_loo_cue_verbosity_compression_v1.json`
- `jrepair_loo_cue_hedging_certainty_v1.json`

### 2. Invariance-axis leave-one-out

These runs remove one paired-invariance axis at a time while keeping the rest of
the repair active.

Generated configs:

- `jrepair_loo_axis_formality_v1.json`
- `jrepair_loo_axis_detox_tone_v1.json`
- `jrepair_loo_axis_fluency_v1.json`
- `jrepair_loo_axis_paraphrase_surface_v1.json`

### 3. Joint leave-one-out where justified

Only use a joint ablation when there is a reasonable mapping between the paired
invariance axis and the cue-adversarial family.

Right now the strongest justified joint case is:

- `jrepair_loo_joint_academic_formality_v1.json`

This is the principled replacement for treating `M3` as a one-off experiment.

That joint academic run should be read as:

- a positive-control manipulation check
- not a final proof that academic formality is the uniquely causal family

## Current Mapping To Older Runs

For continuity:

- [m2_full_v1.json](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/configs/experiments/m2_full_v1.json) is the older internal equivalent of `Jrepair-all`
- [m3_full_v1.json](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/configs/experiments/m3_full_v1.json) is the older internal equivalent of `jrepair_loo_joint_academic_formality_v1`

The new names should be preferred in future analysis and writing.

## Methodological Role

The repair matrix is not the ontology.

Its role is:

1. establish an anchor judge
2. establish a full repair
3. measure sensitivity to removing individual repair components
4. compare those behavioral shifts to the frozen D4 tracing ontology

That separation makes the project defensible as a layered program:

- define candidate cue atoms and bundles
- validate them statistically
- test ecological relevance
- localize them mechanistically
- compare training-time repairs and feature-level interventions

## How The Matrix Feeds The Mechanistic Section

The intended sequence is:

1. run `D3` on `J0`
2. run `D3` on `Jrepair-all`
3. run `D3` on the first-wave leave-one-out variants
4. identify which D4 atoms and bundles are active in `J0`
5. identify which of those are attenuated in `Jrepair-all`
6. identify which of those reappear when specific repair components are removed

Only after those comparisons are frozen should the project define the next,
mechanism-informed intervention target.

## Generator

The config generator is:

- [generate_judge_repair_configs.py](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/src/aisafety/scripts/generate_judge_repair_configs.py)

Spec source:

- [judge_repair_matrix_v1.json](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/configs/experiments/judge_repair_matrix_v1.json)

Regenerate with:

```bash
python -m aisafety.scripts.generate_judge_repair_configs --help
```
