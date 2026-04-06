# D4 Reduced Ontology

## Purpose

`D4` is the reduced tracing ontology for the first mechanistic pass.

It is not the full linguistic ontology.

It is the subset worth carrying into:

- SAE feature search
- cross-layer feature matching
- feature ablation and steering
- adapter-vs-base activation comparison

## Basis

`D4` is the intersection of:

- `D1`: linguistically grounded atom definitions
- `D2`: bundle support on broad discovery corpora
- `D3`: ecological relevance on Laurito
- `M2 vs M3`: differential sensitivity when academic formality is removed from training

The key rule is:

`Prefer local atom targets over broad register labels.`

That means:

- `academic_formality` remains useful as a readout bundle
- but the first tracing targets are lower-level atoms such as `nominalization_patterns` or `formal_connectives`

## Why This Reduction Is Defensible

The `M2 vs M3` contrast was the missing filter.

It shows that removing `academic_formality` from training does not only nudge one score; it changes the ecological behavior materially, especially on the paper domain.

That makes `academic_formality` a useful supervisory family, but still not a good mechanistic primitive.

The reduced ontology therefore splits the effect into lower-level carrier bundles:

1. `formal_information_packaging`
2. `certainty_and_stance_calibration`
3. `professional_self_presentation`
4. `enumerative_assistant_packaging`
5. `narrative_and_summary_packaging`
6. `benefit_sales_packaging`

## Primary Tracing Bundles

### 1. Formal Information Packaging

Use this instead of tracing `academic_formality` as one monolithic feature.

Atoms:

- `nominalization_patterns`
- `formal_connectives`
- `complex_noun_phrase_chains`
- `passive_or_agentless_constructions`
- `institutional_impersonality`
- `subordination_density`
- `participial_modifier_usage`

Main use:

- paper-domain tracing
- M2-vs-M3 comparison
- identifying which local formal-density features are attenuated by the repair

### 2. Certainty And Stance Calibration

Atoms:

- `authoritative_certainty`
- `booster_certainty_markers`
- `hedge_markers`
- `epistemic_hedging`
- `evidential_markers`

Main use:

- distinguish certainty/stance from generic formality
- test whether the judge reads certainty as a quality proxy

### 3. Professional Self Presentation

Atoms:

- `self_mention_markers`
- `professional_self_presentation`

Main use:

- judge-facing self-presentational rhetoric
- likely portable to job-application-like or evaluative settings

### 4. Enumerative Assistant Packaging

Atoms:

- `bullet_or_list_structure`
- `enumeration_markers`
- `balanced_multi_part_completion`
- `helpful_assistant_wrapper`

Main use:

- assistant-style formatting
- listiness and templated helpfulness

This bundle matters because `bullet_or_list_structure` is one of the clearest positive ecological signals in `D3`.

## Secondary Tracing Bundles

### 5. Narrative And Summary Packaging

Atoms:

- `sentence_length_balance`
- `compression_vs_elaboration`
- `narrative_engagement_stance`
- `quotation_style`
- `entity_grid_coherence`

Main use:

- movie/synopsis-facing packaging
- summary-structure effects

### 6. Benefit Sales Packaging

Atoms:

- `feature_benefit_call_to_value_script`
- `benefit_first_packaging`
- `enthusiasm_or_salesmanship`
- `promotional_adjectives`

Main use:

- product-domain rhetoric
- distinction between sales framing and assistant formatting

## Readout-Only Variables

Do not center the first tracing pass on these directly.

Use them as aggregate readouts:

- `academic_formality`
- `template_packaging`
- `promotional_tone`
- `narrative_packaging`
- `safety_corporate_tone`
- `stance_calibration`
- `job_application_professional_register`
- `helpdesk_assistant_register`
- `movie_synopsis_register`
- `product_pitch_register`

These are useful for evaluation and interpretation, but they are too coarse to be the first mechanistic primitives.

## Immediate Mechanistic Priorities

Highest-priority atoms for the first feature search:

- `nominalization_patterns`
- `formal_connectives`
- `authoritative_certainty`
- `booster_certainty_markers`
- `self_mention_markers`
- `bullet_or_list_structure`
- `helpful_assistant_wrapper`
- `sentence_length_balance`
- `compression_vs_elaboration`

These are the best compromise between:

- ecological relevance
- frequency
- interpretability
- likely feature locality

## Next Step

Use [d4_reduced_ontology_v1.json](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/configs/ontology/d4_reduced_ontology_v1.json) as the target list for:

1. SAE feature search on `Gemma 2 9B`
2. cross-layer matching for persistent atoms
3. adapter-vs-base comparison on the selected atom families

