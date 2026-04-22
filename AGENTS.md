# AISafety Single Source Of Truth

This file is the canonical project charter and operating guide for the repo.

If any README or `docs/` file disagrees with this file, follow `AGENTS.md`.
Other planning documents should be treated as supporting analysis notes,
historical records, or narrower method writeups unless they have been brought
back into alignment here.

## Documentation Policy

To avoid stale or conflicting metadata:

- `AGENTS.md` is the only live source of truth for experiment goals,
  implementation goals, model roles, ontology status, and immediate next steps
- `paper/` is the only place where the evolving workshop-paper draft, rough
  narrative structure, and citation database should live
- `docs/` may contain archival or supporting notes, but should not be used as
  the authoritative source for current project direction
- do not add new live planning or implementation-goal documents under `docs/`
- material in `paper/` may discuss motivation, prior art, and manuscript
  structure, but must not override the operational rules in `AGENTS.md`

## Core Objective

The primary goal of the project is **mechanistic interpretation of judge
decision making** in Gemma-based reward judges.

The project is specifically about showing that:

- judges rely on low-level linguistic, rhetorical, and formatting cues as
  proxies for quality in human-vs-LLM comparisons
- these cue pathways can be expressed as a principled atom-to-bundle ontology
- those pathways can be localized inside the model
- later interventions should be redesigned around the traced pathways rather
  than around broad heuristic repair families alone

The repair / debiasing story is important, but it is **secondary to the
mechanistic story**. The main novelty target is:

1. grounded ontology
2. ecological cue relevance
3. mechanistic localization
4. mechanism-informed repair

## Project Split

The repo now supports two linked but distinct programs.

### Program A: Ontology And Mechanistic Tracing

This asks:

- which cue atoms are linguistically grounded?
- which cue bundles are statistically supported?
- which of those matter for real judge choices?
- where are they represented and used inside the model?

This is the main scientific backbone.

### Program B: Repair Interventions

This asks:

- which current repair objectives change judge behavior?
- which cue pathways do those repairs seem to affect?
- which repair components are necessary?
- how should the repair objective be redesigned after tracing?

This program is diagnostic for now. It should not define the ontology by
itself.

## Current Phase Model

The project should be executed in five phases.

### Phase A: Build The Ontology

- `D1`: candidate atom inventory
- `D2`: bundle validation on broad discovery corpora

### Phase B: Establish Model Contrasts

- train `J0`
- train `Jrepair-all`
- train a small number of leave-one-out repair variants

### Phase C: Ecological Screening

- run `D3` on `J0`, `Jrepair-all`, and the first-wave ablations
- identify which candidate D4 targets are active and intervention-sensitive

### Phase D: Mechanistic Tracing

- localize D4 atoms and bundles in Gemma
- compare `J0`, `Jrepair-all`, and selected ablations
- run cross-layer matching, feature search, and activation comparisons

### Phase E: Mechanism-Informed Intervention Redesign

- redesign the repair target around traced D4 objects
- test whether the new intervention is more specific to the theorized failure
  mode
- compare it against the broad legacy repair families

The final intervention target should only be locked after the first mechanistic
pass.

## Ontology Backbone

### D1: Candidate Atom Inventory

Atoms are seeded from formal NLP and adjacent literature, not from adapter
ablations.

Current atom sources:

- register variation / multidimensional analysis
- metadiscourse and stance
- authorship / stylometry / discourse coherence
- human-vs-LLM difference work

The primitive object is the **atom**, not a broad label like
`academic_formality`.

### D2: Bundle Validation

Bundle discovery should be based on broad corpora, not Laurito alone.

Current discovery design:

- discovery core:
  - HC3
  - H-LLMC2
  - HAP-E
- domain bolsters:
  - PubMed abstracts
  - CMU Movie Summary
  - Amazon product metadata
- ecological holdout:
  - Laurito paper / product / movie

Principle:

- use broad domain-based samples for co-occurrence discovery
- use paired/control corpora for confirmation
- use Laurito for ecological relevance, not ontology discovery

### D3: Ecological Bridge

`D3` is the bridge between ontology and mechanism.

It asks:

- which atoms / bundles matter in real judge choices?
- how do those signatures differ across domains?
- how do they change across model variants?

`D3` is:

- an ecological screening layer
- a source of mechanistic prioritization

`D3` is not:

- final proof of ontology
- final proof of mechanism
- final intervention result

### D4: Reduced Tracing Ontology

`D4` is the first mechanistic target set.

It is provisional until frozen against:

- `J0`
- `Jrepair-all`
- the first-wave ablations

The main rule is:

`Prefer local atom targets over broad register labels.`

The project should now use a **broad and cautious D4 bundle panel** while still
allowing a smaller number of high-yield families to anchor the paper narrative.

The adapter ablations only probed a subset of the ontology. They should guide
mechanistic prioritization, but they should not define the full tracing panel by
themselves.

### Current D4 Primary Tracing Bundles

1. `formal_information_packaging`
2. `certainty_and_stance_calibration`
3. `professional_self_presentation`
4. `enumerative_assistant_packaging`
5. `safety_and_compliance_packaging`

### Current D4 Secondary Tracing Bundles

6. `narrative_and_summary_packaging`
7. `benefit_sales_packaging`

### Current D4 Priority Atoms

- `nominalization_patterns`
- `formal_connectives`
- `authoritative_certainty`
- `booster_certainty_markers`
- `self_mention_markers`
- `bullet_or_list_structure`
- `helpful_assistant_wrapper`
- `sentence_length_balance`
- `compression_vs_elaboration`
- `disclaimer_lexicon`
- `compliance_or_safety_stance`

### Readout-Only Variables

Use these as aggregate readouts, not as first mechanistic primitives:

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

### D4 Tracing Principle

The first mechanistic pass should trace a **broad atom family**, not only the
families that happened to be directly isolated by the current repair ablations.

Operational rule:

- trace many atoms
- organize them into a cautious bundle panel
- use a smaller number of high-yield bundles as guiding examples in the paper

The central paper story should therefore emerge from results, not be
pre-committed to a single flagship family such as `academic_formality`.

## Repair Matrix

The current repair matrix is a **diagnostic intervention suite**, not yet the
final ideal intervention suite.

Its role is:

1. establish an anchor judge
2. establish a full repair
3. measure sensitivity to removing individual repair components
4. create informative mechanistic contrasts

It should not be treated as:

- the ontology itself
- the final intervention design
- a one-to-one expression of D4

### Canonical Model Roles

#### `J0`

The neutral anchor judge.

Training:

- SHP preference supervision
- HelpSteer2 anchor supervision

No:

- paired invariance
- cue-adversarial removal

Configs:

- `configs/experiments/j0_anchor_v1.json`
- runtime-equivalent H100 fallbacks:
  - `configs/experiments/j0_anchor_v1_h100safe.json`
  - `configs/experiments/j0_anchor_v1_h100compact.json`

The fallback configs should be treated as the same anchor condition, not as new
scientific conditions.

#### `Jrepair-all`

The full current repair judge.

Training:

- SHP preference supervision
- HelpSteer2 anchor supervision
- paired invariance
- cue-adversarial removal

Config:

- `configs/experiments/jrepair_all_v1.json`

### First-Wave Leave-One-Out Suite

The current minimal first-wave suite is:

- `J0`
- `Jrepair-all`
- `jrepair_loo_joint_academic_formality_v1`
- `jrepair_loo_cue_template_boilerplate_v1`
- `jrepair_loo_cue_hedging_certainty_v1`

Optional extensions after that:

- `jrepair_loo_cue_narrative_packaging_v1`
- `jrepair_loo_cue_promotional_sales_tone_v1`

This matrix is intentionally small. The ontology should stay broad, but the
adapter suite should stay narrow.

Do **not** train one adapter per atom or one adapter per candidate bundle.

## Why The Adapter Matrix Stays Small

The project should not try to solve ontology discovery and intervention design
through an exhaustive adapter grid.

Instead:

- many atoms can be traced inside the same small set of models
- only a few intervention variants are needed to create interpretable
  perturbations

Current design logic:

- broad ontology
- small repair matrix
- broader tracing set within the trained models

## Current Evaluation Design

The standard full reward evaluation suite includes:

- `pref_retention`
- `style_sensitivity`
- `laurito_bias`
- `reward_benchmarks`
- `triads` if rewrite inputs exist

### Laurito Evaluation

Laurito evaluation is mixed across:

- `paper`
- `product`
- `movie`

and should be interpreted both:

- overall
- per domain

### Reward Benchmarks

Current benchmark suite:

- `arc_challenge`
- `hellaswag`
- `winogrande`
- `piqa`
- `social_iqa`
- `boolq`
- `mmlu`

For matrix sweeps, `BENCHMARK_MAX_EXAMPLES=100` is an acceptable first-pass
budgeting choice. This cap applies to the benchmark stage only, not to Laurito
or the pref/style stages.

## Mechanistic Novelty Target

The paper should be framed such that the **novel aspect is mechanistic
interpretation of judge bias**, not merely that a repair adapter changes scores.

The adapter matrix should be used to support the mechanistic story by showing:

- which ecological cue signatures exist in the anchor judge
- which are attenuated by the full repair
- which reappear when a repair component is removed

That creates grounds for mechanistic tracing.

The key claim should then come from:

- locating those signals in the model
- showing how they persist or shift across layers
- connecting them to judge decisions
- eventually using those insights to build a more specific intervention

The high-yield paper examples should be chosen from the mechanistic results, not
fixed in advance. The current best candidates for likely high-yield families
are:

- `professional_self_presentation`
- `safety_and_compliance_packaging`
- `certainty_and_stance_calibration`
- `enumerative_assistant_packaging`

while `formal_information_packaging`, `narrative_and_summary_packaging`, and
`benefit_sales_packaging` should remain in the first tracing panel even if they
are not all emphasized equally in the paper story.

## Real-World Ecological Scope

The ontology and tracing story should be grounded in real-world decision
contexts, not just abstract style labels.

Primary ecological contexts:

- Laurito paper abstracts
- Laurito product descriptions
- Laurito movie synopses

Additional broader text sources used for ontology grounding:

- HC3
- H-LLMC2
- HAP-E
- PubMed abstracts
- CMU Movie Summary
- Amazon product metadata

Motivating additional real-world evaluative contexts may include:

- job applications / professional summaries
- assistant-style answer judging

but the mechanistic paper should stay centered on the established datasets and
the derived ontology.

## End-To-End Consistency Rules

To keep the full experiment design coherent:

1. Do not use repair ablations to define the ontology from scratch.
2. Use broad corpora for ontology discovery and Laurito for ecological
   screening.
3. Use `J0` as the anchor for cue reliance claims.
4. Use `Jrepair-all` and the small leave-one-out suite as diagnostic
   perturbations.
5. Freeze D4 only after `J0` / `Jrepair-all` / first-wave `D3` comparisons are
   available.
6. Keep the mechanistic section focused on atom-to-bundle pathways, not just
   coarse bundle labels.
7. Redesign the final intervention target only after the first tracing pass.

## Current Practical Risks

### 1. Memory Fragility Of `J0`

`J0` has been memory-fragile despite 4-bit loading because the dominant cost is
training-time activations, not model weights. Dynamic padding and MLP-target
LoRA make the run sit close to the VRAM ceiling.

Operational rule:

- prefer `j0_anchor_v1_h100compact.json` if needed
- do not reinterpret the fallback configs as distinct scientific conditions

### 2. Legacy Naming

Older internal names:

- `m2_full_v1`
- `m3_full_v1`

should now be treated as historical equivalents of:

- `Jrepair-all`
- `jrepair_loo_joint_academic_formality_v1`

Prefer the new naming in all current analysis and writing.

### 3. Over-Expansion Of Adapter Variants

Do not expand the repair grid merely because the ontology is rich. The richer
set belongs in tracing, not in training dozens of models.

## Immediate Next Step

The first-wave adapter suite has now been trained, evaluated, and scored through
`D3`.

The next operational step is the first canonical D4 tracing pass:

1. compare ecological cue signatures across:
   - `J0`
   - `Jrepair-all`
   - `jrepair_loo_joint_academic_formality_v1`
   - `jrepair_loo_cue_template_boilerplate_v1`
   - `jrepair_loo_cue_hedging_certainty_v1`
2. freeze the first canonical mechanistic target list from those comparisons
3. materialize the D4 dataset pack:
   - `atom_probe_set`
   - `laurito_ecology_set`
   - `rewrite_control_set`
   - `content_anchor_set`
4. run a `J0`-first atom-recovery and layer-selection pass on the frozen pack
5. then expand the contrast set to:
   - `Jrepair-all`
   - `jrepair_loo_cue_template_boilerplate_v1`
   - `jrepair_loo_cue_hedging_certainty_v1`
   - `jrepair_loo_joint_academic_formality_v1` as a positive-control readout
6. let the main paper families emerge from tracing results rather than fixing
   them in advance

## D4 Mechanistic Implementation Rules

### Toolset

The D4 implementation should use the following tool stack:

1. Gemma Scope SAEs as the main feature basis
2. sparse feature probes for atom labels, bundle labels, authorship labels, and
   judgment-shift labels
3. cross-layer and cross-run feature matching
4. targeted feature ablation / damping / steering
5. run-level comparison between `J0`, `Jrepair-all`, and selected ablations

### Model Strategy

The first canonical mechanistic pass should be **9B-first**.

Use:

- `Gemma 2 9B` as the primary tracing model
- `Gemma 2 2B` only for method pilots or scale-robustness checks
- `Gemma 2 27B` only as a later confirmatory scaling test

Reason:

- Gemma Scope provides comprehensive SAE coverage for Gemma 2 `2B` and `9B`
- for `27B`, only select residual layers are available, so it is not the right
  first microscope

Operational rule:

- first establish the mechanism on `9B`
- then use `2B` and `27B` only if the same atoms and bundles appear recoverable
  and interpretable there

### Which SAE Basis To Prefer

For the reward-judge experiments, the backbone is Gemma 2 `9B-it` plus LoRA
adapters.

Therefore:

- use `Gemma Scope 9B IT residual SAEs` as the exact-backbone primary basis for
  `J0`, `Jrepair-all`, and the ablations
- use the richer `Gemma Scope 9B PT` all-layer residual / attention / MLP SAEs
  as a secondary exploratory microscope when broader layer and submodule
  coverage is needed

This distinction is important:

- exact backbone match is better for direct judge interpretation
- broader all-site PT coverage is better for richer exploratory localization

### D4 Abstraction Level

The first J0 atom-recovery sweep shows many D4 atoms are most recoverable in
late residual layers. This should not be treated as a failure of the ontology.

Operational interpretation:

- low-level text evidence such as punctuation, bullets, first-person markers,
  lexical certainty markers, and connective tokens may be present earlier
- D4 atoms are allowed to be **composed surface-cue concepts**, not only raw
  token detectors
- D4 bundles are higher-level recurring combinations of atoms, not single
  primitive features
- if an atom is only strong in late layers, trace it as a composed feature or
  short late-layer pathway rather than forcing an early-layer explanation

The right abstraction is therefore:

1. token or construction evidence
2. atom-level feature activations
3. bundle-level grouped activations
4. judge-decision contribution

Do not require every atom to appear as a clean early residual feature. Some
surface cues are syntactic or lexical and may appear early; others are register,
stance, or rhetorical-packaging abstractions and may only become linearly and
SAE-recoverable after the model has composed local evidence into a judgement-
relevant representation.

### Bundle Formation Principle

D4 bundles should be justified by both:

- ontological prior from linguistics / discourse analysis / register theory
- empirical co-occurrence and ecological relevance from D2 / D3 / D4

Neither source is sufficient alone.

Use ontology to prevent arbitrary post-hoc clusters. Use co-occurrence and
model evidence to prevent purely hand-written categories from becoming the
paper story without empirical support.

### SAE Feature Analysis Spec

The next implementation stage is `D4-SAE-1`: feature localization and feature
set construction.

Inputs:

- `data/derived/d4_dataset_pack_v1/manifest.json`
- `artifacts/mechanistic/d4_j0_atom_recovery_v1*/best_layers_by_atom.csv`
- `artifacts/mechanistic/d4_j0_atom_recovery_v1*/laurito_transfer_metrics.csv`
- `artifacts/reward/j0_anchor_v1_h100compact`
- later: `Jrepair-all` and first-wave ablations

Primary atom/layer candidates should include both:

- high recovery and high Laurito transfer atoms
- deliberately diverse ontology coverage atoms, even if they are not the top
  recovery scores

Initial high-yield candidates from the J0 sweep:

- `compression_vs_elaboration`
- `narrative_engagement_stance`
- `quotation_style`
- `professional_self_presentation`
- `participial_modifier_usage`
- `complex_noun_phrase_chains`
- `institutional_impersonality`
- `nominalization_patterns`
- `self_mention_markers`
- `bullet_or_list_structure`
- `formal_connectives`
- `hedge_markers`
- `epistemic_hedging`
- `promotional_adjectives`

Candidate layers:

- dense late sweep: `39`, `40`, `41`, `42`
- late controls: `32`, `36`
- mid controls: `16`, `20`, `24`, `28`
- early controls: `4`, `8`, `12`
- optional minimal baseline: `1`

Earlier layers must be swept at this stage. The expected result is that some
token-local evidence appears earlier while the more composed atom/bundle
representations peak late. If early layers are weak, that should be an
empirical result rather than an assumption.

Feature analysis outputs:

1. `sae_atom_feature_scores.csv`
   ranked SAE features per atom and layer
2. `sae_bundle_feature_scores.csv`
   grouped feature sets per D4 bundle
3. `sae_feature_examples.json`
   top activating examples by feature across atom-probe and Laurito texts
4. `sae_laurito_decision_alignment.csv`
   feature activation differences between chosen/rejected sides and judge score
   margins
5. `sae_content_utility_overlap.csv`
   overlap with content-anchor chosen/rejected utility signal
6. `sae_feature_set_manifest.json`
   frozen candidate feature sets for causal ablation

Ranking criteria:

- atom-label separation on atom-probe texts
- correlation with Laurito atom scores
- pair-side activation asymmetry aligned with judge preference
- cross-domain stability across paper / product / movie
- low or explicitly quantified overlap with content-anchor utility
- interpretability from top activating examples

The SAE stage should not immediately claim circuits. It first produces
candidate feature sets. Causal claims require the later ablation / damping /
steering ladder.

### Mechanistic Permutability Use

Mechanistic permutability / SAE feature matching is applicable here, but only
after feature localization has produced stable candidates.

Use it to ask:

- whether the same atom-like feature persists across adjacent late layers
- whether bundle-level signals are built from matched feature families across
  layers
- whether repair adapters attenuate, shift, or replace the same feature family

Do not use cross-layer matching as the first discovery method. First discover
features from labels and ecological alignment; then use matching to describe
feature evolution and short late-layer pathways.

### D4 Dataset Pack

The first mechanistic pass should use four dataset slices:

1. `atom_probe_set`
   balanced excerpt-level texts from the ontology/discovery corpora for atom and
   bundle prediction
2. `laurito_ecology_set`
   the full Laurito decision set with pair-level and text-side scores
3. `rewrite_control_set`
   content-preserving rewrite controls when available
4. `content_anchor_set`
   broad content-sensitive pairs or benchmark slices used to check that traced
   features are not just utility features in disguise

### First Canonical D4 Comparisons

The first mechanistic pass should compare at least:

- `J0`
- `Jrepair-all`
- `jrepair_loo_cue_template_boilerplate_v1`
- `jrepair_loo_cue_hedging_certainty_v1`

Keep:

- `jrepair_loo_joint_academic_formality_v1`

as a positive-control readout model, but not as the sole central contrast.

### First Causal Ladder

After feature localization, causal work should proceed in this order:

1. random matched-feature ablation
2. top atom-feature ablation
3. bundle-level grouped feature ablation
4. steering / damping comparison
5. later adapter redesign

### Model-Expansion Decision Rule

Move from `9B` to `2B` or `27B` only if:

- atom probes are stable
- bundle-family localization is interpretable
- top features transfer reasonably across domains and runs

If those conditions are not met, do not expand model scale yet.

## Repository Rules

### Code And Paths

- keep paths relative where possible
- avoid committing large raw data
- put derived datasets under `data/derived/`
- document provenance for new datasets and corpora

### Python Style

- follow PEP 8
- use 4-space indentation
- prefer `snake_case`
- add type hints and docstrings for new public code

### Testing

- use lightweight tests in `tests/`
- prefer fast schema and config validation checks

### Reproducibility

- keep new scripts runnable via `python path/to/script.py --help`
- preserve config-driven experiment definitions
- if a runtime fallback config is introduced, document whether it is
  scientifically equivalent or a new condition
