# Workshop Implementation Path: From Candidate SAE Features To Perturbation Evidence

Status: active workshop-paper path as of 2026-05-07.

This file narrows the broader scientific plan in `IMPLEMENTAION_PATH.md` to the
shortest credible path from the current D4 findings to a workshop paper. The
target is not a full repair system. The target is evidence that surface-cue
features and atom-to-bundle feature sets measurably affect a reward judge on
content-matched human-vs-LLM pairs while preserving general preference ability.

## Current Empirical State

The strongest current result is the J0 broad human-vs-LLM candidate alignment
run:

- `artifacts/mechanistic/d4_j0_human_llm_candidate_alignment_strat10k_v3`
- `n_pairs = 10000`
- source mix:
  - `hc3 = 2797`
  - `hc3_plus = 2500`
  - `h_llmc2 = 1374`
  - `hape = 3329`
- J0 LLM choice rate: `0.7857`
- mean LLM-minus-human margin: `0.3317`
- discovered candidate tail exceeds matched random controls:
  - discovered p90 absolute controlled rho: `0.0943`
  - random-control p90 absolute controlled rho: `0.0471`
  - discovered p90 choice-AUC delta: `0.0937`
  - random-control p90 choice-AUC delta: `0.0506`

The important change relative to the HC3-only run is that top features now have
nontrivial support across HC3, HC3+, H-LLMC2, and HAP-E. The result should be
framed as cross-corpus cue-alignment evidence, not only as HC3 origin
detection.

## Workshop Claim

Main claim:

> A reward judge's LLM-vs-human margins are partly shaped by sparse features
> that encode surface and discourse cues. These cues can be organized into
> atom-to-bundle hypotheses, and damping the corresponding feature sets should
> reduce cue-aligned judge preference while preserving general preference
> behavior.

Workshop-safe wording:

- "feature-aligned evidence", not "complete circuit"
- "causal perturbation evidence", not "full repair"
- "surface-cue reliance", not "bias elimination"
- "bundle hypothesis", not "exhaustive ontology"

## Frozen Candidate Bundles

For the workshop paper, a bundle should require at least three distinct atoms.
Feature sets with fewer than three atoms can appear as atom-family controls or
diagnostic ablations, but should not carry the atom-to-bundle narrative.

### Bundle A: Structured Assistant Packaging

Atoms:

- `bullet_or_list_structure`
- `balanced_multi_part_completion`
- `helpful_assistant_wrapper`
- `sentence_length_balance`
- `compression_vs_elaboration`
- optional support: `nominalization_patterns`, `participial_modifier_usage`

Primary features:

- `L35 F7508`
- `L38 F1378`
- `L34 F10845`
- `L36 F903`
- `L35 F5154`
- `L38 F12810`
- `L41 F14277`
- `L40 F11031`
- `L35 F12093`
- `L37 F4255`

Rationale:

- strongest global controlled alignment
- strong HC3+ confirmation
- strong H-LLMC2 support
- HAP-E weaker but still validates the broader polished-completion family

### Bundle B: Formal And Institutional Packaging

Atoms:

- `formal_connectives`
- `institutional_impersonality`
- `professional_self_presentation`
- `nominalization_patterns`
- `complex_noun_phrase_chains`
- optional support: `entity_grid_coherence`, `subordination_density`

Primary features:

- `L34 F7691`
- `L40 F15478`
- `L39 F15970`
- `L20 F4882`
- `L16 F4882`
- `L24 F8530`
- `L26 F12358`
- `L8 F836`
- `L12 F10286`
- `L4 F2304`

Rationale:

- best paper-facing bundle because HAP-E also supports it
- less reducible to list formatting alone
- interpretable as professional polish that should not always determine content
  quality

### Bundle C: Benefit And Value Framing

Atoms:

- `benefit_first_packaging`
- `feature_benefit_call_to_value_script`
- `promotional_adjectives`
- `enthusiasm_or_salesmanship`

Primary features:

- `L39 F11694`
- `L38 F1406`
- `L37 F7036`
- `L16 F13713`
- `L4 F6442`
- `L12 F15740`
- `L28 F11816`
- `L27 F9305`
- `L16 F9623`

Rationale:

- clearly cross-source after adding HC3+ and H-LLMC2
- useful contrast to formal packaging because it is more promotional and less
  purely structural

### Secondary Atom Families

Use these as controls or secondary analyses unless they pass a stronger
cross-source screen:

- stance calibration: `epistemic_hedging`, `hedge_markers`,
  `evidential_markers`
- narrative/quotation packaging: `quotation_style`,
  `compression_vs_elaboration`, `sentence_length_balance`

## Required Experiments

### E1: Candidate Freeze And Bundle Registry

Goal:

- freeze the candidate list used by all perturbation runs
- map features to atom families and bundle hypotheses
- keep random controls fixed

Implementation:

- Create a registry file under
  `artifacts/mechanistic/d4_j0_bundle_candidate_registry_v1`.
- Inputs:
  - `candidate_feature_human_llm_alignment.csv`
  - `candidate_feature_source_alignment.csv`
  - `candidate_feature_registry.csv`
  - merged SAE atom and bundle feature scores
- Output tables:
  - `bundle_candidate_features.csv`
  - `atom_candidate_features.csv`
  - `matched_random_feature_controls.csv`
  - `feature_freeze_manifest.json`

Selection rules:

- feature is intervention-eligible if:
  - `n_sources_with_min_pairs >= 4`
  - source sign consistency `>= 0.8` preferred, `>= 0.6` allowed only if clearly
    marked source-sensitive
  - effect exceeds the matched random-control p90 in either controlled rho or
    choice-AUC delta
  - feature has at least one atom or bundle annotation
- bundle is paper-eligible if:
  - at least three distinct atoms are represented
  - at least three SAE features are available
  - at least three source datasets contribute signal

Workshop minimum:

- Bundle A and Bundle B must pass.
- Bundle C is useful but not mandatory for the main result.

### E2: Automated Content-Matched Text Perturbation

Manual dataset creation is out of scope. Text perturbations must be automated.
Their role is supporting evidence and sanity checking, not the main causal
claim.

Goal:

- produce automated cue-plus and cue-minus pairs for selected bundles
- verify that target SAE features move in expected directions
- avoid spending manual time on bespoke examples

Inputs:

- v3 pair file:
  `data/derived/d4_human_llm_alignment_pairs_strat10k_v3/pairs.jsonl`
- candidate registry from E1

Perturbation types:

- `atom_plus`: add one atom cue
- `atom_minus`: remove or neutralize one cue
- `bundle_plus`: add at least three atoms from one bundle
- `bundle_minus`: neutralize at least three atoms from one bundle
- `edit_budget_control`: same approximate token change, no target cue

Automation strategy:

- Prefer deterministic rule-based transforms for obvious atoms:
  - list and bullet conversion
  - formal connective insertion/removal
  - benefit-first sentence reordering
  - hedging insertion/removal
  - quotation-style insertion/removal
- Use an LLM rewrite only where deterministic transforms are poor, and keep it
  fully automated:
  - fixed prompt templates
  - no manual filtering
  - reject by automatic checks only

Automatic rejection checks:

- token length ratio outside `[0.8, 1.25]`
- exact duplicate output
- target cue did not change by lexical or atom-label detector
- high off-target edit distance relative to cue edit
- optional semantic-similarity score below threshold if an embedding model is
  available in the container

Outputs:

- `automated_perturbation_pairs.jsonl`
- `automated_perturbation_manifest.json`
- `perturbation_atom_label_delta.csv`
- `perturbation_sae_activation_delta.csv`

Workshop use:

- one table showing that automated bundle-plus rewrites increase target bundle
  activations more than controls
- one qualitative row per bundle can be included, but the claim should not rely
  on hand-picked examples

### E3: SAE Feature Perturbation In The Judge

This is the central causal experiment.

Goal:

- test whether dampening atom or bundle SAE features changes J0's alignment on
  content-matched human-vs-LLM pairs
- verify that general preference ability is retained

Intervention primitive:

- use SAE feature damping, not hard zero ablation by default
- damping levels: `0.25`, `0.5`, `0.75`, `1.0`
- run hard ablation only as an ablation point, not as the main setting

Intervention units:

- atom-level:
  - one atom's candidate feature set
  - used for features that do not form a three-atom bundle
- bundle-level:
  - union of validated bundle features
  - primary paper result
- matched random controls:
  - same layer distribution
  - same number of features
  - same SAE release
  - no annotation overlap with target bundle

Primary evaluation pairs:

- v3 broad content-matched pair file:
  `data/derived/d4_human_llm_alignment_pairs_strat10k_v3/pairs.jsonl`
- report per dataset:
  - HC3
  - HC3+
  - H-LLMC2
  - HAP-E

Retention pairs:

- content-anchor utility pairs from the recovered split
- general preference benchmark already used for reward evaluation, if available
- Laurito ecological pairs only as an auxiliary screen, not as the retention
  benchmark

Metrics:

- target alignment:
  - change in J0 LLM-choice rate
  - change in mean LLM-minus-human reward margin
  - change in activation-delta to reward-margin correlation
  - source-level effect consistency
- retention:
  - general preference AUC or pair accuracy before vs after intervention
  - Spearman/Pearson correlation of original and perturbed reward margins
  - reward score drift on non-target controls
- specificity:
  - target feature set effect minus matched-random effect
  - bundle intervention effect greater than individual atom median effect
  - target source pairs affected more than content-anchor utility pairs

Success criteria:

- at least one bundle intervention reduces cue-aligned J0 LLM preference more
  than matched random controls
- general preference AUC drops by no more than `0.02`, or margin correlation
  remains at least `0.95`
- effect appears in at least three source datasets
- bundle intervention is stronger than the median of its atom-level
  interventions

### E4: Ablations For Workshop Credibility

Required ablations:

- random same-layer SAE features
- same number of features but shuffled across bundle labels
- atom-level interventions for each feature family in the bundle
- damping-strength sweep
- source-dataset stratification
- length-controlled analysis before and after intervention

Optional ablations if time allows:

- compare J0 to a repair or leave-one-out judge as a diagnostic contrast
- base-model no-adapter feature scoring only as a representation contrast, not
  as the main result
- automated text perturbation plus feature intervention on the same pairs

Do not spend time on:

- new adapter grids
- manual counterfactual dataset creation
- full feature-card falsification for every feature
- causal claims for features that have no intervention result

## Implementation Tasks

### Task 1: Registry Builder

Add:

- `src/aisafety/scripts/build_d4_bundle_candidate_registry.py`

Responsibilities:

- read v3 alignment and source-alignment outputs
- freeze Bundle A/B/C feature sets
- select atom-level and bundle-level matched random controls
- write manifest with all thresholds

Tests:

- registry includes required bundles with at least three atoms
- matched controls preserve layer counts
- source-sensitive features are labeled

### Task 2: Deterministic Surface-Counterfactual Builder

Add:

- `src/aisafety/mech/counterfactuals.py`
- `src/aisafety/scripts/build_d4_surface_counterfactual_pairs.py`
- `cluster/lrz/d4_surface_counterfactual_pairs.sbatch`

Responsibilities:

- sample content-matched human/LLM pairs from the v3 broad pair file
- create deterministic cue-increase and cue-decrease variants for structured
  assistant packaging, formal/institutional packaging, and benefit/value
  framing
- write explicit skip summaries for unchanged, too-short, and length-filtered
  transforms
- avoid LLM-generated rewrites in v1 so that the audit does not import another
  model's style priors

Tests:

- transformations preserve pair ids and role/source metadata
- emitted variants are nonempty and changed
- length-ratio and duplicate filters work
- skipped transforms are counted by axis and direction

### Task 3: Surface-Counterfactual Audit Runner

Add:

- `src/aisafety/scripts/run_d4_surface_counterfactual_audit.py`
- `cluster/lrz/d4_surface_counterfactual_audit.sbatch`

Responsibilities:

- load J0 reward scorer
- score base and variant text for each counterfactual
- compute reward deltas, length-controlled reward deltas, and source-stratified
  summaries
- load frozen bundle features and compute matching SAE bundle activation deltas
- keep text-level counterfactual reward effects separate from feature-muting
  effects

Outputs:

- `surface_counterfactual_scores.csv`
- `axis_summary.csv`
- `source_summary.csv`
- `bundle_activation_delta.csv`
- `manifest.json`

Tests:

- audit CLI imports and help works locally
- length-controlled summaries work on synthetic rows
- SAE scoring remains optional through a skip flag

### Task 4: Readout-Space Nulling Runner

Add:

- `src/aisafety/mech/readout_nulling.py`
- `src/aisafety/scripts/run_d4_readout_surface_nulling.py`
- `cluster/lrz/d4_readout_surface_nulling.sbatch`

Responsibilities:

- fit pooled-state cue directions from train-split counterfactual hidden-state
  deltas
- orthogonalize the selected cue directions
- project the cue subspace out before J0's scalar value head
- evaluate counterfactual reward-delta reduction, broad human-vs-LLM margin
  shift, and SHP-2 preference retention

Outputs:

- `surface_directions.pt`
- `surface_direction_summary.csv`
- `counterfactual_nulling_scores.csv`
- `pair_nulling_scores.csv`
- `preference_retention_scores.csv`
- `nulling_summary.csv`
- `manifest.json`

Tests:

- projection removes a synthetic cue direction while preserving orthogonal
  signal
- zero or duplicate directions are handled safely
- CLI imports and help works locally

### Task 5: Intervention Readout

Add:

- `src/aisafety/scripts/inspect_d4_feature_intervention.py`

Outputs:

- `intervention_readout.md`
- paper-ready tables:
  - bundle effect table
  - retention table
  - random-control comparison table
  - per-source table

## LRZ Run Order

1. Build or refresh registry on CPU.
2. Build deterministic surface counterfactuals on CPU.
3. Run a small GPU counterfactual audit with `MAX_COUNTERFACTUALS=500`.
4. If the audit is nonempty, run the full counterfactual audit with SAE bundle
   activation deltas.
5. Run readout-space nulling:
   - fit on half of the counterfactual rows
   - evaluate on held-out counterfactual rows
   - evaluate broad human-vs-LLM margin shift
   - evaluate SHP-2 preference retention
6. Keep SAE feature muting as a diagnostic baseline, not the only intervention
   result.
7. Generate readout and freeze paper tables.

## Paper Narrative

### Abstract Claim

Reward judges can rely on sparse surface-cue features that distinguish LLM from
human texts across multiple corpora. SAE features make those cues interpretable
as atoms and bundles. Deterministic text counterfactuals test whether the judge
rewards those surface changes under approximate content preservation, and
readout-space nulling tests whether suppressing the corresponding pooled-state
subspace reduces cue sensitivity while retaining ordinary preference behavior.

### Figure And Table Plan

Figure 1:

- pipeline: atom ontology -> SAE candidates -> broad alignment ->
  counterfactual audit -> readout nulling -> retention check

Table 1:

- v3 broad alignment dataset composition and random-control baseline

Table 2:

- frozen Bundle A/B/C candidates with atoms, features, source consistency, and
  alignment strength

Table 3:

- surface counterfactual audit:
  - axis and direction
  - reward delta
  - length-controlled reward delta
  - matching SAE bundle activation delta
  - source consistency

Table 4:

- readout nulling and retention results:
  - counterfactual reward-delta shrinkage
  - human-vs-LLM margin shift
  - general preference accuracy before/after
  - preference margin drift

Small qualitative panel:

- one automated counterfactual per bundle showing target activation movement
  and J0 score movement

### Minimum Workshop Result

The paper is viable if:

- Bundle A and Bundle B have frozen multi-atom definitions
- v3 alignment table is reported
- deterministic counterfactuals show a source-consistent J0 reward delta for at
  least one surface axis
- SAE bundle activations move in the expected direction for that axis
- readout nulling reduces counterfactual cue sensitivity while preserving most
  ordinary preference behavior
- SAE feature muting is reported as a diagnostic baseline, even if its effect is
  weaker than readout nulling

The paper should be framed as a mechanistic case study, not a complete judge
debiasing method.

## Decision Points

If SAE bundle interventions do not move judge margins:

- report broad alignment as correlational and use perturbation failure as a
  limitation
- test whether text counterfactuals move activations but not reward
- do not claim causal judge reliance

If readout nulling moves margins but damages retention:

- frame as evidence that the features are entangled with useful preference
  computation
- report retention failure clearly
- avoid repair language

If counterfactuals move reward but readout nulling does not:

- center the paper on measurement rather than mitigation
- treat readout nulling as a negative baseline
- consider a later surface-invariant adapter trained on explicit axes

If Bundle B works and Bundle A is too formatting-heavy:

- center the paper on formal/institutional packaging
- keep list-structure features as a diagnostic control

If only HC3/HC3+ move and HAP-E does not:

- claim cross-QA and semantic-control evidence only
- do not claim broad cross-domain generality
