# Step-by-Step Experiment Plan

## Purpose

This document is the execution plan for the project.

It turns the higher-level research framing into a concrete sequence of experiments, deliverables, and decision points.

It is designed to work even if the exact final cue-family set is not yet fixed.

## Project goal

The goal is to determine whether LLM judges rely on internal representations of authorship-correlated surface cues rather than grounded utility, and whether those cue pathways can be repaired without materially degrading content-sensitive judgment.

The project therefore has two linked outputs:

1. a repaired judge
2. a mechanistic account of what was repaired

## Guiding principle

Do not block the project on final cue-family selection.

Instead:

1. start with a stable provisional cue set
2. run the first behavioral and mechanistic passes
3. prune or expand cue families only after seeing which ones produce stable, interpretable signal

## Provisional cue-family set

### Flagship families

- `academic_formality`
- `template_packaging`

### Secondary families

- `promotional_tone`
- `narrative_packaging`
- `fluency_polish`
- `verbosity_compression`
- `hedging_certainty`
- `safety_corporate_tone`
- `paraphrase_surface`

### Why this split

The flagship families are sufficient to keep the project moving:

- `academic_formality` is the strongest domain-specific positive control
- `template_packaging` is the strongest cross-domain candidate

The secondary families remain in scope, but they should not delay the first mechanistic runs.

## Phase 0: Freeze the baseline

Goal:

- lock the first runnable training and evaluation stack before the project expands further

### Inputs

- current training configs
- cue corpus v2
- style-group corpora
- full eval suite

### Tasks

1. finish the first `Broad Cue Repair` and `No-Academic Repair` runs
2. archive their configs, checkpoints, metrics, and eval outputs in stable artifact directories
3. record exact command lines and environment information
4. export a single summary table for:
   - Laurito bias by domain
   - SHP retention
   - HelpSteer2 retention
   - benchmark retention

### Deliverables

- frozen baseline checkpoints
- frozen evaluation summary
- exact run metadata

### Exit criterion

You have at least one repaired checkpoint that improves bias metrics without obvious catastrophic capability loss.

## Phase 1: Finalize the first paper-facing comparison set

Goal:

- move from internal run IDs to a stable experimental comparison set

### Paper-facing names

- `Base Judge`
- `Counterfactual Repair`
- `Broad Cue Repair`
- `No-Academic Repair`

### Tasks

1. map current internal runs to those names
2. define which of them are part of the paper core
3. decide whether `Counterfactual Repair` needs to be rerun or can remain a secondary baseline

### Minimum paper core

- `Base Judge`
- `Broad Cue Repair`
- `No-Academic Repair`

### Preferred paper core

- `Base Judge`
- `Counterfactual Repair`
- `Broad Cue Repair`
- `No-Academic Repair`

### Exit criterion

There is a stable baseline ladder that can be reused for every later mechanistic comparison.

## Phase 2: Strengthen the cue-discovery substrate

Goal:

- ensure cue discovery is not overly tied to Laurito or one generator family

### Inputs

- Laurito domains
- HC3
- H-LLMC2
- broader abstract-heavy and parallel human-vs-LLM corpora when added

### Tasks

1. expand the abstract-heavy corpus first
2. keep prompt-matched multi-generator data as a priority
3. maintain balanced train/val/test splits by:
   - domain
   - generator family
   - human versus LLM source
4. re-run weak-label cue scoring after each meaningful data extension
5. track whether authorship leakage remains high across held-out generators

### Deliverables

- `cue_discovery_v3` or later corpus
- updated weak labels
- updated cue-leakage probe results

### Exit criterion

Cue discovery works on a substrate that is broader than Laurito and broader than one LLM family.

## Phase 3: Lock the first cue-family program

Goal:

- decide which families are treated as first-class mechanistic objects

### Tasks

1. promote `academic_formality` and `template_packaging` to flagship status
2. choose one third family from:
   - `promotional_tone`
   - `narrative_packaging`
3. keep the remaining families as support or exploratory families
4. define one negative control:
   - random matched features
   - or a weak family not expected to dominate Laurito bias

### Decision rule

Choose the third flagship family based on:

- weak-label quality
- cross-domain stability
- plausibility of internal interpretability
- ecological relevance to Laurito domains

### Deliverables

- a locked flagship cue-family list
- a support-family list
- a negative-control definition

### Exit criterion

The first mechanistic study can proceed with two guaranteed families and one additional test family.

## Phase 4: Rebuild the repair training matrix around the cue-family program

Goal:

- align repair runs with the mechanistic paper story

### Repair runs

1. `Base Judge`
2. `Counterfactual Repair`
3. `Broad Cue Repair`
4. `No-Academic Repair`
5. `No-Template Repair`
6. optional:
   - `No-Promotional Repair`
   - `No-Narrative Repair`

### Tasks

1. create configs for the missing ablation runs
2. keep the architecture fixed across runs
3. vary only the cue-family inclusion or exclusion
4. re-run the full evaluation suite on every kept run

### Deliverables

- stable repair comparison matrix
- eval summaries for every repair variant

### Exit criterion

You have a clean behavioral table showing which cue families matter most for repair.

## Phase 5: Build the mechanistic dataset pack

Goal:

- construct the exact examples needed for feature discovery, patching, and ablation tests

### Required dataset slices

1. `cue_probe_set`
- balanced examples with cue-family labels and authorship labels

2. `rewrite_control_set`
- paired content-preserving rewrites for flagship families

3. `judgment_shift_set`
- pairs where the judge changes preference under surface rewrites

4. `content_anchor_set`
- pairs where content quality differs while style is approximately controlled

5. `generator_transfer_set`
- matched prompts across multiple LLM generators

### Tasks

1. sample each set from existing corpora first
2. add generated examples only when strictly necessary
3. keep provenance and split membership explicit
4. freeze these sets for reproducibility

### Deliverables

- a documented mechanistic dataset pack

### Exit criterion

There is a reproducible dataset pack for all internal-state analyses.

## Phase 6: Run feature-localization studies

Goal:

- find internal features linked to the flagship cue families

### Primary method

- Gemma Scope SAEs on Gemma 2 9B

### Tasks

1. extract SAE activations for:
   - `Base Judge`
   - `Broad Cue Repair`
   - `No-Academic Repair`
2. train sparse-feature probes for:
   - cue-family labels
   - human versus LLM source
   - preference shift labels
3. rank features by:
   - cue selectivity
   - authorship selectivity
   - judgment-shift selectivity
4. identify top candidate features per family
5. compare candidate features across:
   - layers
   - submodules
   - domains
   - generators

### Analyses

1. layer localization
2. cross-domain transfer
3. cross-generator transfer
4. overlap between cue features and utility-sensitive features

### Deliverables

- top candidate feature lists by family
- localization plots
- transfer results

### Exit criterion

At least two flagship cue families yield stable, interpretable feature candidates.

## Phase 7: Analyze feature geometry

Goal:

- determine whether cue families form meaningful internal structure rather than isolated lists of features

### Tasks

1. cluster top cue-linked features
2. inspect neighborhood structure
3. test overlap between:
   - `academic_formality`
   - `template_packaging`
   - utility-linked features
4. use cross-layer feature matching where possible
5. test whether families are:
   - coherent clusters
   - overlapping bundles
   - or layer-evolving motifs

### Deliverables

- feature-neighborhood visualizations
- overlap statistics
- cross-layer matching results

### Exit criterion

You can say something nontrivial about how cue families are organized internally.

## Phase 8: Run causal feature ablations

Goal:

- establish that the localized features are actually used for judgment

### Ablation ladder

1. random matched-feature ablation
2. `academic_formality` feature ablation
3. `template_packaging` feature ablation
4. third-family feature ablation
5. combined-family ablation

### Tasks

1. ablate top candidate features during inference
2. compare effects on:
   - Laurito bias
   - style sensitivity
   - SHP retention
   - content anchor discrimination
3. run matched random-feature controls
4. repeat on at least one repaired and one unrepaired model

### Deliverables

- causal ablation table
- random-control comparison

### Exit criterion

Targeted feature ablation reduces bias more selectively than random ablation.

## Phase 9: Distinguish wrapper repair from deep repair

Goal:

- explain how the adapter changes behavior

### Tasks

1. compare feature activations across:
   - pretrained Gemma
   - Base Judge
   - Broad Cue Repair
   - No-Academic Repair
2. use layer patching to ask which layers restore old bias
3. test whether repair primarily:
   - attenuates cue features
   - shifts downstream readout from cue features
   - or both

### Interpretation targets

- late wrapper correction
- deeper representational attenuation

### Deliverables

- model-diff feature analysis
- layer-patching analysis

### Exit criterion

You can explain whether repair is mostly late-stage or more globally distributed.

## Phase 10: Optional steering comparison

Goal:

- compare learned repair to direct causal intervention

### Tasks

1. build cue-family steering or damping interventions
2. compare them to:
   - feature ablations
   - Broad Cue Repair
   - random-feature steering
3. test whether steering can mimic repair selectively

### Deliverables

- steering-versus-repair comparison

### Exit criterion

Either:

- steering gives a useful causal confirmation

or

- steering fails cleanly and becomes a negative result

## Evaluation block for every major phase

Every major model or intervention should be evaluated on the same four blocks.

### Block 1: behavioral repair

- Laurito bias by domain
- SHP preference retention
- HelpSteer2 utility retention
- HelpSteer2 attribute retention
- reward-benchmark sanity checks

### Block 2: invariance

- paired counterfactual style sensitivity
- held-out cue-family sensitivity
- held-out generator transfer

### Block 3: mechanistic evidence

- cue-feature probe quality
- layer localization
- cross-domain probe transfer
- cross-generator probe transfer
- feature geometry or overlap

### Block 4: causal evidence

- targeted feature ablations
- random controls
- patching or steering

## Step-by-step execution order

This is the recommended order of work.

1. finish and freeze the first repaired checkpoints
2. lock the paper-facing comparison set
3. strengthen the cue-discovery substrate
4. lock the first flagship cue families
5. build missing repair-ablation runs
6. freeze the mechanistic dataset pack
7. run feature-localization for `academic_formality`
8. run feature-localization for `template_packaging`
9. select the third family
10. run cross-domain and cross-generator probe transfer
11. run feature-geometry analysis
12. run targeted feature ablations plus random controls
13. compare repaired versus unrepaired models via patching and model diffing
14. add steering only if time remains

## Minimum viable paper path

If time becomes tight, the minimum strong path is:

1. `Base Judge` versus `Broad Cue Repair` versus `No-Academic Repair`
2. cue localization for `academic_formality` and `template_packaging`
3. random-control versus targeted feature ablation
4. one adapter-interpretation comparison

That is the minimum configuration that still supports a mechanistic paper claim.

## Stop-go criteria

### Continue with academic formality as a centerpiece if:

- it remains behaviorally important
- it yields stable feature candidates
- feature ablation affects paper-domain bias selectively

### Demote academic formality if:

- the effect does not replicate
- features are unstable across domains or runs
- template packaging turns out to explain more of the broad bias pattern

### Continue with feature-geometry analysis if:

- at least two families yield stable candidate features

### Skip or minimize steering if:

- feature localization or ablation evidence is already strong
- steering adds complexity without interpretive gain

## Final success condition

The project succeeds if it can support the following chain:

1. the judge shows a measurable behavioral bias
2. specific cue families explain part of that bias
3. those cue families have identifiable internal feature correlates
4. those features are causally involved in the judgment
5. repair changes those feature pathways without destroying general judgment competence

That is the core experimental path to a strong mechanistic interpretability result in this setting.
