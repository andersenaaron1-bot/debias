# Implementation Path: Atom-To-Bundle Surface-Cue Mechanistic Tracing

Status: active implementation path as of 2026-04-28.

Filename note: this file intentionally preserves the requested spelling
`IMPLEMENTAION_PATH.md`. Do not create a duplicate `IMPLEMENTATION_PATH.md`
unless the project renames this file in one cleanup change.

## Purpose

This document is the scientific and engineering path for understanding how
surface cue atoms become bundle-level signals that influence reward-judge
opinions in human-vs-LLM comparisons.

The immediate refactor is not a new model family, a new repair proposal, or an
adapter grid. It is a stricter mechanistic pipeline:

1. recover atom signals in the reward judge residual stream
2. localize atom-aligned sparse features
3. assemble atom features into bundle hypotheses
4. test whether bundle-level feature sets align with judge choices
5. only then run causal damping, ablation, or steering tests

The earlier failed run is treated as a recovery target, not as evidence. Any
utility-control result from the invalid split is retired.

## Scientific Contract

The main claim to build toward is:

> Reward judges can form preferences partly from recurring surface and
> discourse cues. These cues are measurable as literature-grounded atoms,
> cohere into bundles, are represented by sparse model features, and may align
> with judge decisions on ecological human-vs-LLM pairs.

The project must keep three levels separate:

- Atom: a local, operational cue such as list structure, hedging, nominalized
  phrasing, self-mention, compression, or disclaimer language.
- Bundle: a recurring cue constellation such as formal information packaging,
  enumerative assistant packaging, safety/compliance packaging, narrative
  packaging, or benefit/sales packaging.
- Opinion signal: reward-margin or LLM-choice behavior by a trained judge.

Do not infer causality from atom recovery, SAE feature separation, or Laurito
alignment alone. Causal language is reserved for intervention stages.

## Literature Basis

The ontology remains rooted in the bibliography under `paper/references.bib`.
The core atom and bundle families come from register variation, computational
sociolinguistics, metadiscourse, stylometry, discourse coherence, LLM-vs-human
text studies, and AI-AI bias evaluations.

Existing anchors:

- Register and multidimensional analysis: `biber1993register`,
  `passonneau2014biberredux`
- Computational sociolinguistics: `nguyen2016sociolinguistics`
- Metadiscourse and stance: `hyland2005metadiscourse`
- AI-AI bias and judge preference behavior: `laurito2025aiaibias`,
  `zheng2023mtbench`, `panickssery2024selfrecognition`,
  `lee2024remodetect`
- Human-vs-LLM corpora and style variation: `guo2023hc3`,
  `reinhart2024llmshumans`, `browndw2024hape`, `noepsl2025hllmc2`
- SAE and feature-circuit methods: `cunningham2023sae`,
  `gao2024scaling`, `lieberum2024gemmascope`,
  `marks2024sparsecircuits`, `dunefsky2024transcoders`,
  `kissane2024attention`, `balagansky2024permutability`

Recent-art additions used by this path:

- `karvonen2025saebench`: SAE proxy metrics do not automatically predict
  practical interpretability performance, so the project must evaluate features
  against task-relevant atom, ecological, decision, and utility controls.
- `zhang2025sarm`: reward-model interpretability through SAE feature spaces is
  a live research direction; this project keeps a post-hoc tracing design rather
  than replacing the reward architecture.
- `shu2025saesurvey`: SAE evaluation needs structural and functional checks,
  not only visually plausible feature examples.
- `ma2025falsifying`: feature explanations require close-negative
  falsification; top activations alone are insufficient.
- `farrell2025saeunlearning`: feature interventions can have side effects, and
  zero ablation is not automatically the right repair primitive. Damping is the
  default intervention candidate after localization.

## Retired Assumptions

The following historical assumptions are no longer project direction:

- `M2` and `M3` names as active scientific conditions. Use current `J*` names
  in writing and manifests.
- The full repair adapter as a final repair method. It remains a diagnostic
  contrast.
- One adapter per atom or per bundle.
- Treating the PT Gemma Scope SAE basis as exact-backbone evidence for an
  instruction-tuned reward judge. PT residual SAEs may be used as an
  exploratory dictionary only when the manifest states this explicitly.
- Treating broad register labels as first mechanistic primitives.
- Treating the invalid utility-control split as a usable result.
- Starting intervention claims before feature localization, feature
  interpretation, ecological alignment, judge-decision alignment, and content
  controls are complete.

Model and SAE targets must be read from run configs and run manifests. This
file should not hard-code a future architecture roadmap. If the model substrate
or SAE basis changes, update the run manifest and this path before interpreting
results.

## Current Assets

Primary local assets:

- D1 atom inventory:
  `data/derived/style_groups/candidate_atom_inventory_d1.tsv`
- D4 ontology config:
  `configs/ontology/d4_reduced_ontology_v1.json`
- D4 dataset-pack config:
  `configs/datasets/d4_dataset_pack_v1.json`
- Current D4 pack:
  `data/derived/d4_dataset_pack_v1/manifest.json`
- Residual atom recovery script:
  `src/aisafety/scripts/run_d4_atom_recovery.py`
- SAE feature analysis script:
  `src/aisafety/scripts/run_d4_sae_feature_analysis.py`

Current caveat:

- The first LRZ D4 pack contained the 27 atoms present in that manifest. It did
  not fully cover every safety/compliance atom in the current ontology config.
  Rebuild the D4 pack before treating safety/compliance as fully traced.

## Refactor Goal

The D4 scripts should become a composable mechanistic workflow rather than two
large monolithic scripts.

Target software modules:

- `src/aisafety/mech/d4_io.py`
  - read D4 manifests, JSONL rows, Laurito tables, and content anchors
  - validate required fields and path resolution
  - write run manifests and resumable status files
- `src/aisafety/mech/labels.py`
  - quantile atom labels
  - source- and item-type-aware splits
  - deterministic pair-level content-anchor splits
- `src/aisafety/mech/activations.py`
  - residual hidden-state extraction
  - token pooling policies
  - batch scheduling and cache-friendly chunking
- `src/aisafety/mech/sae.py`
  - SAE loading
  - hidden-layer to SAE-layer mapping
  - max, last-token, and span-local feature aggregation
  - explicit PT-vs-IT SAE basis metadata
- `src/aisafety/mech/probes.py`
  - sparse residual probes
  - content-anchor utility probes
  - model-selection summaries
- `src/aisafety/mech/feature_ranking.py`
  - atom feature separation
  - validation/test AUC
  - Laurito text transfer
  - Laurito pair-decision alignment
  - content-anchor overlap
- `src/aisafety/mech/bundles.py`
  - atom-feature to bundle-feature graph construction
  - bundle stability checks
  - cross-layer feature matching hooks
- `src/aisafety/mech/examples.py`
  - top activation examples
  - close negative examples
  - falsification packs
- `src/aisafety/mech/interventions.py`
  - matched random features
  - atom-feature ablation
  - bundle-feature ablation
  - damping and optional steering

The existing CLI scripts should remain as thin orchestration entry points. Keep
them runnable with `python -m ... --help`.

## Run Recovery

The previously failed utility-control run failed because the content-anchor
split was not a valid train/validation/test split. One observed run had
thousands of training examples, no validation examples, and only a few test
examples. That run is retired.

Recovery rule:

- Use deterministic pair-level splitting for SHP content-anchor pairs.
- Verify split counts before any utility-overlap claim.
- Write recovery outputs to a fresh artifact directory with a `resplit` or
  `recovery` suffix.
- Do not merge failed utility-control outputs with the recovered outputs.

Recovery command pattern on LRZ:

```bash
python -m aisafety.scripts.run_d4_atom_recovery \
  --manifest-json data/derived/d4_dataset_pack_v1/manifest.json \
  --reward-run-dir artifacts/reward/j0_anchor_v1_h100compact \
  --content-anchor-only \
  --content-max-pairs 4000 \
  --out-dir artifacts/mechanistic/d4_j0_atom_recovery_v1_resplit
```

Required checks:

- `content_anchor_utility_summary.json` exists
- `content_anchor_split_counts` has nonzero `train`, `val`, and `test`
- `content_anchor_utility_metrics.csv` has `status=ok` for at least one layer
- no paper claim says utility independence until these checks pass

## Pipeline Stages

### Stage 0: Manifest And Ontology Audit

Goal:

- confirm which atoms are actually present in the D4 pack
- confirm which bundles those atoms support
- decide whether safety/compliance needs a D4 rebuild before the SAE run

Inputs:

- `configs/ontology/d4_reduced_ontology_v1.json`
- `data/derived/d4_dataset_pack_v1/manifest.json`
- `data/derived/d4_dataset_pack_v1/summary.json`

Outputs:

- run-local `manifest_audit.json`
- list of traced atoms
- list of missing current-ontology atoms
- decision: use current D4 pack or rebuild

Rebuild trigger:

- any main paper claim depends on a safety/compliance atom absent from the D4
  manifest
- current D4 labels have degenerate prevalence in discovery or Laurito slices
- the D4 pack cannot reproduce path or split integrity checks

### Stage 1: Residual Atom Recovery

Goal:

- identify which atoms are linearly recoverable from pooled residual states
- choose layer bands for SAE analysis
- recover the failed utility-control result with a valid split

Current implementation:

- `src/aisafety/scripts/run_d4_atom_recovery.py`

Run order:

1. content-anchor-only recovery
2. full residual atom recovery if needed
3. summary audit against previous best-layer results

Outputs:

- `atom_probe_metrics.csv`
- `best_layers_by_atom.csv`
- `laurito_transfer_metrics.csv`
- `content_anchor_utility_metrics.csv`
- `summary.json`

Interpretation:

- High residual recovery means the judge represents information sufficient to
  recover an atom at that layer and pooling policy.
- It does not mean the feature is sparse, interpretable, causal, or surface-only.

### Stage 2: SAE Feature Localization

Goal:

- rank sparse features that align with atom labels
- measure whether those features transfer to Laurito text-side scores
- measure whether pairwise LLM-minus-human activation asymmetry aligns with
  judge decisions
- measure overlap with SHP chosen-vs-rejected content utility

Current implementation:

- `src/aisafety/scripts/run_d4_sae_feature_analysis.py`

Refactor requirements:

- write outputs after each layer
- include run metadata for model, adapter, SAE release, SAE id, aggregation,
  layers, D4 pack hash if available, and content split counts
- separate feature ranking from examples and bundle aggregation
- support scout and full modes

Scout mode:

- layers: late layer band plus one early and one middle control
- atom subset: known recoverable atoms plus one weak/negative control
- content pairs: small but valid
- purpose: validate memory, SAE availability, output schemas, and split counts

Full mode:

- layers: selected bands from residual recovery
- atoms: all D4 manifest atoms unless a rebuild has updated the manifest
- content pairs: full configured cap
- aggregation: max-pool first, last-token as a sensitivity run for composed
  atoms and list structure

Outputs:

- `sae_atom_feature_scores.csv`
- `sae_bundle_feature_scores.csv`
- `sae_laurito_decision_alignment.csv`
- `sae_content_utility_overlap.csv`
- `sae_feature_examples.json`
- `sae_feature_set_manifest.json`

Interpretation:

- A candidate atom feature needs atom separation, validation/test performance,
  Laurito transfer, and interpretable examples.
- A candidate judge-use feature also needs pair-decision alignment.
- A candidate surface-cue feature needs weak or controlled utility overlap plus
  falsification against close negatives.

### Stage 3: Feature Explanation And Falsification

Goal:

- convert ranked features into interpretable feature cards
- prevent top-example storytelling

For each candidate feature:

- list top D4 atom-probe activations
- list top Laurito activations
- list close negative texts with similar topic/source but lower atom score
- record tokens or spans associated with max activation when available
- record whether examples are atom-specific, bundle-general, domain-specific,
  or likely content/quality-linked

Required feature-card fields:

- `feature_id`: release, SAE id, layer, index, aggregation
- `primary_atom`
- `secondary_atoms`
- `bundle_candidates`
- `positive_examples`
- `close_negative_examples`
- `failure_modes`
- `laurito_text_transfer`
- `laurito_decision_alignment`
- `content_utility_overlap`
- `claim_status`: exploratory, candidate, or rejected

Claim gate:

- Do not promote a feature to `candidate` without at least one close-negative
  falsification check.

### Stage 4: Atom-To-Bundle Formation

Goal:

- explain how atoms compose into bundle-level representations or judge-facing
  cue packages

Build a feature graph:

- atom nodes: D4 atoms
- feature nodes: SAE features
- bundle nodes: ontology bundles
- decision nodes: Laurito LLM-choice and reward-margin measures
- guardrail nodes: SHP content-anchor utility measures

Edges:

- atom to feature: validation/test AUC and signed separation
- feature to Laurito atom score: Spearman
- feature to decision: LLM-choice AUC and reward-margin Spearman
- feature to utility: content AUC and activation means
- feature to bundle: membership if the feature supports multiple bundle atoms
  or a bundle-level readout

Bundle candidate criteria:

- at least two member atoms have feature support
- feature support is not entirely from one source domain
- Laurito transfer is present for at least one member atom
- judge-decision alignment is present for at least one feature in the bundle
- content-anchor overlap is reported, even if high

Bundle outputs:

- `bundle_feature_graph.csv`
- `bundle_candidate_sets.json`
- `bundle_feature_cards.json`
- `bundle_domain_stability.csv`

Interpretation:

- A bundle is not a latent object by definition. It is a supported grouping of
  atom features, ecological transfer, and decision alignment.
- If a single broad feature dominates a bundle, report that as a broad readout
  rather than atom-to-bundle formation.

### Stage 5: Cross-Layer Formation

Goal:

- distinguish token-local atom evidence from later composed bundle evidence

Analyses:

- early/mid/late feature localization by atom level
- feature matching across layers where SAE basis allows it
- persistence or transformation of atom features into broader bundle features
- comparison of max-pool and last-token aggregation

Use mechanistic permutability as a tool, not as a claim shortcut. Cross-layer
matching can suggest formation paths, but causal pathways still require
intervention.

Outputs:

- `feature_layer_trajectories.csv`
- `atom_bundle_layer_summary.csv`
- `cross_layer_matches.csv`

Interpretation:

- Early local features plus late bundle readouts support a formation hypothesis.
- Late-only recovery is acceptable for composed discourse cues.
- Lack of early recovery does not refute atom use if the atom is inherently
  compositional.

### Stage 6: Diagnostic Cross-Model Contrast

Goal:

- compare the stable J0 feature set to repair and leave-one-out contrasts
  without letting repair define the ontology

Run only after stable J0 candidates exist:

- `Jrepair-all`
- `jrepair_loo_cue_template_boilerplate_v1`
- `jrepair_loo_cue_hedging_certainty_v1`
- `jrepair_loo_joint_academic_formality_v1` as a positive-control readout

Outputs:

- per-model SAE feature scores
- feature-set overlap summary
- bundle movement summary
- diagnostic contrast report

Interpretation:

- Repair movement can prioritize intervention tests.
- Repair movement does not define atoms or prove mechanisms.
- Do not expand to a full adapter grid unless a specific causal hypothesis
  requires it.

### Stage 7: Causal Intervention

Goal:

- test whether candidate atom or bundle features causally affect judge choices

Order:

1. matched random feature ablation
2. top atom-feature ablation
3. bundle-level grouped feature ablation
4. damping sensitivity sweep
5. optional steering comparison
6. mechanism-informed repair redesign

Default intervention:

```text
z'_f = alpha * z_f, 0 < alpha < 1
```

Hard zeroing:

```text
z'_f = 0
```

Hard zeroing is an ablation test, not the default repair design.

Causal claim gate:

- The intervention must change biased Laurito decisions more than matched
  random features.
- Utility benchmarks and SHP content-anchor behavior must be reported.
- Side effects must be described, not hidden.

## Claim Ladder

Allowed after residual recovery:

- atom information is recoverable from residual states at layer/pooling sites

Allowed after SAE localization:

- sparse features align with atom labels
- sparse features transfer to Laurito atom scores
- sparse features align with pair-side judge decisions
- sparse features show measured utility overlap

Allowed after feature cards and falsification:

- candidate features plausibly represent specified surface or discourse cues

Allowed after bundle graph:

- atom features support a bundle-level formation hypothesis

Allowed after intervention:

- candidate features or feature groups causally affect a measured judge behavior

Not allowed before intervention:

- the feature drives the judge
- the bundle is the circuit
- damping repairs the judge
- low SHP overlap proves superficiality

## Implementation Checklist

Immediate:

- run manifest/ontology audit
- rerun fixed content-anchor utility control
- run SAE scout mode on J0
- verify layer-by-layer resumability
- inspect top feature cards for the strongest atoms

Near term:

- refactor D4 IO, labels, activations, SAE loading, ranking, examples, and
  bundles into `src/aisafety/mech/`
- add tests for deterministic split counts, SAE id formatting, bundle graph
  construction, and run-manifest schema
- add LRZ job wrappers for atom recovery and SAE feature analysis
- add close-negative feature-card generation

After stable J0 feature candidates:

- run cross-model diagnostic contrasts
- build bundle feature graph
- run first matched random-feature ablation

Deferred:

- new repair objective
- optional additional leave-one-out training
- exhaustive adapter grid
- causal language in the paper

## LRZ Execution Policy

All GPU, reward-model, dataset-pack, SAE, and intervention runs are LRZ runs.
Local execution is limited to editing and lightweight unit tests.

Use:

- `WORKDIR` for the LRZ project checkout
- `HF_HOME` for the LRZ Hugging Face cache
- `CACHE_DIR` only when a script expects it explicitly
- Slurm output and error logs stored with the job or copied into the artifact
  directory

Every LRZ run must record:

- command
- git commit or archive hash if available
- D4 manifest path
- reward run directory
- model id from config
- adapter path
- SAE release and id template
- selected layers
- aggregation policy
- output directory
- split counts
- skipped or missing layers

## Paper Alignment

The paper should frame the contribution as:

1. literature-grounded cue atom ontology
2. empirical bundle validation outside Laurito
3. ecological Laurito judge-choice alignment
4. residual and SAE localization
5. atom-to-bundle formation hypotheses
6. later causal intervention tests

Do not frame the current story as "we repaired judge bias." The current repair
adapters are diagnostic perturbations.

## End State For This Refactor

The refactor is complete when the repo can answer these questions from
artifacts alone:

- Which D4 atoms were present in the run?
- Which atoms were recoverable in residual states?
- Which sparse features aligned with each atom?
- Which features transferred to Laurito text-side atom scores?
- Which features aligned with pair-side judge decisions?
- Which features overlapped with SHP content utility?
- Which atom features formed bundle candidates?
- Which bundle candidates survived feature-card falsification?
- Which causal intervention, if any, changed judge behavior relative to matched
  controls?

