# Extension-Oriented First Experiment

## Why this is not a replication

The previous narrow setup was the smallest executable path with the existing scripts.

That is not the right framing for the project you described.

The actual first publishable experiment should be:

`discover a broad ontology of authorship-correlated surface cues, train a judge to become invariant to them, and test whether this reduces Laurito-style bias while preserving broader reward-model behavior`

Laurito remains an ecological target, not the whole project.

## Core claim

The first paper should aim to support this claim:

`LLM judges rely on a structured family of authorship-correlated surface cues; combining counterfactual invariance with cue-adversarial training reduces this reliance and improves robustness beyond the original Laurito domains`

That is already a real extension. It is not just a rerun with one axis removed.

## Cue ontology

Start with the ontology in `data/derived/style_groups/cue_ontology_v1.json`.

The ontology has three tiers.

### Tier 1: paired counterfactual families

These are the easiest families to train invariance on directly because public rewrite corpora exist.

- `academic_formality`
- `fluency_polish`
- `safety_corporate_tone`
- `paraphrase_surface`

### Tier 2: weak-label families

These are cue families that matter, but do not need expensive prompt generation in the first pass.

- `verbosity_compression`
- `hedging_certainty`
- `promotional_sales_tone`
- `narrative_packaging`
- `template_boilerplate`

These should be weak-labeled automatically on real human and LLM texts.

### Tier 3: residual discovery

After the first model is trained, cluster the remaining bias failures and define new cue families only if they explain held-out errors.

This keeps the ontology open without turning the project into an endless taxonomy exercise.

## Data foundation

Use three data layers instead of a single Laurito-style benchmark.

### Layer A: anchor preference and reward data

This is for preserving broad judgment competence while the model is pushed toward invariance.

Use:

- `SHP` for stable pairwise preference anchoring
- `HelpSteer2` or `HelpSteer2-Preference` for broader helpfulness and attribute coverage

Reason:

- `SHP` alone is too narrow
- `HelpSteer2` broadens the anchor without requiring any manual data creation

### Layer B: cue discovery corpus

This is the main corpus for finding the relevant surface-cue families.

Pool together:

- existing `paper`, `product`, and `movie` human vs LLM corpora already in the project
- `HC3` as a larger general human-vs-LLM corpus
- Laurito examples

Target size for the first pass:

- paper: `1000` human + `1000` LLM texts
- product: `1000` human + `1000` LLM texts
- movie: `1000` human + `1000` LLM texts
- HC3 general: `5000` human + `5000` LLM texts

This is large enough to support cue mining and small enough to build quickly.

### Layer C: counterfactual invariance corpora

Use public rewrite corpora for the paired invariance families:

- formality transfer
- fluency correction
- detox or tone neutralization
- paraphrastic variation

This keeps training mostly off-the-shelf and low-cost.

### Small generated set

Use generation only where it buys something unique.

Budget:

- `150` seeds per Laurito domain
- `2` rewrites per seed
- one domain-specific cue family per domain

That is `900` generated rewrites total.

Use them for:

- held-out evaluation
- weak-label calibration

Do not depend on them as the main training source.

## Automatic cue acquisition

The ontology should not be built by hand at scale.

Use this pipeline:

1. Train a strong authorship detector on Layer B.
2. Score every text with a structured feature bank.
3. Rank features by contribution to authorship prediction.
4. Cluster high-contribution features into cue families.
5. Convert families into weak labels.

The structured feature bank should include:

- length and compression features
- sentence and clause statistics
- lexical diversity and repetition
- punctuation patterns
- hedging and certainty markers
- passive voice and nominalization proxies
- discourse markers and template phrases
- promotional and safety lexicons
- hidden-state features from a frozen encoder

This is important: the ontology starts theory-backed, but the assignment of examples to cue families is automated.

## Training setup

The first serious training objective should no longer be just:

- pairwise preference loss
- plus score MSE on rewrites

Instead, use one shared encoder with:

- a scalar utility head
- cue-family classification heads
- optional uncertainty head

Loss:

```text
L = L_anchor
  + lambda_cf * L_counterfactual_invariance
  + lambda_adv * L_adversarial_cue_removal
  + lambda_group * L_group_robustness
  + lambda_wl * L_weak_label_supervision
```

Where:

- `L_anchor`
  - pairwise preference loss on SHP
  - plus broad reward supervision from HelpSteer2-style data
- `L_counterfactual_invariance`
  - matched-score loss on paired rewrite families
- `L_adversarial_cue_removal`
  - gradient-reversal loss so the shared representation cannot easily encode cue-family signal
- `L_group_robustness`
  - worst-group or balanced loss across domains and cue families
- `L_weak_label_supervision`
  - trains auxiliary cue heads using the automatically induced cue labels

This is the first meaningful extension beyond the current repo.

## Concrete model matrix

Run four models, not just baseline vs invariance.

### M0: anchor-only reward model

Use SHP plus the broader anchor data.

Purpose:

- establish base competence

### M1: paired-counterfactual invariance only

Add only Tier 1 paired families.

Purpose:

- estimate the effect of direct rewrite invariance alone

### M2: paired invariance plus weak-label cue removal

Add Tier 1 plus Tier 2.

Purpose:

- test whether broader cue removal is needed beyond paired rewrites

### M3: M2 without academic formality

Purpose:

- isolate whether academic formality explains the paper-domain effect

This is a much stronger design than a plain replication because it separates:

- generic invariance
- broader cue suppression
- the specific academic-formality hypothesis

## Minimal implementation path under time constraints

Do not try to build the full source-grounded judge yet.

The time-budget-aware version is:

1. keep the current reward-model backbone
2. add auxiliary cue heads
3. add weak-label training and adversarial removal
4. use public rewrite corpora for paired invariance
5. keep generation limited to held-out domain-specific evaluation

That is still substantial new work, but it is feasible.

## Evaluation

Laurito is only one slice.

The first paper-quality evaluation battery should include:

- SHP preference retention
- RewardBench-style sanity checks
- style sensitivity on paired counterfactuals
- held-out weak-label cue families
- Laurito bias by domain
- held-out generated domain-specific suites
- authorship leakage probes on the learned representation

Key measurements:

- bias reduction
- retained reward-model competence
- reduced cue predictability from hidden states
- transfer to unseen domain-specific nuisance families

## Publishable outcome

The paper is not:

`we reduced bias on 451 Laurito pairs`

The paper is:

`we built a broader nuisance-cue ontology, trained an invariant judge using paired and weakly supervised cue signals, and showed reduced authorship-linked bias with better generalization across domains and held-out cue families`

## Immediate build order

The next implementation sequence should be:

1. build the cue discovery corpus
2. compute automatic cue features and weak labels
3. extend the reward model with cue heads and adversarial loss
4. run `M0`, `M1`, `M2`, `M3`
5. only then spend generation budget on the held-out domain-specific evaluation suite

That gives you a real extension while staying inside a realistic time budget.
