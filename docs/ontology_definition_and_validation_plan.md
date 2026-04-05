# Ontology Definition and Validation Plan

## Purpose

This document defines:

1. how the full candidate ontology should be constructed
2. how cue atoms should be grouped into higher-level bundles
3. how those bundles should be statistically validated
4. how the validated ontology should then be used in target judgment samples such as Laurito

The key shift is methodological:

- the ontology should not be treated as a hand-authored list of style labels
- it should be treated as a literature-seeded candidate inventory whose structure is tested empirically

## Principle

The project should avoid claiming that the ontology is `proven` in a strict sense.

The stronger and more defensible claim is:

`The ontology is supported by convergent statistical evidence across discovery corpora, held-out corpora, and ecological target tasks.`

That is the standard to aim for.

## Research basis

The ontology should draw from four established research traditions.

### 1. Register variation and multidimensional analysis

This tradition is the strongest basis for defining lexicogrammatical and informational style units as co-occurring dimensions rather than isolated markers.

Relevant sources:

- Biber and Egbert feature inventory:
  https://www.cambridge.org/core/books/register-variation-online/appendix-a-linguistic-features-included-in-the-multidimensional-analysis/F9F9B4FE80ED9B9A3927AD589C31E815
- Overview of multidimensional analysis as co-occurrence-based register analysis:
  https://www.uni-bamberg.de/fileadmin/eng-ling/fs/Chapter_21/21Summary.html

### 2. Metadiscourse, stance, and engagement

This tradition is the strongest basis for discourse-signaling, reader-guidance, stance, and interactional atoms.

Relevant sources:

- Systematic review of metadiscourse in academic writing:
  https://www.sciencedirect.com/science/article/pii/S0024384123000852
- Metadiscourse across communicative contexts:
  https://www.sciencedirect.com/science/article/pii/S037821661830105X
- Multidimensional metadiscourse analysis across written registers:
  https://journals.sagepub.com/doi/abs/10.1177/1461445615623907
- Multidimensional metadiscourse analysis across spoken registers:
  https://www.sciencedirect.com/science/article/pii/S037821661730111X

### 3. Authorship and stylometry

This tradition is the strongest basis for syntax, lexical selection, and discourse coherence as robust cross-domain style markers.

Relevant sources:

- What represents style in authorship attribution:
  https://aclanthology.org/C18-1238/
- Local discourse coherence for authorship attribution:
  https://academic.oup.com/dsh/article/29/2/191/974196
- Computational sociolinguistics survey:
  https://direct.mit.edu/coli/article/42/3/537/1536/Computational-Sociolinguistics-A-Survey

### 4. Human-vs-LLM text differences

This tradition is the strongest basis for deciding which style and register dimensions are likely to matter in human-vs-LLM settings.

Relevant sources:

- Survey on LLM-generated text detection:
  https://direct.mit.edu/coli/article/51/1/275/127462/A-Survey-on-LLM-Generated-Text-Detection-Necessity
- Human-AI parallel corpus evidence:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11874169/
- Laurito ecological judge benchmark:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12337326/

## What the ontology should contain

The ontology should be built from `atoms`, not from bundles alone.

Each candidate atom must satisfy all of the following:

1. it has a clear linguistic interpretation
2. it can be extracted or weak-labeled reproducibly from text
3. it is plausibly content-light enough to function as a style/register cue
4. it is meaningful across at least one ecological judgment setting
5. it can be mapped to one or more bundles

## Candidate atom inventory plan

The full candidate ontology should be built in six passes.

## Pass 1: Literature-seeded inventory

Create the first inventory by merging atom candidates from the four research traditions above.

The output of this step is not the final ontology.

It is a large candidate pool.

### Required atom families

#### A. Lexicogrammatical register atoms

Seed from multidimensional analysis and complexity literature.

Examples:

- pronoun rate
- determiner density
- noun density
- abstract noun density
- nominalization rate
- passive rate
- preposition density
- adjective density
- clause subordination
- phrasal elaboration
- sentence length distribution

#### B. Metadiscourse atoms

Seed from Hyland-style and reflexive metadiscourse work.

Examples:

- transitions and connectives
- frame markers
- code glosses
- evidentials
- self-mentions
- hedges
- boosters
- engagement markers
- directives
- questions

#### C. Discourse and rhetorical atoms

Seed from discourse coherence and genre analysis.

Examples:

- entity-grid coherence
- local entity transition patterns
- rhetorical move indicators
- summary or takeaway moves
- problem-solution moves
- feature-benefit moves
- abstract background-method-result moves
- setup-conflict-resolution moves

#### D. Stance and appraisal atoms

Seed from stance and evaluation work.

Examples:

- epistemic hedging
- certainty
- affect
- evaluation
- professionalism
- impersonality
- enthusiasm
- compliance stance

#### E. Formatting and packaging atoms

Seed from stylometry and LLM detection literature.

Examples:

- bullet or list structure
- enumeration markers
- heading-like openings
- regular sectioning
- punctuation density
- answer wrapper phrases
- balanced multi-part completion

### Output

A candidate atom table where each row includes:

- atom name
- linguistic level
- theoretical source
- extraction method
- expected relation to domains
- expected relation to human-vs-LLM distinction
- likely bundle memberships

## Pass 2: Operationalization

Every atom must be linked to an extraction or labeling method.

### Allowed extractor types

- lexicon-based
- POS-tag pattern
- dependency pattern
- constituency pattern
- sentence or paragraph statistics
- discourse heuristic
- entity-grid or coherence model
- weak classifier

### Rule

No atom should enter the validated ontology without an explicit extraction recipe.

That recipe must specify:

- units
- normalization
- text length sensitivity
- known confounds

### Example

`nominalization_patterns` should not just be a label.

It should specify:

- suffix or lexicon heuristics
- POS/dependency context
- normalization per 1k tokens
- minimum length for reliability

## Pass 3: Content leakage screening

Some candidate atoms will carry too much topic content.

Before bundle construction, remove or demote atoms that are too content-heavy.

### Screening criteria

Reject or demote atoms if they:

- depend mainly on domain-specific nouns
- fail cross-domain transfer badly
- mostly reflect topical entities rather than register or rhetoric

This follows authorship-attribution findings that proper nouns and topical nouns often interfere with style measurement.

Source:

- https://aclanthology.org/C18-1238/

## Pass 4: Theory-seeded bundle mapping

Only after the atom inventory is defined should bundles be specified.

Bundles should be many-to-many mappings over atoms.

### Example bundle construction

`academic_formality` should be modeled as a proposed bundle over:

- formal connectives
- metadiscourse
- abstract noun density
- nominalization
- passive or agentless constructions
- pronoun suppression
- background-method-result moves
- epistemic hedging
- institutional impersonality

`template_packaging` should be modeled as a proposed bundle over:

- enumeration markers
- frame markers
- summary moves
- balanced multi-part completion
- assistant wrapper phrases
- regular sentence balance

At this stage, bundle membership is still a hypothesis.

## Pass 5: Statistical discovery and pruning

This is where the candidate ontology becomes a validated ontology.

The main discovery corpus should be much larger than Laurito and should include:

- HC3
- H-LLMC2
- abstract-heavy corpora
- Laurito ecological domains
- any added real-world judge settings such as job applications

Laurito alone is too small to carry this step.

### Statistical objectives

There are four separate things to validate.

#### Objective A: Atom quality

Check:

- prevalence
- extraction stability
- length sensitivity
- cross-corpus consistency

Recommended tests:

- bootstrap confidence intervals for mean prevalence
- test-retest or resampling stability
- correlation with length and domain indicators

Atoms that are too rare, too unstable, or too length-driven should be dropped or marked exploratory.

#### Objective B: Co-occurrence structure

Check whether atoms selectively co-occur in ways consistent with bundles.

Recommended analyses:

- pairwise correlation or mutual information
- exploratory factor analysis
- hierarchical clustering or community detection
- sparse network estimation if dimensionality is high

Why:

- multidimensional analysis explicitly defines register structure through co-occurrence
- bundle validity should therefore be tested as a co-occurrence problem

Recommended interpretation standard:

- a bundle is stronger if several of its atoms co-load or cluster together stably across resamples

#### Objective C: Stability across domains and generators

Check whether the bundle structure survives:

- across domains
- across generators
- across train and held-out splits

Recommended analyses:

- multi-group exploratory analysis
- confirmatory factor analysis or equivalent held-out verification
- bootstrap stability of cluster membership

A bundle should not be considered stable if it appears only in one generator family unless it is explicitly intended as domain-specific.

#### Objective D: Source association

Check whether atoms and bundles actually distinguish human from LLM text.

Recommended analyses:

- mixed-effects logistic regression for human vs LLM label
- random intercepts for prompt, domain, and generator when available
- ablations comparing atom-only, bundle-only, and combined predictors

This step shows ecological relevance beyond pure linguistic co-occurrence.

## Pass 6: Ecological validation in target judgment samples

Only after bundle structure is estimated on larger corpora should Laurito be used as a confirmatory target.

This is the critical design decision.

Laurito should validate ecological importance, not define bundle structure from scratch.

## Statistical verification in target samples

The target question is:

`Do the validated atoms and bundles predict judge decisions in ecologically realistic settings?`

### Target samples

- Laurito paper abstracts
- Laurito product descriptions
- Laurito movie synopses
- optional extension domains such as job applications

### Main predictor form

For pairwise judgments, use atom and bundle differences between option A and option B.

That means:

- atom score difference: `atom(A) - atom(B)`
- bundle score difference: `bundle(A) - bundle(B)`

The judged choice becomes the outcome.

### Recommended models

#### Model 1: Atom-level ecological association

Use mixed-effects logistic regression:

- outcome: judge chooses B or LLM side
- predictors: atom score differences
- controls: length difference, domain, optional readability proxies
- random intercepts: item or prompt family

Purpose:

- identify which low-level cues are associated with judgments

#### Model 2: Bundle-level ecological association

Use mixed-effects logistic or Bayesian hierarchical regression:

- outcome: judged choice
- predictors: validated bundle scores
- domain interactions

Purpose:

- identify which higher-level bundles matter by domain

#### Model 3: Atom-to-bundle mediation

Where sample size allows:

- test whether atom effects are partially mediated by bundle scores

Purpose:

- show that bundles are not arbitrary labels but aggregate lower-level signals

## How to handle Laurito's small sample size

Laurito is too small for ontology discovery.

It is still useful for:

- confirmatory ecological testing
- domain interaction estimates with partial pooling
- bootstrap uncertainty estimates

### Recommended small-sample strategy

1. fit ontology structure on larger discovery corpora
2. freeze atom and bundle definitions
3. evaluate only a small number of pre-registered bundle hypotheses on Laurito
4. use:
   - stratified bootstrap
   - exact tests where applicable
   - hierarchical Bayesian partial pooling across domains

This lets Laurito act as a realistic but not overloaded target set.

## What counts as bundle validation

A bundle should only be called `validated` if it satisfies all of:

1. theoretical interpretability
2. explicit atom extraction recipes
3. stable co-occurrence structure in larger corpora
4. some cross-domain or cross-generator stability, unless intentionally domain-specific
5. predictive relevance for human-vs-LLM source or judge decision

If one of these fails, the bundle should be marked:

- exploratory
- domain-specific only
- or unsupported

## Minimum viable validation program

If time is limited, the smallest defensible plan is:

1. seed the atom inventory from the literature
2. operationalize the atoms with reproducible extractors
3. fit co-occurrence structure on the large discovery corpus
4. validate two bundles strongly:
   - `academic_formality`
   - `template_packaging`
5. test those two bundles in Laurito with partial pooling

That is enough to support the first mechanistic phase.

## Stronger full validation program

For a stronger paper:

1. validate atom quality
2. validate bundle co-occurrence
3. validate cross-domain and cross-generator stability
4. validate ecological relevance on Laurito
5. validate generalization on one extra realistic decision family, such as job applications

## Practical deliverables

The ontology-definition project should produce:

### D1

A candidate atom table with:

- theoretical source
- extractor type
- content-leakage risk
- likely bundle memberships

### D2

A validated bundle table with:

- member atoms
- co-occurrence statistics
- stability statistics
- domain relevance

### D3

A target-sample ecological validation table with:

- atom-level effects
- bundle-level effects
- domain interactions

### D4

A reduced, mechanistically useful ontology for SAE analysis:

- only atoms and bundles with both linguistic validity and ecological relevance

## How this connects to mechanistic interpretation

Only after D1 to D4 exist should the project ask:

- where are these atoms represented in Gemma
- where do they merge into bundle neighborhoods
- which features interface with the verdict

That sequence is necessary so the mechanistic target is not fabricated.

## Repo grounding

This plan should be used alongside:

- [linguistic_ontology_and_project_charter.md](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/docs/linguistic_ontology_and_project_charter.md)
- [linguistic_cue_ontology_v2.json](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/data/derived/style_groups/linguistic_cue_ontology_v2.json)

It is the next conceptual layer needed before the mechanistic experiments are specified in full detail.
