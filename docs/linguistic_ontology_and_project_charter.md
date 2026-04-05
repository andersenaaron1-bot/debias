# Linguistic Ontology and Project Charter

## Purpose

This document replaces the earlier broad `style-family` framing with a layered linguistic ontology.

The project is now organized around:

1. `cue atoms`
2. `cue bundles`
3. `verdict pathways`

That change is necessary because broad labels such as `academic formality` are too coarse to be the mechanistic target by themselves.

The mechanistic target must instead be:

- linguistically grounded
- independently labelable from text
- compositional enough to support feature localization in Gemma

## Core charter

The project charter is now:

`Determine which low-level linguistic cues are encoded inside Gemma-based judges, how those cues compose into higher-level register and rhetorical bundles, and how those internal bundles influence human-vs-LLM decisions across ecological judgment settings.`

The project therefore no longer asks only:

- can we debias a judge?

It asks:

- which grounded cue atoms and bundles does the judge use
- how are they represented internally
- how do they affect the verdict
- how does repair alter those pathways

## Scientific object

The scientific object of the project is:

`cue_atom -> cue_bundle -> verdict pathway`

This is a more principled unit than:

- style family as a single latent
- global AI-authorship representation
- adapter-layer deltas alone

## Linguistic basis

The ontology is grounded in formal register, style, discourse, and authorship work.

The main conceptual basis is:

- register variation and multidimensional analysis
- computational sociolinguistics
- cross-domain authorship attribution
- discourse-level style analysis
- modern style-transfer decomposition

Key sources:

- Register variation / multidimensional analysis:
  https://www.cambridge.org/core/books/register-variation-online/appendix-a-linguistic-features-included-in-the-multidimensional-analysis/F9F9B4FE80ED9B9A3927AD589C31E815
- Computational sociolinguistics survey:
  https://direct.mit.edu/coli/article/42/3/537/1536/Computational-Sociolinguistics-A-Survey
- Style transfer survey:
  https://direct.mit.edu/coli/article/48/1/155/108845/Deep-Learning-for-Text-Style-Transfer-A-Survey
- What represents style in authorship attribution:
  https://aclanthology.org/C18-1238
- Discourse information for authorship analysis:
  https://academic.oup.com/dsh/article/29/2/191/974196
- Survey on LLM-generated text detection:
  https://direct.mit.edu/coli/article/51/1/275/127462/A-Survey-on-LLM-Generated-Text-Detection-Necessity

## Layered ontology

The ontology has seven levels.

### Level 1: Orthographic and formatting cues

Examples:

- enumeration markers
- bullet or list structure
- punctuation density
- quote style
- heading-like openers

Why it matters:

- these cues are easy for models to detect
- they often change perceived clarity and professionalism without changing content

### Level 2: Lexical selection

Examples:

- formal connectives
- metadiscourse markers
- promotional adjectives
- disclaimer lexicon
- hedge markers
- certainty markers
- reporting verbs
- abstract technical nouns

Why it matters:

- lexical choice is one of the easiest channels through which register and stance are signaled

### Level 3: Morphosyntactic constructions

Examples:

- passive or agentless constructions
- nominalization
- complex noun phrase chains
- subordination density
- participial modifiers
- personal pronoun suppression
- imperative or instructional constructions

Why it matters:

- these are stronger cross-topic style indicators than raw content words
- they are plausible candidates for stable SAE features

### Level 4: Information packaging

Examples:

- sentence length balance
- compression versus elaboration
- definition-then-expansion pattern
- benefit-first packaging
- balanced multi-part completion
- high-density summary style

Why it matters:

- this level captures how content is arranged, not just what words are present

### Level 5: Discourse and rhetorical moves

Examples:

- background-method-result script
- problem-solution script
- feature-benefit script
- setup-conflict-resolution script
- conclusion or takeaway move
- helpful-assistant wrapper

Why it matters:

- this is where local cues become genre scripts
- it is likely central for LLM judges because pairwise judgments are very sensitive to answer structure

### Level 6: Stance and pragmatics

Examples:

- epistemic hedging
- authoritative certainty
- institutional impersonality
- compliance or safety stance
- enthusiasm or salesmanship
- narrative engagement stance

Why it matters:

- stance strongly influences perceived trustworthiness, competence, and quality

### Level 7: Register and genre scripts

Examples:

- academic abstract register
- product pitch register
- movie synopsis register
- job-application professional register
- helpdesk assistant register

Why it matters:

- these are the high-level bundles that the judgment task sees
- they should be treated as composite objects built from lower-level atoms

## High-level bundles

These are the bundles that matter most for the project.

### Academic formality

Not a primitive.

It should be understood as a bundle over:

- formal connectives
- metadiscourse
- technical abstract nouns
- passive or agentless constructions
- nominalization
- complex noun phrases
- pronoun suppression
- background-method-result moves
- hedging
- institutional impersonality

Primary ecological domain:

- paper abstracts

Secondary domain:

- professional or job-application writing

### Template packaging

The best cross-domain bundle.

It is composed of:

- enumeration and list structure
- metadiscourse
- sentence balance
- balanced multi-part completion
- helpful-assistant wrapper
- explicit takeaway or summary moves

Primary ecological domains:

- paper
- product
- movie
- general judge settings
- job applications

### Promotional tone

Composed of:

- promotional adjectives
- certainty markers
- benefit-first packaging
- feature-benefit moves
- enthusiasm or salesmanship

Primary ecological domain:

- product descriptions

Secondary domain:

- job applications and professional self-presentation

### Narrative packaging

Composed of:

- compressed-but-smooth event packaging
- setup-conflict-resolution structure
- narrative engagement stance
- takeaway closure

Primary ecological domain:

- movie synopses

### Safety corporate tone

Composed of:

- disclaimer lexicon
- imperative guidance
- problem-solution or compliance structure
- institutional impersonality

Primary ecological domains:

- general judge tasks
- product
- professional or administrative writing

## Ecological judgment settings

The ontology should be anchored in real decision families, not only in abstract style bundles.

### Laurito paper abstracts

Main risk:

- the judge treats academic register as a proxy for scientific quality

Most relevant bundles:

- academic formality
- template packaging
- stance calibration

### Laurito product descriptions

Main risk:

- the judge treats polished marketing language as a proxy for usefulness or factual adequacy

Most relevant bundles:

- promotional tone
- template packaging
- safety corporate tone

### Laurito movie synopses

Main risk:

- the judge treats narrative smoothness or dramatic packaging as a proxy for content quality

Most relevant bundles:

- narrative packaging
- template packaging
- stance calibration

### Job applications and professional summaries

This is a natural extension domain.

It is useful because it sits between:

- academic register
- self-promotion
- template packaging
- stance management

Main risk:

- the judge treats polished professional register as a proxy for competence

This makes job-application style an excellent stress test for whether cue bundles generalize beyond Laurito.

### General LLM judge settings

Main risk:

- the judge treats assistant-like form, disclaimers, or polished helpfulness wrappers as a proxy for correctness

Most relevant bundles:

- template packaging
- safety corporate tone
- fluency polish
- stance calibration

## Mechanistic interpretation target

The mechanistic target should be layered in the same way as the ontology.

### Atom-selective features

These are sparse features selective for one low-level cue, for example:

- nominalization-like features
- hedge-marker features
- list-template features
- disclaimer-like features

These are the most grounded first targets.

### Bundle neighborhoods

These are clusters or matched features spanning several cue atoms that together instantiate a bundle such as:

- academic formality
- template packaging

These are the right objects for feature geometry analysis.

### Verdict-interface features

These are features whose activation predicts not only the presence of a cue, but its influence on the final preference score.

These are the right objects for causal ablation and patching.

## Training implications

The adapter should be trained against bundles, but the mechanistic analysis should localize atoms first.

That means:

- use bundle labels for weak supervision and repair objectives
- use atom labels for analysis and decomposition

This is a crucial distinction.

The repair model is allowed to operate at the bundle level.

The mechanistic analysis should not start there.

## Recommended training structure

### Alignment anchors

- SHP pairwise preference supervision
- HelpSteer2 pointwise utility and attribute supervision

Purpose:

- keep general judgment competence

### Counterfactual supervision

Use paired rewrites for bundles where clean interventions exist:

- academic formality
- paraphrase surface
- safety or compliance tone
- fluency polish

Purpose:

- make the score invariant under controlled bundle changes

### Weak-label cue supervision

Use the layered ontology to weak-label bundle membership on real human-vs-LLM corpora.

Purpose:

- broaden cue coverage beyond curated rewrite corpora

### Mechanistic follow-up

After training:

- localize atom features
- group them into bundle neighborhoods
- trace verdict-interface features

## Revised loss interpretation

The existing mixed loss remains useful, but it should now be interpreted through the layered ontology.

### Preference loss

Preserves the ability to rank outputs by content-sensitive utility.

### Anchor utility and attribute losses

Prevent collapse into a narrow anti-style classifier.

### Invariance loss

Acts at the bundle level.

It says:

- if two texts differ only by a bundle change that should be irrelevant, their score should stay similar

### Cue-adversarial loss

Acts at the bundle level too.

It says:

- the shared representation should not make it easy to linearly decode those bundles

The mechanistic claim should therefore not be:

- we directly removed atom features during training

It should be:

- we trained bundle-level repair and then analyzed how that affected atom-level and bundle-level internal representations

## Revised project hypotheses

### H1

High-level judge bias is mediated by a structured set of low-level cue atoms rather than a monolithic AI-authorship representation.

### H2

Those cue atoms compose into interpretable bundles such as academic formality and template packaging.

### H3

Different ecological domains place weight on different bundles.

### H4

Some bundles are cross-domain, especially template packaging.

### H5

Academic formality is a strong domain-specific bundle for paper abstracts, but not the sole organizing principle of the project.

### H6

The repair adapter changes either:

- bundle activations
- or the downstream readout from those bundles

and this can be distinguished mechanistically.

## Practical charter

Going forward, the project should be described as:

`A mechanistic study of how Gemma-based judges use low-level linguistic cues, rhetorical structure, and register scripts as quality proxies in human-vs-LLM evaluation tasks, and how those pathways can be repaired.`

This is a better charter than:

- style-invariance for judges
- academic-formality debiasing
- broad AI-authorship suppression

## Repo grounding

The machine-readable ontology is in:

- [linguistic_cue_ontology_v2.json](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/data/derived/style_groups/linguistic_cue_ontology_v2.json)

This document should now be treated as the main conceptual basis for:

- cue-family expansion
- weak-label design
- mechanistic feature localization
- causal ablation planning

The earlier style-family ontology can remain as a legacy intermediate artifact, but the layered linguistic ontology should become the new default frame.
