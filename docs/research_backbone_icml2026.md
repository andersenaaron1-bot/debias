# Research Backbone for ICML 2026 Mech Interp

## Purpose

This document is the working backbone for the project after the first end-to-end `M2`/`M3` reward-adapter loop.

It answers three questions:

1. Does the current experiment already fit the ICML 2026 Mechanistic Interpretability Workshop?
2. What focus shift is needed to make it a strong fit rather than only a partial fit?
3. What is the structured research plan from here?

## Workshop fit

The workshop call asks for work that uses internal states of neural networks to understand them, with explicit interest in:

- feature geometry and representations
- circuit analyses and causal methods
- interpretability for practical applications, benchmarking, safety, and model repair
- scaling mechanistic methods to realistic settings and larger models

Source:

- Mechanistic Interpretability Workshop at ICML 2026: https://mechinterpworkshop.com/
- Call for Papers: https://mechinterpworkshop.com/cfp/

## Assessment of current fit

### What already fits

The current project already fits the workshop in a narrow but real way if framed as:

- interpretability for practical applications
- interpretability for safety and model repair
- mechanistic hypotheses about authorship-correlated surface cues in LLM judges

That case is strongest because the project studies a concrete failure mode:

- LLM judges may rely on authorship-correlated surface cues rather than grounded utility

and because the current adapter work is already a form of model repair:

- identify a family of undesirable signals
- intervene on model behavior
- measure whether the intervention reduces the failure mode while preserving general capability

This aligns well with the workshop topics on practical applications, safety, and debugging undesirable model behavior.

### What does not yet fit strongly enough

As it stands, the core `M2`/`M3` experiment is still primarily a behavioral robustness and finetuning paper, not a strong mechanistic interpretability paper.

The main gaps are:

- the central evidence is output-side and benchmark-side, not internal-state evidence
- the current paper story does not yet show where the relevant cues live in Gemma
- it does not yet show whether those cues are localized features, distributed geometry, or small circuits
- the adapter is trained to suppress cue use, but the mechanism of that suppression is not yet characterized

So the current project is a plausible workshop fit under `practical applications` and `model repair`, but not yet under `feature geometry` or `circuit analyses`.

## Required focus shift

To become a strong workshop fit, the project should be reframed from:

`Can we train a judge to be less biased by stylistic artifacts?`

to:

`Which internal features and pathways cause LLM judges to use authorship-correlated surface cues, and can we causally repair that mechanism without destroying content judgment?`

That reframing keeps the current experiments, but changes their role.

### Role of the current adapter experiments

`M0`/`M1`/`M2`/`M3` should become the behavioral and repair baseline, not the headline result.

They answer:

- whether the failure mode is repairable at all
- whether academic formality is a special case
- what the performance tradeoff is

### Role of mechanistic analysis

The mechanistic contribution should answer:

- which internal features encode the cue families
- where in the model they are strongest
- whether those features are shared across domains or domain-specific
- whether suppressing them changes the judge because it directly targets those features, or only because of generic regularization

That is the part that makes the work feel native to a mechanistic interpretability venue.

## Central thesis

The project should now be organized around this thesis:

`AI ingroup bias in LLM judges is partly mediated by identifiable internal features that encode authorship-correlated surface cues such as academic formality, template packaging, and promotional tone. These features can be localized, causally intervened on, and partially repaired, reducing bias while preserving content-sensitive judgment.`

## Narrow paper claim

For the workshop, the right claim is narrower than a broad debiasing claim.

Recommended claim:

`We identify internal feature families that mediate authorship-correlated cue use in Gemma-based judges, show that academic-formality-like features are especially relevant in paper-abstract comparisons, and demonstrate causal repair through targeted interventions and reward-adapter finetuning.`

This is stronger and more workshop-aligned than:

- "we debias LM judges"
- "we remove style"
- "we train a better reward model"

## Research questions

### RQ1

Which authorship-correlated surface cue families explain the strongest share of bias in Laurito-style pairwise judgments?

### RQ2

Where are those cue families represented inside Gemma 9B, and how stable are they across domains and generators?

### RQ3

Are the relevant representations better described as:

- localized SAE features
- clusters or directions in feature geometry
- or broader distributed circuits across layers

### RQ4

Can causal intervention on those internal features reduce bias without materially degrading content-sensitive judgment?

### RQ5

What does the trained repair adapter actually change inside the model?

## Hypotheses

### H1

Academic formality is a domain-specific nuisance channel that contributes disproportionately to paper-domain bias.

### H2

Template packaging, fluency polish, and promotional or safety tone are broader cross-domain nuisance channels.

### H3

These cues are not purely diffuse; at least part of the signal is recoverable in interpretable SAE feature families on Gemma.

### H4

The `M2` adapter reduces bias partly by dampening or rerouting those cue-linked features rather than only by generic preference regularization.

## Data strategy

### Why the current hybrid approach is defensible

The project should explicitly justify the hybrid setup as:

- preset counterfactual datasets for known nuisance factors
- automatic cue discovery for residual shortcut discovery

This is defensible because curated perturbation sets are the cleanest way to test known invariances, while data-driven discovery is needed because hand-built cue lists are incomplete.

Relevant background:

- counterfactual invariance: https://proceedings.neurips.cc/paper/2021/hash/8710ef761bbb29a6f9d12e4ef8e4379c-Abstract.html
- automatic shortcut discovery at scale: https://aclanthology.org/2022.findings-naacl.130/
- importance of domain knowledge when defining spurious correlations: https://aclanthology.org/2022.naacl-main.321/
- limits and costs of counterfactual data creation: https://aclanthology.org/2023.findings-emnlp.10/

The correct framing is therefore:

- `preset style groups` define intervention targets
- `automatic cue discovery` expands coverage and finds residual cues

These are complementary, not contradictory.

### Role of Laurito

Laurito should remain:

- the motivating case study
- the ecological target benchmark
- the domain-specific residual audit

Laurito should not carry the full cue-discovery burden because the dataset is too small for that.

Source:

- Laurito et al. PNAS 2025: https://pmc.ncbi.nlm.nih.gov/articles/PMC12337326/

### Broader cue-discovery context

For a workshop or paper-ready version, cue discovery should use a wider context than Laurito alone.

### Highest-priority additions

- broader academic abstracts
- broader human-vs-LLM parallel corpora
- more generator diversity across the same prompts

This is important because multi-domain detection work shows that generator and domain shifts are hard, and detector-style shortcuts often fail to generalize.

Sources:

- M4: https://aclanthology.org/2024.eacl-long.83/
- M4GT-Bench: https://aclanthology.org/2024.acl-long.218/

### Strongest expected signal

The strongest likely signal is:

1. academic formality and abstract discourse for papers
2. template packaging and instruction-tuned boilerplate across all domains
3. promotional tone for products
4. narrative packaging for movie synopses

This is consistent with:

- the project's initial ablation signal on academic formality
- broader evidence that instruction-tuned LLMs differ from humans in grammatical and rhetorical style

Sources:

- Human-AI Parallel Corpus / HAP-E: https://huggingface.co/datasets/browndw/human-ai-parallel-corpus
- Reinhart et al. PNAS 2025: https://pmc.ncbi.nlm.nih.gov/articles/PMC11874169/

## Project structure

The project should now run on three connected tracks.

### Track A: Behavioral repair baseline

This is the current active training line in the repo.

Goal:

- establish that the failure mode is real
- establish that it is partially repairable
- estimate the tradeoff between debiasing and general reward competence

Primary models:

- `M0`: anchor-only reward model
- `M1`: paired invariance only
- `M2`: paired invariance plus weak-label cue removal
- `M3`: `M2` without academic formality

Primary outputs:

- Laurito bias shift
- style sensitivity
- preference retention
- anchor retention
- reward-benchmark retention

This track is necessary, but it is not sufficient as the final workshop story.

### Track B: Mechanistic discovery

This is the main extension required for a strong workshop fit.

Goal:

- find the internal features that correspond to cue families
- localize them in Gemma
- determine whether they are domain-specific or shared

Recommended method:

- use Gemma Scope SAEs on Gemma 2 9B as the internal microscope
- measure activation differences on matched human-vs-LLM and rewrite-controlled pairs
- identify SAE features whose activation predicts:
  - authorship
  - cue-family weak labels
  - judge preference shifts

Source:

- Gemma Scope: https://huggingface.co/google/gemma-scope
- Gemma Scope paper: https://huggingface.co/papers/2408.05147

### Concrete analyses

1. Feature localization

- which layers and sublayers carry the strongest cue-family signal
- compare residual, attention, and MLP SAE spaces

2. Feature geometry

- cluster cue-linked features
- test whether cue families form coherent local regions or overlap heavily with utility features

3. Cross-domain transfer

- train feature-level cue probes on one domain and test on another
- identify domain-specific versus shared cue features

4. Cross-generator transfer

- test whether the same feature family captures multiple LLM generators, not only one model family

### Track C: Causal intervention and repair interpretation

Goal:

- show that the internal features are not only correlated with the failure mode
- show that intervening on them changes behavior in the predicted direction

Recommended intervention ladder:

1. feature ablation
- zero or damp top cue-linked SAE features during judging

2. feature steering
- contrastive or targeted steering on the most credible cue-linked features

3. adapter interpretation
- compare pretrained versus `M2`/`M3` activations on the same feature set
- test whether the adapter reduces activation on cue-linked features or changes downstream readout from them

Relevant steering background:

- SAIF: https://huggingface.co/papers/2502.11356
- Improving Steering Vectors by Targeting Sparse Autoencoder Features: https://huggingface.co/papers/2411.02193
- Caution on naive SAE decomposition of steering vectors: https://huggingface.co/papers/2411.08790

## Evaluation plan

The canonical evaluation should remain:

- counterfactual invariance
- grounded or content-sensitive discrimination
- general capability retention
- internal-mechanism validation

### Behavioral checks

- Laurito bias by domain
- SHP preference retention
- HelpSteer2 anchor preservation
- RewardBench-style sanity checks
- held-out cue-family sensitivity
- held-out generator transfer

### Mechanistic checks

- fresh external cue probe on frozen activations
- layerwise localization of cue signal
- overlap between cue-linked features and utility-linked features
- change in discovered feature activations before versus after adapter repair

### Causal checks

- does ablating cue-linked features reduce bias
- does it preserve content ranking better than random-feature ablations
- does targeted intervention outperform untargeted generic steering

## Publication strategy

### Strong workshop version

The strongest workshop framing is:

`Interpreting and Repairing Authorship-Correlated Surface Cue Reliance in LLM Judges`

with the structure:

1. behavioral evidence of the failure mode
2. cue ontology and broader cue-discovery substrate
3. mechanistic localization with Gemma Scope features
4. causal intervention on discovered features
5. comparison to repair by reward-adapter finetuning

### Weaker but still viable workshop version

If the causal-intervention results are incomplete, the paper can still fit if it emphasizes:

- careful mechanistic localization
- rigorous negative or partial evidence
- the mismatch between behavioral repair and internal repair

The workshop explicitly welcomes negative results, replications, datasets, and practical benchmarking contributions.

## Repository backbone

The repo should now be treated as supporting two layers:

### Layer 1: active executable baseline

- reward-adapter training and evaluation
- cue discovery corpus and weak labels
- cluster execution path

### Layer 2: next mechanistic extension

- SAE feature extraction on Gemma
- feature-level probes and clustering
- causal ablations and steering
- interpretation of adapter-induced feature changes

## Minimum next steps

1. Finish the first `M2` and `M3` cluster runs and store full evaluation outputs.
2. Freeze the current behavioral baseline as the `repair` arm of the project.
3. Add a dedicated mechanistic analysis module centered on Gemma Scope features.
4. Expand cue discovery beyond Laurito with broader abstract-heavy and multi-generator corpora.
5. Run the first feature-localization study for:
   - academic formality
   - template packaging
   - promotional tone
6. Run causal ablations on the strongest recovered features.
7. Compare those interventions with the learned `M2` adapter.

## Decision rule

If the mechanistic feature analyses produce clean, causal evidence, the paper should be pitched as a mechanistic interpretability and model-repair paper.

If the internal evidence remains weak, the project is still viable, but it should be pitched to a robustness, evaluation, or safety venue instead of leaning too hard on mechanistic interpretability.
