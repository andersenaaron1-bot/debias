# Mechanistic Interpretability Research Plan for Judge Debiasing

## Status

This is the main project plan going forward.

It supersedes the older habit of describing the project mainly through internal run IDs such as `M0`, `M1`, `M2`, and `M3`.

Those run IDs should still be kept for experiment bookkeeping and cluster outputs, but they should no longer define the paper narrative.

The paper-facing structure should be:

1. behavioral failure and repair
2. cue-family discovery
3. internal feature localization
4. causal intervention
5. adapter interpretation

## Core thesis

The central claim of the project is:

`LLM judges partly rely on internal representations of authorship-correlated surface cues rather than grounded utility. These cue representations can be localized inside Gemma, organized into interpretable feature families, and causally repaired through targeted intervention and reward-adapter finetuning.`

This is more precise than:

- "judges are biased toward LLM outputs"
- "style matters"
- "the adapter removes stylistic artifacts"

and it is more aligned with mechanistic interpretability.

## Paper framing

The paper should not be framed as:

- a broad judge-debiasing paper
- a pure reward-model finetuning paper
- a generic style-invariance paper

It should be framed as:

`Interpreting and Repairing Authorship-Correlated Surface Cue Reliance in LLM Judges`

The motivating case remains Laurito-style human-vs-LLM comparisons, but the underlying scientific object is:

- the internal representation of cue families
- the pathway from cue representation to pairwise judgment
- the effect of repair on that pathway

## Why the old `M*` framing is not enough

The earlier `M0` to `M3` framing was useful for getting the first executable line running:

- anchor-only model
- rewrite-invariance model
- broader repair model
- no-academic-formality ablation

That is still valuable as an internal baseline ladder.

But it is too narrow for the full project because it encourages a story about:

- adding and removing losses
- comparing aggregate metrics
- treating academic formality as the main object

The stronger story is about:

- multiple cue families
- shared versus domain-specific internal structure
- causal intervention on the discovered mechanism

## Role of academic formality

Academic formality should remain the flagship ablation, not the only one.

It is the best positive-control cue family because:

- it already showed an effect in your prior experiments
- it is domain-relevant for paper abstracts
- it is likely to be recoverable as a coherent internal feature family

But one cue family is not enough for a strong result.

The project should therefore use:

- `academic formality` as the central domain-specific ablation
- at least one `cross-domain cue family` such as template packaging
- at least one `domain-specific non-academic family` such as promotional tone or narrative packaging
- at least one `negative or random control` ablation

That gives the project a real structure rather than a single-axis story.

## Main hypotheses

### H1

Academic formality is a disproportionately important nuisance channel in paper-abstract comparisons.

### H2

Template packaging and instruction-tuned boilerplate are broader cross-domain nuisance channels that contribute to human-vs-LLM preference shifts in all three Laurito domains.

### H3

Promotional tone and narrative packaging are narrower domain-specific channels that matter mainly in product descriptions and movie synopses respectively.

### H4

These cue families are represented by partially localized internal feature groups rather than only a diffuse "AI authorship" signal.

### H5

The strongest judge bias does not come from a single authorship detector, but from a cue-to-verdict pipeline in which multiple proxy cues are aggregated into a quality judgment.

### H6

The broader repair adapter changes this behavior in one of two ways:

- late-stage wrapper correction
- deeper representational attenuation of cue-linked features

The project should explicitly distinguish those two possibilities.

## High-level research questions

### RQ1

Which cue families account for the strongest share of human-vs-LLM preference bias?

### RQ2

Where are these cue families represented inside Gemma 9B?

### RQ3

Are cue representations shared across domains, or do they branch into domain-specific variants?

### RQ4

Do cue families exhibit meaningful internal geometry or topology in feature space?

### RQ5

Which internal pathway maps cue-family evidence into final pairwise judgment?

### RQ6

What does reward-adapter repair actually alter inside the model?

## Project architecture

The project should proceed in five connected phases.

## Phase 1: Behavioral baseline and repair

Goal:

- establish that the failure mode exists
- establish that repair is possible
- create the model variants needed for mechanistic comparison

This phase is already largely implemented in the repo.

### Internal baseline ladder

Keep the old internal run IDs for execution:

- `M0`: anchor-only reward model
- `M1`: paired invariance only
- `M2`: paired invariance plus broader cue removal
- `M3`: `M2` without academic formality

But paper-facing names should be:

- `Base Judge`
- `Counterfactual Repair`
- `Broad Cue Repair`
- `No-Academic Repair`

### Required outputs

- Laurito bias by domain
- SHP preference retention
- HelpSteer2 anchor retention
- reward-benchmark retention
- held-out style sensitivity
- held-out generator transfer

### Purpose in the final paper

This phase is the repair baseline and the source of model checkpoints for later mechanistic comparison.

It is not the full paper.

## Phase 2: Cue-family program

Goal:

- define the cue families cleanly enough to support both mechanistic discovery and causal testing

The cue-family ontology should have three layers.

### Layer A: flagship cue families

These are the families around which the main paper is organized.

- `academic_formality`
- `template_packaging`
- `promotional_tone`
- `narrative_packaging`

These should be treated as the main named cue families in the paper.

### Layer B: support cue families

- `fluency_polish`
- `verbosity_compression`
- `hedging_certainty`
- `safety_corporate_tone`
- `paraphrase_surface`

These are important, but they are less likely to be the central paper figures.

### Layer C: residual discovered families

These are added only if:

- they are stable across runs
- they explain held-out errors
- they are interpretable enough to support causal tests

This prevents the project from turning into an endless taxonomy exercise.

### Data justification

The project should explicitly justify its hybrid data strategy as:

- curated counterfactual corpora for hypothesis-driven invariance
- automatic cue discovery for residual shortcut discovery

This follows the logic of counterfactual invariance and shortcut-learning work:

- https://proceedings.neurips.cc/paper/2021/hash/8710ef761bbb29a6f9d12e4ef8e4379c-Abstract.html
- https://aclanthology.org/2022.findings-naacl.130/
- https://aclanthology.org/2022.naacl-main.321/
- https://aclanthology.org/2023.findings-emnlp.10/

### Data priorities

Laurito remains:

- the motivating benchmark
- the ecological target
- the small, domain-specific audit set

It should not be the only discovery substrate.

The broader discovery substrate should prioritize:

1. more academic abstracts
2. more human-vs-LLM parallel data with multi-model coverage
3. more same-prompt multi-generator comparisons
4. only then broader movie and product expansions

The main reason is simple:

- the strongest prior signal is academic formality
- abstract discourse is likely the cleanest domain-specific cue family
- multi-generator coverage is needed so discovered features are not just ChatGPT or one-model signatures

Relevant context:

- Laurito et al.: https://pmc.ncbi.nlm.nih.gov/articles/PMC12337326/
- HAP-E: https://huggingface.co/datasets/browndw/human-ai-parallel-corpus
- M4: https://aclanthology.org/2024.eacl-long.83/
- M4GT-Bench: https://aclanthology.org/2024.acl-long.218/

## Phase 3: Mechanistic discovery

Goal:

- identify internal features that encode the cue families
- localize them by layer and submodule
- test whether they are cue-specific, domain-specific, or shared

### Primary microscope

Use Gemma Scope SAEs on Gemma 2 9B.

This remains the strongest reason to keep the project centered on Gemma.

Relevant sources:

- Gemma Scope model suite: https://huggingface.co/google/gemma-scope
- Gemma Scope paper: https://huggingface.co/papers/2408.05147
- Scaling Monosemanticity: https://transformer-circuits.pub/2024/scaling-monosemanticity/

### Main analyses

#### A. Cue-feature probes

Train sparse-feature probes that predict:

- cue-family labels
- human versus LLM authorship
- judgment preference shifts

The question is not just whether a feature predicts authorship.

The real question is:

`Does this feature mediate the shift from surface cue to verdict?`

#### B. Layer localization

Measure where cue-family signal is strongest:

- early layers
- middle layers
- late layers
- residual stream versus attention output versus MLP output

This is where your earlier longitudinal adapter observation becomes important.

If a small-data adapter mainly changes late layers, while a broader adapter alters cue-linked features earlier or more globally, that is already mechanistic evidence.

#### C. Cross-domain feature transfer

For each flagship family:

- train feature-level probes in one domain
- test in another domain

This separates:

- domain-general cue features
- domain-specific cue variants

#### D. Cross-generator feature transfer

Test whether the same internal feature families remain predictive across:

- ChatGPT-like outputs
- Llama/Qwen/DeepSeek-style outputs
- other available generators

#### E. Feature geometry

Treat this as a serious subproject, not just an optional visualization.

Questions:

- do cue-linked features cluster into local semantic neighborhoods
- do some cue families overlap strongly
- is academic formality embedded near nominalization, passive voice, or abstract boilerplate features
- do utility-linked and cue-linked features occupy separable or overlapping regions

Relevant background:

- Transcoders Find Interpretable LLM Feature Circuits: https://huggingface.co/papers/2406.11944
- Sparse Feature Circuits: https://huggingface.co/papers/2403.19647
- The Geometry of Concepts: https://huggingface.co/papers/2410.19750
- Mechanistic Permutability: https://openreview.net/forum?id=MDvecs7EvO

## Phase 4: Causal mechanism tests

Goal:

- move from correlation to mechanism

This phase is what makes the project feel genuinely mechanistic.

### Causal ladder

#### Level 1: random-control ablation

Ablate random matched features.

Purpose:

- establish a baseline for generic disruption

#### Level 2: cue-family feature ablation

Ablate top features linked to:

- academic formality
- template packaging
- promotional tone
- narrative packaging

Measure:

- Laurito bias
- counterfactual style sensitivity
- content ranking retention

#### Level 3: layer patching

Patch activations between:

- pretrained Gemma judge
- repaired adapter variants

Purpose:

- identify layers sufficient to restore or remove the old bias
- distinguish late wrapper repair from deeper representational repair

Relevant analogue:

- fine-tuning as wrappers: https://iclr.cc/virtual/2024/23794

#### Level 4: cue-to-verdict pathway tracing

Use attribution or patching from cue-linked features to the final score.

The target claim is:

`These feature families are not merely present; they are used on the path to the judgment.`

#### Level 5: targeted steering

Steer or damp the top cue-linked features and compare the effect to:

- random-feature steering
- generic layer steering
- reward-adapter repair

Relevant background:

- SAIF: https://huggingface.co/papers/2502.11356
- Improving Steering Vectors by Targeting SAE Features: https://huggingface.co/papers/2411.02193
- Caution on naive SAE decomposition: https://huggingface.co/papers/2411.08790

## Phase 5: Adapter interpretation

Goal:

- explain what the repair adapter actually learned

This phase should replace any simplistic "adapter acts more on early vs late layers" story.

### Main questions

1. Does the adapter reduce the activation of cue-linked features?
2. Does it preserve content-linked or utility-linked features?
3. Does it alter the downstream readout from cue-linked features rather than the feature activations themselves?
4. Is `Broad Cue Repair` a global rewrite while `No-Academic Repair` is a narrower late-stage correction?

### Suggested analysis

For the same prompt pairs, compare:

- pretrained Gemma judge
- Base Judge
- Counterfactual Repair
- Broad Cue Repair
- No-Academic Repair

in the same SAE feature basis.

This allows:

- feature activation comparisons
- family-level attenuation analysis
- attribution-path changes into the value head

## Main ablation suite

The core paper should have multiple ablations.

### A. Repair ablations

- `Base Judge`
- `Counterfactual Repair`
- `Broad Cue Repair`
- `No-Academic Repair`
- `No-Template Repair`
- `No-Promotional Repair`

At minimum, the last two can be lighter-weight if compute is tight.

### B. Mechanistic ablations

- top `academic_formality` feature ablation
- top `template_packaging` feature ablation
- top `promotional_tone` feature ablation
- top `narrative_packaging` feature ablation
- combined cue-family ablation
- random matched-feature ablation

### C. Layer ablations

- early-only intervention
- middle-only intervention
- late-only intervention
- all-layer intervention

### D. Transfer ablations

- seen versus unseen generators
- seen versus unseen domains
- seen versus held-out cue families

## Minimal ablation set under time constraints

If compute is constrained, the smallest strong version is:

1. `Base Judge`
2. `Broad Cue Repair`
3. `No-Academic Repair`
4. `No-Template Repair`
5. academic-formality feature ablation
6. template-packaging feature ablation
7. random matched-feature ablation

This keeps:

- one flagship domain-specific family
- one flagship cross-domain family
- one negative control

That is the minimum configuration I would trust for a workshop submission.

## Canonical evaluations

The project should retain four evaluation blocks.

### Block 1: behavioral repair

- Laurito bias by domain
- SHP preference retention
- HelpSteer2 utility and attribute retention
- reward-benchmark sanity checks

### Block 2: invariance

- paired counterfactual style sensitivity
- held-out cue-family sensitivity
- held-out generator transfer

### Block 3: mechanistic evidence

- cue-family feature localization
- layer localization
- cross-domain transfer of feature probes
- cross-generator transfer of feature probes
- feature geometry / neighborhood structure

### Block 4: causal evidence

- targeted feature ablation
- random-feature ablation controls
- patching between repaired and unrepaired judges
- targeted steering or damping

## Deliverables

The project should aim to produce these durable assets.

### D1

A cleaned and documented cue-family ontology with:

- flagship families
- support families
- residual family criteria

### D2

A broader cue-discovery corpus with:

- abstract-heavy expansion
- multi-generator coverage
- provenance tracking

### D3

A set of Gemma Scope feature maps for the main cue families.

### D4

A causal-ablation benchmark for cue-family features.

### D5

A repair-versus-mechanism comparison between pretrained, repaired, and ablated judges.

## Decision points

### Decision point 1

If academic formality is not mechanistically recoverable as a coherent feature family, do not keep it as the sole centerpiece.

In that case, the paper should pivot toward:

- template packaging as the main family
- or a broader "cue bundle" story

### Decision point 2

If the adapter mainly acts as a late wrapper, the paper should emphasize:

- repair via altered readout
- not deep representational rewriting

### Decision point 3

If cue-linked features can be localized but not causally intervened on cleanly, the paper remains viable as:

- mechanistic localization plus repair comparison

but should not oversell circuit-level understanding.

## Strongest expected paper story

The strongest likely final story is:

1. LLM judges use multiple authorship-correlated proxy cues.
2. These cues are partly encoded in interpretable Gemma features.
3. Academic formality is especially important in paper-abstract comparisons.
4. Template packaging is a broader cross-domain cue family.
5. Targeted ablation of these features reduces bias more selectively than random ablation.
6. The broader repair adapter works by attenuating or rerouting these cue-family features.

That story is strong enough for the mechanistic interpretability workshop and still useful even if the circuit-level evidence remains partial.

## Execution order

The recommended order is:

1. finish the first behavioral repair runs
2. freeze the best current checkpoints
3. run cue-feature localization for academic formality and template packaging
4. run cross-domain and cross-generator probe transfer
5. run academic-formality and template-packaging feature ablations
6. run random-feature controls
7. run patching between repaired and unrepaired judges
8. add promotional and narrative families if time remains

## Repo implication

The repo should now be understood as supporting one integrated project with two technical layers:

### Layer 1

behavioral repair and evaluation

### Layer 2

mechanistic discovery and causal intervention

The second layer should now become the main expansion target for new code and experiments.
