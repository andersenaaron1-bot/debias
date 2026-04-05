# Corpus Specification For Bundle Creation V1

## Purpose

This document defines the corpus specification that should support defensible atom-to-bundle discovery.

The main design rule is:

- use `broad domain-based source-labeled text` for bundle discovery
- use `paired or semantic-control corpora` for confirmation
- use `Laurito` only for ecological validation

This is necessary because the current D2 outputs in [bundle_validation.tsv](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/data/derived/style_groups/d2_validation_v1/bundle_validation.tsv) still leave `academic_formality`, `template_packaging`, `narrative_packaging`, and `safety_corporate_tone` as exploratory, which is consistent with a discovery base that is still too QA-heavy.

## Core principle

Paired prompt corpora are not the primary substrate for bundle discovery.

They are most useful for:

- semantic-control confirmation
- robustness checks
- falsifying bundle claims that only arise from topic or prompt leakage

The primary discovery substrate should instead be:

- large
- source-labeled
- multi-domain
- multi-generator
- excerpt-level

## Roles

### 1. Discovery core

Purpose:

- estimate atom prevalence
- estimate co-occurrence structure
- estimate source association

This layer should dominate the statistical ontology work.

### 2. Domain bolsters

Purpose:

- make sure the target domains are represented strongly enough for bundle formation
- especially for paper abstracts, product descriptions, and movie synopses

These are allowed to include both:

- source-labeled public corpora
- human-only corpora plus small LLM supplements

### 3. Controlled confirmation

Purpose:

- verify that the discovered bundles survive semantic control
- ensure that atom co-occurrence is not purely a topic artifact

This is where prompt-paired and rewrite-paired corpora belong.

### 4. Ecological validation

Purpose:

- test whether validated bundles predict actual judge choices

This is where Laurito belongs.

### 5. Held-out transfer

Purpose:

- test whether bundle structure and judge effects survive unseen generators and unseen domains

## Recommended dataset stack

### A. Discovery core

#### HC3

Role:

- required
- general human-vs-LLM source-labeled base

Why:

- prompt-paired
- widely used
- source labels are clean
- covers several QA and advice-like domains

Use:

- primary discovery corpus
- keep balanced by source and subset

Target:

- `10k` to `16k` text excerpts after balancing

Source:

- https://huggingface.co/datasets/Hello-SimpleAI/HC3

#### H-LLMC2

Role:

- required
- generator-diversity expansion

Why:

- extends HC3 prompts with several additional open-weight models
- reduces the risk that bundle formation reflects only ChatGPT-style artifacts

Use:

- primary discovery corpus
- cap model outputs per prompt group so one question does not dominate
- exclude reasoning-trace fields

Target:

- `8k` to `15k` text excerpts
- at least `5` distinct non-ChatGPT model families represented

Source:

- https://huggingface.co/datasets/noepsl/H-LLMC2

#### HAP-E or HAP-E mini

Role:

- required if storage and preprocessing permit
- otherwise strongly preferred

Why:

- multi-genre
- includes academic and television or movie-script material
- provides source-labeled human-vs-LLM text under closer stylistic continuation control than HC3

Use:

- primary discovery corpus
- stratify by text type
- cap per source chunk so repeated generations do not dominate

Target:

- `6k` to `12k` text excerpts

Source:

- https://huggingface.co/datasets/browndw/human-ai-parallel-corpus
- https://huggingface.co/datasets/browndw/human-ai-parallel-corpus-mini

### B. Domain bolsters

#### Academic abstracts human base

Role:

- required

Why:

- paper-domain signal is central to Laurito
- current discovery pool underrepresents true abstract register

Use:

- human-only baseline for atom prevalence and co-occurrence in academic register
- do not use alone for human-vs-LLM source association

Target:

- `2k` to `5k` abstracts

Preferred source:

- https://huggingface.co/datasets/ncbi/pubmed

#### Movie synopsis human base

Role:

- required

Why:

- `movie` is the strongest over-correction domain in current M2 results
- synopsis-specific narrative packaging needs stronger direct representation

Use:

- human-only narrative baseline

Target:

- `2k` to `5k` summaries

Preferred source:

- https://www.cs.cmu.edu/~ark/personas/

#### Product description human base

Role:

- required

Why:

- product pitch language is underrepresented in the current discovery base
- promotional bundle support should not depend only on Laurito

Use:

- human-only product-description baseline using metadata descriptions rather than reviews when possible

Target:

- `2k` to `5k` product descriptions

Preferred sources:

- https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
- https://snap.stanford.edu/data/amazon/productGraph/

#### Target-domain LLM supplements

Role:

- required only if public source-labeled target-domain corpora remain too sparse

Why:

- paper, product, and movie target domains otherwise remain weak on the LLM side

Use:

- automatically templated prompts only
- no manual prompt authoring
- generate from metadata or titles
- use `2` open-weight instruct models
- one generation per prompt
- keep these as a distinct stratum, not mixed blindly into the core

Target:

- `1k` to `1.5k` texts per domain

Rule:

- these supplements should be used to improve domain coverage, not to define bundle structure alone

### C. Controlled confirmation

#### HC3 Plus

Role:

- required

Why:

- specifically covers semantic-invariant tasks like summarization, translation, and paraphrasing
- useful for verifying that discovered bundles survive when semantics are tightly constrained

Use:

- not as the main discovery pool by default
- use as confirmation and sensitivity analysis

Target:

- `4k` to `8k` excerpts

Source:

- https://huggingface.co/papers/2309.02731

#### Local rewrite-controlled pairs

Role:

- required

Why:

- gives the strongest project-specific semantic control

Use:

- test whether bundle scores remain informative on content-preserving rewrites
- should not drive initial ontology formation

Target:

- use all available high-quality rewrites after filtering

### D. Ecological validation

#### Laurito paper/product/movie

Role:

- required

Why:

- direct target task
- judge decisions, not only source labels

Use:

- hold out from primary bundle discovery
- use only for D3 ecological validation

Rule:

- Laurito should not define bundle structure from scratch

### E. Held-out transfer

#### M4 or equivalent multi-generator benchmark

Role:

- optional but strongly preferred

Why:

- useful as a held-out stress test for unseen domains and unseen generators

Use:

- transfer-only
- not needed for initial bundle discovery

Source:

- https://aclanthology.org/2024.eacl-long.83/

## Sampling constraints

These constraints matter as much as dataset choice.

### Text unit

- one text excerpt or answer per row
- no prompt-pair objects as the unit for D2 discovery

### Length

- preferred range: `120` to `800` tokens
- allow shorter HC3-style answers down to about `80` tokens
- exclude extremely short snippets that cannot support discourse atoms

### Prompt-group caps

- no more than `2` to `3` model outputs per prompt group in the discovery core
- no more than `1` sampled continuation per seed chunk for HAP-E-like corpora in the primary pool

Reason:

- otherwise prompt duplication overwhelms the covariance structure

### Domain balance

- no single corpus should exceed `40%` of the main discovery pool
- no single item type should exceed `25%` of the discovery core before explicit domain-bolster strata are added

### Generator balance

- do not let one model family dominate the LLM side
- ChatGPT-only discovery is insufficient

### Register independence

- keep Laurito out of the primary discovery pool
- keep target-domain LLM supplements flagged as a separate stratum

## Minimum defensible configuration

If compute and storage remain constrained, the smallest defensible bundle-creation stack is:

1. `HC3`
2. `H-LLMC2`
3. `HAP-E mini`
4. human-only academic abstract bolster
5. human-only movie synopsis bolster
6. human-only product-description bolster
7. `HC3 Plus` for confirmation
8. Laurito for ecological validation only

## Preferred configuration

The preferred stack is:

1. `HC3`
2. `H-LLMC2`
3. `HAP-E`
4. PubMed abstract bolster
5. CMU Movie Summary bolster
6. Amazon product-description bolster
7. small target-domain LLM supplements for paper/product/movie
8. `HC3 Plus`
9. local rewrite-controlled pairs
10. Laurito
11. M4 transfer-only evaluation

## What this specification is trying to avoid

- deriving bundles mainly from Laurito
- deriving bundles mainly from QA/helpful-assistant text
- mistaking prompt duplication for atom co-occurrence
- mistaking one generator family for general LLM style
- using paired corpora as the sole discovery basis

## Decision

For this project, bundle creation should be based on:

- `general domain-based source-labeled samples` as the main discovery substrate
- `paired semantic-control corpora` as confirmation layers
- `Laurito` as ecological validation

That is the most defensible path for later mechanistic interpretation.
