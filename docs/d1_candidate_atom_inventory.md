# D1 Candidate Atom Inventory

## Status

`D1` is now implemented as a machine-readable candidate atom table:

- [candidate_atom_inventory_d1.tsv](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/data/derived/style_groups/candidate_atom_inventory_d1.tsv)

This is the first concrete deliverable from the ontology-definition phase described in:

- [ontology_definition_and_validation_plan.md](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/docs/ontology_definition_and_validation_plan.md)

## What D1 contains

Each row includes:

- `atom_id`
- `level`
- `level_name`
- `description`
- `theoretical_source`
- `extractor_type`
- `extractor_recipe`
- `content_leakage_risk`
- `likely_bundle_memberships`
- `priority`

This table is intentionally broader than the eventual validated ontology.

It is a `candidate inventory`, not the final reduced mechanistic ontology.

## Source legend

The source tags in the TSV correspond to these bases:

- `BIBER_MDA`
  Register variation and multidimensional analysis
  https://www.cambridge.org/core/books/register-variation-online/appendix-a-linguistic-features-included-in-the-multidimensional-analysis/F9F9B4FE80ED9B9A3927AD589C31E815
- `NGUYEN_SOCIOLING`
  Computational sociolinguistics survey
  https://direct.mit.edu/coli/article/42/3/537/1536/Computational-Sociolinguistics-A-Survey
- `SUNDARARAJAN_WOODARD_2018`
  What represents style in authorship attribution?
  https://aclanthology.org/C18-1238/
- `FENG_HIRST_2014`
  Local discourse coherence for authorship attribution
  https://academic.oup.com/dsh/article/29/2/191/974196
- `METADISCOURSE_REVIEW_2023`
  Systematic review of metadiscourse in academic writing
  https://www.sciencedirect.com/science/article/pii/S0024384123000852
- `ZHANG_2016_2018_MDA`
  Multidimensional metadiscourse analyses across registers
  https://journals.sagepub.com/doi/abs/10.1177/1461445615623907
- `WU_2025_LLM_DETECTION`
  Survey on LLM-generated text detection
  https://direct.mit.edu/coli/article/51/1/275/127462/A-Survey-on-LLM-Generated-Text-Detection-Necessity
- `REINHART_2025_HAPE`
  Human-AI parallel corpus evidence
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11874169/
- `LAURITO_2025`
  Ecological human-vs-LLM judge benchmark
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12337326/

## Why this format

The TSV is designed to support three immediate uses:

1. extractor implementation
2. bundle definition and pruning
3. mechanistic targeting

The table is compact enough to edit, sort, and filter while staying grounded in extant literature.

## Important caution

Level-7 items such as `academic_abstract_register` or `helpdesk_assistant_register` are included as `macro script candidates`.

They are useful bridge variables, but they should not be treated as the first mechanistic primitives.

The first mechanistic primitives should still be lower-level atoms such as:

- nominalization
- passive constructions
- frame markers
- disclaimer lexicon
- benefit-first packaging
- entity-grid coherence

## Immediate next step

The next deliverable should operationalize this table further by adding:

- exact lexicons
- exact parser patterns
- exact normalization rules
- exact notes on known confounds

That is the shortest path from `D1` to statistical validation.
