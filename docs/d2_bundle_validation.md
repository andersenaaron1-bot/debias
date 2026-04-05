# D2 Bundle Validation

## Purpose

`D2` turns the D1 candidate atom inventory into an empirical validation layer.

It does three things:

1. operationalizes each atom with an explicit extractor
2. scores those atoms on a discovery corpus
3. tests whether theory-seeded bundles are supported by selective atom co-occurrence

This is the stage at which higher-level bundles stop being purely hand-authored labels and become statistically argued groupings.

## Entry point

Use:

```bash
python -m aisafety.scripts.build_bundle_validation_d2 --help
```

Default input:

- [corpus.jsonl](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/data/derived/cue_discovery_v2/corpus.jsonl)

Default output directory:

- [d2_validation_v1](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/data/derived/style_groups/d2_validation_v1)

## What the script writes

- `summary.json`
- `atom_operationalization.json`
- `atom_summary.json`
- `pairwise_cooccurrence.csv`
- `bundle_validation.json`
- `bundle_validation.tsv`

Optional:

- `atom_scores.jsonl`

## Statistical foundation

The D2 implementation uses:

- atom prevalence and length sensitivity summaries
- within-item-type z-scoring before co-occurrence analysis
- pairwise Pearson and Spearman co-occurrence estimates
- co-activation Jaccard overlap
- bootstrap confidence intervals for within-bundle mean pairwise correlation
- random-set null comparisons for empirical `p`-values
- data-driven clustering over the atom correlation graph
- bundle-to-cluster overlap via Jaccard
- first-PC explained variance as a compact convergence signal

This is intentionally a `convergent evidence` standard rather than a single test.

## Interpretation

For a bundle to be treated as supported, the important outputs are:

- positive within-bundle mean pairwise correlation
- stronger within-bundle correlation than random atom sets
- non-trivial co-clustering probability across bootstrap resamples
- non-trivial overlap with a data-driven derived cluster

The current code marks bundles as `supported` or `exploratory` using simple heuristic thresholds.

Those thresholds are not the scientific claim.

The scientific claim should come from the statistics themselves.

## Important scope rule

Level-7 register variables such as `academic_abstract_register` remain bridge variables.

They are still scored and reported, but they are excluded from the primary validation subset used for bundle derivation.

That keeps the co-occurrence analysis focused on lower-level atoms rather than circularly validating macro labels from macro labels.
