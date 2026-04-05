# D3 Ecological Validation

## Purpose

`D3` is the ecological bridge between:

- `D2` bundle support on broad discovery corpora
- later mechanistic localization inside Gemma

It answers:

`Which atom and bundle deltas actually track judge choices on Laurito-style paper, product, and movie comparisons?`

## Entry point

Use:

```bash
python -m aisafety.scripts.build_ecological_validation_d3 --help
```

Main script:

- [build_ecological_validation_d3.py](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/src/aisafety/scripts/build_ecological_validation_d3.py)

Core helper module:

- [ecology.py](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/src/aisafety/ontology/ecology.py)

## What it does

1. loads scored Laurito A/B trials or scores them with a reward model
2. collapses balanced A/B swaps into one de-biased pair-level decision
3. scores both texts in each pair with the D2 atom ontology
4. computes `llm - human` deltas for every atom and validated bundle
5. estimates ecological effect sizes overall and by domain

## Output

Default output directory:

- [d3_ecological_validation_v1](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/data/derived/style_groups/d3_ecological_validation_v1)

Files written:

- `summary.json`
- `atom_effects.json`
- `atom_effects.tsv`
- `bundle_effects.json`
- `bundle_effects.tsv`
- `pair_level_inputs.csv`
- `text_atom_scores.csv`

## Statistical basis

For each atom or bundle predictor, D3 computes:

- mean `llm - human` delta
- standardized signed effect on de-biased `llm_chosen`
- bootstrap 95% CI for that signed effect
- AUC for predicting whether the judge chose the LLM side
- Spearman correlation with the de-biased LLM margin
- per-domain versions for `paper`, `product`, and `movie`

`signed_effect_z` is the main ecological effect quantity.

Positive values mean:

`higher values of this cue on the LLM side are associated with the judge choosing the LLM side`

Negative values mean the opposite.

## Scope

D3 is not yet the mechanistic ontology.

It is the filter that shows which D2-supported or D2-exploratory bundles are actually relevant in the real judgment setting.

That means D3 should be used to decide:

- which bundles deserve SAE tracing first
- which bundles are probably not worth centering in the paper
- which domains show the strongest ecological dependence on each bundle

## Intended next step

After D3, construct `D4`:

- a reduced ontology containing only atoms and bundles with both
  - linguistic validity from D1/D2
  - ecological relevance from D3

That reduced D4 ontology is the object that should be taken into Gemma Scope and later feature-circuit analysis.
