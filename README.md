# AISafety

This repo is focused on one project:

- Train and evaluate LoRA reward adapters that are as invariant as possible to stylistic or surface-form variation, so preference judgments depend on content rather than expression.

The active workflow is:

1. Build Laurito and HC3 human-vs-LLM evaluation trials.
2. Build parallel style groups for invariance supervision.
3. Build SHP-2 preference pairs for content-preference supervision.
4. Train a scalar reward model with a mixed preference + invariance objective.
5. Measure what the adapter gains and what it loses on preference retention, style sensitivity, Laurito bias, HC3 bias, rewrite-controlled trials, triads, and optional general benchmarks.

Older selector-LoRA, steering, probe, and stylometry branches are not part of the active scope anymore.

## Core Training

Build the two training inputs, then train the reward scorer:

```bash
python -m aisafety.scripts.build_style_groups_hf --help
python -m aisafety.scripts.build_pref_pairs_shp2 --help
python -m aisafety.scripts.train_reward_lora --help
```

The training objective is implemented in [train_reward_lora.py](src/aisafety/scripts/train_reward_lora.py):

- preference batches from SHP-2
- invariance batches from parallel rewrite groups
- a LoRA adapter on the backbone
- a scalar value head

## Evaluation

Primary evaluation scripts:

```bash
python -m aisafety.scripts.eval_pref_retention --help
python -m aisafety.scripts.eval_style_sensitivity --help
python -m aisafety.scripts.eval_laurito_bias_reward --help
python -m aisafety.scripts.build_hc3_trials_csv --help
python -m aisafety.scripts.eval_triads_reward --help
python -m aisafety.scripts.eval_reward_benchmarks --help
```

These cover:

- held-out preference retention
- shrinkage of style sensitivity
- human-vs-LLM preference shifts on Laurito
- HC3 human-vs-LLM preference shifts
- H/G/R triads using rewrite-controlled data
- optional transfer to candidate-selection benchmarks

## Rewrite-Controlled Diagnostics

Use OpenRouter rewrites to test whether preference shifts are mediated by expression:

```bash
python -m aisafety.scripts.prepare_trials --out artifacts/trials.csv

python -m aisafety.scripts.rewrite_laurito_trials_openrouter \
  --in-csv artifacts/trials.csv \
  --out-csv artifacts/trials_human_plain.csv \
  --dimension ai_tone \
  --target-label human_plain \
  --model openai/gpt-4o-mini
```

Related helpers:

```bash
python -m aisafety.scripts.build_openrouter_style_pairs --help
python -m aisafety.scripts.build_leakage_free_rewrite_trials --help
```

## Optional Adapter Analysis

For reward-adapter-specific analysis:

```bash
python -m aisafety.scripts.analyze_lora_weights --help
python -m aisafety.scripts.scan_reward_activation_fingerprints --help
python -m aisafety.scripts.pivot_activation_fingerprints --help
```
