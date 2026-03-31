# First Experiment Setup

## Goal

Run a narrow, publishable first experiment that tests one claim cleanly:

`counterfactual invariance to a small core set of authorship-correlated surface cues reduces Laurito-style bias, and formality is a likely driver of the paper-domain effect`

This experiment is intentionally narrower than the full project. It avoids:

- large-scale manual prompt authoring
- large-scale rewrite generation on flagship models
- training on mixed axes that are likely to change content, such as simplification and sentence splitting

## Ontology

The first training ontology has four `Tier A` axes:

- `formality`
  - main hypothesis axis
  - expected to matter most for paper abstracts
- `fluency`
  - captures edit polish and grammar cleanup
- `detox_tone`
  - captures tone shifts into more neutral, safety-adjacent register
- `paraphrase_surface`
  - captures shallow rewording without a strong target register

The following are excluded from `v1` training and reserved for later stress tests:

- `simplification`
- `splitting`
- `archaic`
- domain-specific generated cues such as product hype or synopsis dramaticity

Reason: in a time-constrained first pass, the cleanest result is to train on cue families that are most defensibly nuisance-like after semantic filtering.

## Data Foundation

### Preference anchor

Use the existing SHP pairwise dataset as the anchor objective for the first run because the current trainer already supports it directly.

This is not the final data mixture for the project. It is the fastest clean baseline that can be run with the current code.

### Invariance data

Use `data/derived/style_groups/style_axes_config_publishable_v1.json`.

Build the style groups with aggressive semantic filtering:

```powershell
python -m aisafety.scripts.build_style_groups_hf `
  --config data\derived\style_groups\style_axes_config_publishable_v1.json `
  --out-dir data\derived\style_groups\publishable_v1 `
  --apply-number-filter `
  --apply-embed-filter `
  --sim-threshold 0.80 `
  --embed-model-id sentence-transformers/all-MiniLM-L6-v2 `
  --max-groups-per-axis 25000
```

Expected properties of the resulting training mixture:

- balanced across the four axes
- almost entirely sourced from public rewrite corpora
- no reliance on expensive prompt generation for training
- group split by hash, so train/val/test leakage is controlled at the meaning-group level

### Generated evaluation only

Do not use generated rewrites at scale for training in the first experiment.

Use a small generated evaluation suite only:

- sample `100` paper items, `100` product items, `100` movie items
- create `2` rewrites per item for the domain-relevant cue
- keep the total generation budget around `600` rewrites

Domain-relevant generated evaluation axes:

- paper: `academic_formality`
- product: `marketing_hype`
- movie: `dramatic_packaging`

These are for evaluation only in `v1`. They test whether the generic invariance adapter transfers to the exact nuisance channels that matter in Laurito-style settings.

## Training Objective

Use the current mixed objective in `train_reward_lora.py` as the first executable baseline:

- preference stream: SHP pairwise loss
- invariance stream: score-matching loss across meaning-preserving variants

This is still a scalar reward model, not yet the full source-grounded multi-objective judge. That is acceptable for the first experiment because the immediate question is whether the core invariance recipe changes bias in the expected direction.

## Concrete Run Matrix

Run exactly three models first.

### Run A: preference-only baseline

Use the same data inputs, but disable the invariance effect by setting:

- `--pref-prob 1.0`
- `--lambda-max 0.0`

### Run B: core invariance without formality

This isolates the marginal effect of the formality axis.

Use:

- `--exclude-axes formality`

### Run C: core invariance with formality

Use all four axes from `publishable_v1`.

## Recommended Training Hyperparameters

Start with one stable recipe rather than a broad sweep:

- model: `google/gemma-2-9b-it`
- quantization: `--use-4bit`
- precision: `--bf16`
- memory: `--gradient-checkpointing`
- LoRA rank: `32`
- LoRA alpha: `64`
- LoRA dropout: `0.05`
- max steps: `4000`
- pref probability: `0.75`
- preference batch pairs: `8`
- invariance batch groups: `16`
- lambda max: `0.20`
- lambda ramp frac: `0.10`

Example training command for `Run C`:

```powershell
python -m aisafety.scripts.train_reward_lora `
  --model-id google/gemma-2-9b-it `
  --pref-train-jsonl data\derived\pref_pairs_shp2\pref_pairs_train.jsonl `
  --pref-val-jsonl data\derived\pref_pairs_shp2\pref_pairs_val.jsonl `
  --style-train-jsonl data\derived\style_groups\publishable_v1\style_groups_train.jsonl `
  --style-val-jsonl data\derived\style_groups\publishable_v1\style_groups_val.jsonl `
  --output-dir artifacts\reward\publishable_v1_core_all `
  --use-4bit `
  --bf16 `
  --gradient-checkpointing `
  --lora-r 32 `
  --lora-alpha 64 `
  --max-steps 4000 `
  --pref-prob 0.75 `
  --lambda-max 0.20 `
  --lambda-ramp-frac 0.10
```

## Evaluation

Use Laurito as an ecological stress test, not the main statistical foundation.

Primary comparisons:

- `Run A` vs `Run B`
- `Run B` vs `Run C`

That gives you the cleanest answer to:

- does generic invariance reduce natural authorship bias at all?
- does `formality` specifically account for the paper-domain shift?

Minimal evaluation battery:

- preference retention on SHP
- style sensitivity reduction on the four training axes
- Laurito bias evaluation by domain
- generated domain-specific mini-suite
- reward benchmark sanity check

## Success Criteria

For a first-pass result, I would treat these as meaningful:

- `Run C` reduces paper-domain favoritism relative to `Run B`
- `Run B` and `Run C` both reduce overall authorship bias relative to `Run A`
- SHP preference retention does not collapse
- style sensitivity drops strongly on the training axes
- transfer appears on at least one held-out generated domain-specific cue

## What This Experiment Does Not Claim

This experiment does not show invariance to all surface cues.

It shows a stronger and publishable claim:

`a balanced core nuisance ontology, especially including formality, causally contributes to reducing authorship-linked preference bias`

## Immediate Follow-up If This Works

If `Run C` clearly beats `Run B` on paper abstracts, the next step is not a wider net immediately.

The next step is:

- keep the four-axis core
- add a small number of domain-specific evaluation-only rewrite suites
- broaden the anchor beyond SHP
- move from scalar reward modeling toward source-grounded multi-objective judging
