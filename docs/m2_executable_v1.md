# M2 Executable v1

This is the first runnable `M2` configuration in the current codebase.

It includes:

- SHP pairwise preference anchoring
- Tier 1 paired invariance on `formality`, `fluency`, `detox_tone`, and `paraphrase_surface`
- Tier 2 weak-label cue supervision with gradient-reversal cue removal

It does not yet include the broader HelpSteer-style anchor or explicit worst-group optimization.

## Data inputs

- SHP anchor:
  - `data/derived/pref_pairs_shp2/pref_pairs_train.jsonl`
  - `data/derived/pref_pairs_shp2/pref_pairs_val.jsonl`
- Rebuilt Tier 1 style groups:
  - `data/derived/style_groups/m2_publishable_v1/style_groups_train.jsonl`
  - `data/derived/style_groups/m2_publishable_v1/style_groups_val.jsonl`
- Balanced cue corpus with broadened generator coverage:
  - `data/derived/cue_discovery_v2/balanced_splits/corpus_scored_balanced_train.jsonl`
  - `data/derived/cue_discovery_v2/balanced_splits/corpus_scored_balanced_val.jsonl`

## Rationale for the current mix

- `pref_prob = 0.60`
  - keeps the SHP anchor dominant
- `cue_prob = 0.15`
  - adds steady cue suppression pressure without overwhelming the anchor
- remaining `0.25`
  - is the paired invariance stream

This is the most defensible first `M2` setting with the current trainer.

## Recommended command

```powershell
python -m aisafety.scripts.train_reward_lora `
  --model-id google/gemma-2-9b-it `
  --pref-train-jsonl data\derived\pref_pairs_shp2\pref_pairs_train.jsonl `
  --pref-val-jsonl data\derived\pref_pairs_shp2\pref_pairs_val.jsonl `
  --style-train-jsonl data\derived\style_groups\m2_publishable_v1\style_groups_train.jsonl `
  --style-val-jsonl data\derived\style_groups\m2_publishable_v1\style_groups_val.jsonl `
  --cue-train-jsonl data\derived\cue_discovery_v2\balanced_splits\corpus_scored_balanced_train.jsonl `
  --cue-val-jsonl data\derived\cue_discovery_v2\balanced_splits\corpus_scored_balanced_val.jsonl `
  --output-dir artifacts\reward\m2_executable_v1 `
  --use-4bit `
  --bf16 `
  --gradient-checkpointing `
  --lora-r 32 `
  --lora-alpha 64 `
  --lora-dropout 0.05 `
  --max-steps 4000 `
  --warmup-steps 200 `
  --gradient-accumulation-steps 4 `
  --pref-prob 0.60 `
  --cue-prob 0.15 `
  --pref-batch-pairs 8 `
  --inv-batch-groups 16 `
  --cue-batch-size 16 `
  --lambda-max 0.20 `
  --lambda-ramp-frac 0.10 `
  --lambda-cue 1.0 `
  --cue-grl-scale 1.0 `
  --cue-families academic_formality,safety_corporate_tone,promotional_sales_tone,narrative_packaging,template_boilerplate,verbosity_compression,hedging_certainty
```

## Expected outputs

- `artifacts/reward/m2_executable_v1/run_config.json`
- `artifacts/reward/m2_executable_v1/metrics_train.csv`
- `artifacts/reward/m2_executable_v1/lora_adapter/`
- `artifacts/reward/m2_executable_v1/value_head.pt`
- `artifacts/reward/m2_executable_v1/cue_heads.pt`

## Immediate follow-up after training

Run:

- Laurito domain evaluation
- SHP preference retention
- style sensitivity evaluation on the rebuilt `m2_publishable_v1` axes
- reward benchmark sanity checks

If this run is stable, the next trainer extension should be the broader non-SHP anchor, not a wider nuisance ontology.
