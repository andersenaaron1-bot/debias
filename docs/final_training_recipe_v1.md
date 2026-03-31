# Final Training Recipe v1

This is the current full training recipe for the reward-invariance project.

It upgrades the earlier executable `M2` by adding the broader pointwise anchor and by making the architecture explicit.

## Architecture

Base model:

- `google/gemma-2-9b-it`

Adapter:

- LoRA on attention and MLP projection layers
- target modules:
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj`
  - `gate_proj`
  - `up_proj`
  - `down_proj`

Heads on the shared pooled representation:

- `value_head`
  - scalar utility for pairwise preference and pointwise reward supervision
- `attribute_heads`
  - `helpfulness`
  - `correctness`
  - `coherence`
  - `complexity`
  - `verbosity`
- `cue_heads`
  - `academic_formality`
  - `safety_corporate_tone`
  - `promotional_sales_tone`
  - `narrative_packaging`
  - `template_boilerplate`
  - `verbosity_compression`
  - `hedging_certainty`

The cue heads are attached through gradient reversal so minimizing the cue loss discourages cue information in the shared representation.

## Full loss profile

The implemented objective is:

```text
L = L_pref
  + L_anchor
  + L_inv
  + L_cue
```

Expanded:

```text
L_pref = pairwise_logsigmoid(value_head(chosen), value_head(rejected))

L_anchor = lambda_anchor_utility * MSE(value_head(x), utility_target)
         + lambda_anchor_attr * RobustMean_k MSE(attribute_head_k(x), attribute_target_k)

L_inv = lambda(step) * MSE(value_head(x_a), value_head(x_b))

L_cue = lambda_cue * RobustMean_j BCE(cue_head_j(GRL(x)), cue_target_j)
```

Where:

- `RobustMean`
  - is implemented as mean-plus-max interpolation over attribute heads or cue families
  - controlled by `lambda_group`
- `lambda(step)`
  - is a linear ramp for the paired invariance stream

This is not a full dual-optimizer DRO implementation. It is the current bounded version that is both executable and aligned with the project hypothesis.

## Final datasets

Training:

- pairwise anchor:
  - [pref_pairs_train.jsonl](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/data/derived/pref_pairs_shp2/pref_pairs_train.jsonl)
  - [pref_pairs_val.jsonl](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/data/derived/pref_pairs_shp2/pref_pairs_val.jsonl)
- pointwise anchor:
  - [anchor_train.jsonl](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/data/derived/helpsteer2_anchor/anchor_train.jsonl)
  - [anchor_val.jsonl](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/data/derived/helpsteer2_anchor/anchor_val.jsonl)
- paired invariance for `M2`:
  - [style_groups_train.jsonl](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/data/derived/style_groups/m2_publishable_v1/style_groups_train.jsonl)
  - [style_groups_val.jsonl](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/data/derived/style_groups/m2_publishable_v1/style_groups_val.jsonl)
- paired invariance for `M3`:
  - [style_groups_train.jsonl](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/data/derived/style_groups/m3_publishable_v1/style_groups_train.jsonl)
  - [style_groups_val.jsonl](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/data/derived/style_groups/m3_publishable_v1/style_groups_val.jsonl)
- cue removal:
  - [corpus_scored_balanced_train.jsonl](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/data/derived/cue_discovery_v2/balanced_splits/corpus_scored_balanced_train.jsonl)
  - [corpus_scored_balanced_val.jsonl](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/data/derived/cue_discovery_v2/balanced_splits/corpus_scored_balanced_val.jsonl)

Evaluation:

- Laurito-style domains under `data/paper`, `data/product`, `data/movie`
- SHP preference retention
- style sensitivity on the rebuilt invariance suites
- reward benchmark sanity checks

The machine-readable dataset manifest is in [final_reward_alignment_v1.json](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/configs/datasets/final_reward_alignment_v1.json).

## Experiments

- `M2-full` config:
  - [m2_full_v1.json](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/configs/experiments/m2_full_v1.json)
- `M3` config:
  - [m3_full_v1.json](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/configs/experiments/m3_full_v1.json)

The intended comparison is:

- `M2-full`
  - with academic formality present in both paired invariance and cue removal
- `M3`
  - without academic formality in either location

That isolates whether the paper-domain effect is specifically tied to academic-formality suppression or to the broader nuisance-cue program.
