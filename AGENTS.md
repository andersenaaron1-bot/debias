# AISafety LRZ Operations Guide

This file is the operational guide for developing, maintaining, and executing
the AISafety project on LRZ.

The scientific implementation path is `IMPLEMENTAION_PATH.md`. If this file and
`IMPLEMENTAION_PATH.md` disagree about project direction, follow
`IMPLEMENTAION_PATH.md`. If they disagree about LRZ commands, environment, or
maintenance practice, update this file.

## Documentation Policy

- `IMPLEMENTAION_PATH.md` is the active scientific implementation plan.
- `AGENTS.md` is the LRZ operations and maintenance guide.
- `paper/` contains manuscript construction material and bibliography only.
- Do not add live planning material under `docs/`.
- Retired or historical notes must not override these files.

## Current Work

The active work is the D4 mechanistic tracing refactor:

1. recover the failed content-anchor utility-control run with a valid
   deterministic pair-level split
2. complete the J0 SAE feature-localization pass
3. inspect and falsify top atom features
4. build atom-to-bundle feature graphs
5. run diagnostic cross-model contrasts only after stable J0 candidates exist
6. reserve causal claims for intervention tests

Do not train a broad new adapter grid. Do not define the ontology from repair
ablations. Do not make utility-independence or causal claims until the required
controls and interventions exist.

## LRZ-Only Execution

All GPU, model, dataset-pack, reward-evaluation, SAE, and intervention runs are
LRZ runs. The local workstation is for editing, file inspection, and lightweight
unit tests only.

Standard LRZ environment variables:

```bash
export WORKDIR=/path/to/aisafety
export HF_HOME=$WORKDIR/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export PYTHONUNBUFFERED=1
```

When using Slurm, keep logs with the job output and copy important logs into
the artifact directory before interpreting results.

## Repository Layout

- `IMPLEMENTAION_PATH.md`: active scientific and engineering implementation path
- `AGENTS.md`: LRZ operations guide
- `paper/references.bib`: bibliography used by the paper and implementation path
- `paper/workshop_paper_draft.tex`: manuscript draft
- `configs/ontology/`: ontology configs
- `configs/datasets/`: dataset-pack configs
- `configs/experiments/`: reward-judge experiment configs
- `cluster/lrz/`: LRZ Slurm wrappers
- `src/aisafety/scripts/`: CLI entry points
- `src/aisafety/`: package code
- `tests/`: lightweight validation tests
- `data/derived/`: derived datasets
- `artifacts/`: reward, evaluation, and mechanistic outputs

Avoid committing large raw data or model artifacts.

## Canonical Scripts

Dataset and ontology:

```bash
python -m aisafety.scripts.build_bundle_validation_d2 --help
python -m aisafety.scripts.build_ecological_validation_d3 --help
python -m aisafety.scripts.build_d4_dataset_pack --help
```

Reward training and evaluation:

```bash
sbatch cluster/lrz/train_from_config.sbatch configs/experiments/j0_anchor_v1_h100compact.json
sbatch cluster/lrz/eval_reward_suite.sbatch artifacts/reward/j0_anchor_v1_h100compact
```

D4 residual recovery:

```bash
python -m aisafety.scripts.run_d4_atom_recovery --help
```

D4 SAE feature analysis:

```bash
python -m aisafety.scripts.run_d4_sae_feature_analysis --help
```

## Immediate LRZ Commands

Recover the fixed content-anchor utility control:

```bash
python -m aisafety.scripts.run_d4_atom_recovery \
  --manifest-json data/derived/d4_dataset_pack_v1/manifest.json \
  --reward-run-dir artifacts/reward/j0_anchor_v1_h100compact \
  --content-anchor-only \
  --content-max-pairs 4000 \
  --out-dir artifacts/mechanistic/d4_j0_atom_recovery_v1_resplit
```

Run a J0 SAE scout pass:

```bash
python -m aisafety.scripts.run_d4_sae_feature_analysis \
  --manifest-json data/derived/d4_dataset_pack_v1/manifest.json \
  --reward-run-dir artifacts/reward/j0_anchor_v1_h100compact \
  --selected-layers 42,40,32,20,8 \
  --aggregation max \
  --content-max-pairs 500 \
  --out-dir artifacts/mechanistic/d4_j0_sae_feature_analysis_scout
```

Run the full J0 SAE pass only after the scout outputs are valid:

```bash
python -m aisafety.scripts.run_d4_sae_feature_analysis \
  --manifest-json data/derived/d4_dataset_pack_v1/manifest.json \
  --reward-run-dir artifacts/reward/j0_anchor_v1_h100compact \
  --selected-layers 39,40,41,42,32,36,20,28,12,4,8,16,24,1 \
  --aggregation max \
  --content-max-pairs 1000 \
  --out-dir artifacts/mechanistic/d4_j0_sae_feature_analysis_v1
```

Use `--skip-missing-sae` only for exploratory scout runs. Full runs should fail
when a required SAE is missing unless the missing layer is documented in the
run manifest.

## Run Integrity Checks

Before interpreting any mechanistic run, verify:

- the output directory is new or intentionally resumable
- `manifest.json` paths resolve on LRZ
- selected atoms match the D4 manifest
- content-anchor split counts have nonzero train, validation, and test rows
- model id, reward run directory, adapter path, SAE release, and selected layers
  are recorded
- skipped layers are recorded
- Slurm logs do not show out-of-memory, download, or partial-write failures
- output CSVs have nonempty `status=ok` rows for the intended stage

For the recovered utility-control run, never use the invalid old split as a
comparison baseline.

## Claim Gates

Residual recovery supports:

- atom information is recoverable from residual states

SAE localization supports:

- sparse features align with atom labels
- sparse features transfer to Laurito text-side atom scores
- sparse features align with pair-side judge decisions
- sparse features have measured utility overlap

Feature-card inspection supports:

- a feature is an interpretable candidate cue feature after top examples and
  close negatives are checked

Bundle graph construction supports:

- atom features form a bundle-level hypothesis

Only intervention tests support:

- causal influence on judge behavior
- repair or mitigation claims

## Software Maintenance

Follow the existing Python style:

- PEP 8
- 4-space indentation
- `snake_case`
- type hints and docstrings for new public code
- scripts runnable through `python -m ... --help`

Use small, focused tests in `tests/`. Prefer schema, split, helper, and config
validation tests. Do not rely on GPU tests for normal local validation.

The D4 mechanistic code should be refactored toward:

- `src/aisafety/mech/d4_io.py`
- `src/aisafety/mech/labels.py`
- `src/aisafety/mech/activations.py`
- `src/aisafety/mech/sae.py`
- `src/aisafety/mech/probes.py`
- `src/aisafety/mech/feature_ranking.py`
- `src/aisafety/mech/bundles.py`
- `src/aisafety/mech/examples.py`
- `src/aisafety/mech/interventions.py`

Keep the current CLI scripts as compatibility wrappers while the modules are
introduced.

## Artifact Naming

Use explicit stage, judge, and purpose names:

- `artifacts/mechanistic/d4_j0_atom_recovery_v1_resplit`
- `artifacts/mechanistic/d4_j0_sae_feature_analysis_scout`
- `artifacts/mechanistic/d4_j0_sae_feature_analysis_v1`
- `artifacts/mechanistic/d4_j0_bundle_graph_v1`
- `artifacts/mechanistic/d4_j0_intervention_scout_v1`

For cross-model contrasts, include the judge id:

- `artifacts/mechanistic/d4_jrepair_all_sae_feature_analysis_v1`
- `artifacts/mechanistic/d4_jrepair_loo_template_sae_feature_analysis_v1`

## Bibliography Maintenance

When a new source affects project direction or paper claims:

- add it to `paper/references.bib`
- cite it by key in `IMPLEMENTAION_PATH.md` or the paper
- do not rely on uncited background memory for methodological claims

## Hard Rules

1. Use broad corpora for ontology discovery and Laurito for ecological
   screening.
2. Use J0 as the anchor for cue-reliance claims.
3. Use repair and leave-one-out runs as diagnostic perturbations.
4. Keep mechanistic analysis atom-first and bundle-aware.
5. Do not treat broad register labels as first mechanistic primitives.
6. Do not train an exhaustive adapter grid.
7. Do not make causal claims before intervention tests.
8. Do not claim utility independence until content-anchor controls are valid.
9. Keep execution guidance LRZ-centered.
10. Keep live planning out of `docs/`.

