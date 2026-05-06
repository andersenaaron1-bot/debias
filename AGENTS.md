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

1. maintain the recovered content-anchor utility-control path with a valid
   deterministic pair-level split
2. maintain the J0 SAE feature-localization and merged candidate registry
3. run broad human-vs-LLM candidate-feature alignment on HC3/HC+/H-LLMC2/HAP-E
   style pair corpora before intervention claims
4. inspect and falsify top atom and bundle features with close negatives
5. build atom-to-bundle feature graphs from stable candidates
6. run diagnostic cross-model contrasts only after stable J0 candidates exist
7. reserve causal claims for intervention tests

Do not train a broad new adapter grid. Do not define the ontology from repair
ablations. Do not make utility-independence or causal claims until the required
controls, broad human-vs-LLM alignment checks, and interventions exist.

## LRZ-Only Execution

All GPU, model, dataset-pack, reward-evaluation, SAE, and intervention runs are
LRZ runs. The local workstation is for editing, file inspection, and lightweight
unit tests only.

LRZ workflow facts:

- LRZ AI Systems use Slurm for batch jobs.
- LRZ AI Systems use Enroot/Pyxis for containers. Use Slurm container flags,
  not local `docker`.
- Enroot is not available on SSH login nodes. Container tests must be submitted
  as Slurm jobs.
- `lrz-cpu` is the CPU partition and requires `--qos=cpu`.
- GPU partitions require `--gres=gpu:1` or the appropriate GPU request.
- Tools such as local `module`, `conda`, and `pip` are not the project
  dependency strategy on LRZ. Use the project container.
- The code checkout may be small and live under `~/debias`; large artifacts and
  derived data may live on DSS. Do not assume `~/debias/artifacts` or
  `~/debias/data/derived` contains the real run outputs.

Primary paths for the current LRZ state:

```bash
export WORKDIR=$HOME/debias
export ARTROOT=/dss/dssfs04/lwp-dss-0002/pn76ko/pn76ko-dss-0000/proc_mining_dfg/go75meh2/debias
export IMAGE=ghcr.io#andersenaaron1-bot/debias:sae-mech-v1
export PYTHONPATH=$WORKDIR/src
export HF_HOME=$ARTROOT/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export PYTHONUNBUFFERED=1
```

Use `WORKDIR` for the current code and `ARTROOT` for existing data, adapters,
cache, logs, and mechanistic outputs. Prefer absolute paths for Slurm commands.

Container syntax on LRZ:

```bash
--container-image="$IMAGE"
--container-mounts="$WORKDIR:$WORKDIR,$ARTROOT:$ARTROOT,$ARTROOT:/workspace"
--container-workdir="$WORKDIR"
--container-env=PYTHONPATH,HF_HOME,TRANSFORMERS_CACHE,HF_DATASETS_CACHE
```

For GHCR images, Pyxis/Enroot uses `#` between registry and image path:

```bash
ghcr.io#andersenaaron1-bot/debias:sae-mech-v1
```

Use `sae-mech-v1` for SAE work. The image is intentionally headless and should
not import `torchvision`; the SAE/text stack does not need it, and optional
vision imports have caused `torchvision::nms` binary-registration failures on
LRZ.
The image must keep a driver-compatible Torch stack. Current SAE builds use
`torch==2.6.0` with CUDA 11.8, constrained by
`requirements/sae-container-constraints.txt`. If a smoke test reports CUDA 13
or `torch 2.11`, pip upgraded Torch during the image build; rebuild with the
SAE container constraints.
Use an official GPU-enabled PyTorch or NGC base image. A plain `python:slim`
base can expose `/dev/nvidia*` but still leave PyTorch with zero CUDA devices
inside Pyxis.
Keep `gcc`, `g++`, and `make` in the runtime image. Triton may JIT small CUDA
launch modules during model loading or bitsandbytes setup; without a compiler,
SAE runs fail with `Failed to find C compiler`.

If GHCR import fails, the image may be private or credentials may be missing.
Resolve the container import first; do not fall back to system Python for model
or SAE jobs.

NGC fallback:

- `nvcr.io#nvidia/pytorch:24.05-py3` imports on LRZ and provides a public
  PyTorch/CUDA base.
- That NGC image uses Python 3.10. Do not install `requirements/cluster.txt`
  unchanged into it because the repo cluster file currently pins Python 3.11+
  packages such as `numpy==2.3.5`.
- If using the NGC fallback, install Python 3.10-compatible dependencies into a
  DSS `--target` directory and prepend that directory to `PYTHONPATH`.
- Do not use `python -m venv` in that NGC image; it lacks `ensurepip`.

NGC dependency setup, CPU partition:

```bash
cd "$WORKDIR" && DEPS="$ARTROOT/.python_deps/sae-ngc-24.05-py310" && sbatch --parsable --partition=lrz-cpu --qos=cpu --job-name=setup-sae-ngc --cpus-per-task=4 --mem=48G --time=01:30:00 --chdir="$WORKDIR" --output="$ARTROOT/slurm_logs/%x-%j.out" --error="$ARTROOT/slurm_logs/%x-%j.err" --container-image="nvcr.io#nvidia/pytorch:24.05-py3" --container-mounts="$WORKDIR:$WORKDIR,$ARTROOT:$ARTROOT,$ARTROOT:/workspace" --container-workdir="$WORKDIR" --export=ALL,PIP_NO_CACHE_DIR=1,PIP_DISABLE_PIP_VERSION_CHECK=1,PIP_CACHE_DIR="$ARTROOT/.cache/pip",PYTHONPATH="$DEPS:$WORKDIR/src" --wrap="rm -rf '$DEPS' && mkdir -p '$DEPS' && python -m pip install --target '$DEPS' --upgrade 'numpy==2.2.6' 'scipy==1.15.3' 'pandas==2.3.3' 'scikit-learn==1.7.2' 'accelerate==1.12.0' 'transformers==4.57.3' 'peft==0.18.1' 'datasets==2.21.0' 'huggingface_hub==0.36.0' 'sentencepiece==0.2.1' 'httpx==0.28.1' 'safetensors==0.7.0' 'bitsandbytes>=0.43.0' 'sae-lens>=6.7.0' && PYTHONPATH='$DEPS:$WORKDIR/src' python -c 'import torch, numpy, pandas, sklearn, transformers, peft, datasets, sae_lens, aisafety; print(\"torch\", torch.__version__, torch.__file__); print(\"numpy\", numpy.__version__, numpy.__file__); print(\"sae_lens ok\"); print(\"aisafety ok\")'"
```

Keep Slurm logs under DSS:

```bash
mkdir -p "$ARTROOT/slurm_logs"
```

LRZ references:

- AI Systems compute partitions and GPU memory:
  `https://doku.lrz.de/1-general-description-and-resources-10746641.html`
- Enroot introduction and login-node limitation:
  `https://doku.lrz.de/4-1-enroot-introduction-1895502566.html`
- Interactive Slurm jobs, `lrz-cpu --qos=cpu`, and Pyxis examples:
  `https://doku.lrz.de/6-running-applications-as-interactive-jobs-on-the-lrz-ai-systems-10746640.html`
- Single-GPU batch jobs and Pyxis batch guidance:
  `https://doku.lrz.de/5-2-slurm-batch-jobs-single-gpu-1898974516.html`

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

D4 broad human-vs-LLM candidate alignment:

```bash
python -m aisafety.scripts.build_d4_human_llm_alignment_pairs --help
python -m aisafety.scripts.run_d4_candidate_feature_pair_alignment --help
python -m aisafety.scripts.inspect_d4_candidate_alignment --help
```

Run these help commands inside the container or with `PYTHONPATH=$WORKDIR/src`
when only imports from this repo are needed.

## LRZ Preflight

Do not start with full SAE over every adapter. First verify paths and container
execution.

Static artifact discovery from the login node:

```bash
find "$HOME" /dss/dssfs04 -name value_head.pt -o -name adapter_model.safetensors -o -path "*/d4_dataset_pack_v1/manifest.json" 2>/dev/null
```

Static check of the current DSS artifact root:

```bash
cd "$WORKDIR" && test -f "$ARTROOT/data/derived/d4_dataset_pack_v1/manifest.json" && test -f "$ARTROOT/artifacts/reward/j0_anchor_v1_h100compact/value_head.pt" && echo OK
```

CPU Slurm jobs:

- Use `--partition=lrz-cpu --qos=cpu`.
- Use CPU jobs for filesystem, manifest, and non-Torch checks.
- Do not assume CPU jobs can validate CUDA or V100/A100/H100 runtime behavior.
- If Pyxis container submission fails on `lrz-cpu`, use CPU for non-container
  static checks only and run the shortest possible GPU container smoke test.

CPU static manifest check, no container:

```bash
cd "$WORKDIR" && sbatch --parsable --partition=lrz-cpu --qos=cpu --job-name=d4-static --cpus-per-task=2 --mem=4G --time=00:10:00 --chdir="$WORKDIR" --output="$ARTROOT/slurm_logs/%x-%j.out" --error="$ARTROOT/slurm_logs/%x-%j.err" --export=ALL,WORKDIR="$WORKDIR",ARTROOT="$ARTROOT",PYTHONPATH="$WORKDIR/src" --wrap="python - <<'PY'
from pathlib import Path
import json
root = Path('$ARTROOT')
manifest = root / 'data/derived/d4_dataset_pack_v1/manifest.json'
print('manifest', manifest, manifest.exists())
if manifest.exists():
    payload = json.load(open(manifest))
    print('d4_atoms', len(payload.get('d4_atoms', [])))
    for key, value in sorted((payload.get('outputs') or {}).items()):
        if value is None:
            print(key, 'None')
            continue
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        print(key, path.exists(), path)
PY"
```

GPU container smoke test:

```bash
cd "$WORKDIR" && sbatch --parsable --job-name=container-smoke --gres=gpu:1 --cpus-per-task=2 --mem=16G --time=00:10:00 --chdir="$WORKDIR" --output="$ARTROOT/slurm_logs/%x-%j.out" --error="$ARTROOT/slurm_logs/%x-%j.err" --container-image="$IMAGE" --container-mounts="$WORKDIR:$WORKDIR,$ARTROOT:$ARTROOT,$ARTROOT:/workspace" --container-workdir="$WORKDIR" --container-env=PYTHONPATH,HF_HOME,TRANSFORMERS_CACHE,HF_DATASETS_CACHE --export=ALL,PYTHONPATH="$WORKDIR/src",HF_HOME="$HF_HOME",TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE",HF_DATASETS_CACHE="$HF_DATASETS_CACHE" --wrap="python -c 'import torch, transformers, sae_lens, aisafety; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available()); print(\"sae_lens import ok\"); print(\"aisafety import ok\")'"
```

## Immediate LRZ Human-vs-LLM Alignment Commands

The current high-value next run is broad J0 candidate-feature alignment on
human-vs-LLM pairs. This is not another feature-discovery sweep. It freezes a
broad candidate registry from the merged SAE discovery outputs, scores
human/LLM pairs with J0, and tests whether activation deltas predict J0
LLM-minus-human reward margins after length/source controls.

Build human-vs-LLM alignment pairs from the normalized bundle-creation corpus.
Use `EXTRA_RECORDS_JSONL` for staged supplements such as HC3+ instead of
shell-concatenating files before submission:

```bash
cd "$WORKDIR" && sbatch --parsable --partition=lrz-cpu --qos=cpu --job-name=d4-hllm-pairs --cpus-per-task=2 --mem=32G --time=00:30:00 --chdir="$WORKDIR" --output="$ARTROOT/slurm_logs/%x-%j.out" --error="$ARTROOT/slurm_logs/%x-%j.err" --container-image="$IMAGE" --container-mounts="$WORKDIR:$WORKDIR,$ARTROOT:$ARTROOT,$ARTROOT:/workspace" --container-workdir="$WORKDIR" --container-env=PYTHONPATH --export=ALL,WORKDIR="$WORKDIR",ARTROOT="$ARTROOT",PYTHONPATH="$WORKDIR/src",RECORDS_JSONL="$ARTROOT/data/derived/bundle_creation_corpus_v1/all_records.jsonl",EXTRA_RECORDS_JSONL="$ARTROOT/data/external/bundle_creation_v1/hc3_plus_subset.jsonl",OUT_DIR="$ARTROOT/data/derived/d4_human_llm_alignment_pairs_strat10k_v2",INCLUDE_DATASETS=hc3:hc3_plus:h_llmc2:hape,REQUIRE_DATASETS=hc3:hc3_plus:h_llmc2:hape,CAP_STRATEGY=dataset_subset,MAX_PAIRS_PER_DATASET=0,MAX_TOTAL_PAIRS=10000,MAX_LLM_PER_GROUP=1 cluster/lrz/d4_human_llm_alignment_pairs.sbatch
```

Set `REQUIRE_DATASETS=hc3:hc3_plus:h_llmc2:hape` on broad confirmation jobs to
fail fast if any required source is absent from the emitted pair file. Use
colon-separated lists in Slurm `--export`; commas are parsed by Slurm as
variable separators.
For the broad confirmation pass, use `CAP_STRATEGY=dataset_subset`,
`MAX_PAIRS_PER_DATASET=0`, and `MAX_TOTAL_PAIRS=10000` to sample by
deterministic round-robin across dataset/subset strata instead of letting a
large subcategory dominate.

Queue the broad targeted candidate alignment pass:

```bash
cd "$WORKDIR" && PARTS="lrz-hgx-h100-94x4,lrz-dgx-a100-80x8,lrz-hgx-a100-80x4" && sbatch --parsable --partition="$PARTS" --job-name=d4-hllm-align --gres=gpu:1 --cpus-per-task=8 --mem=160G --time=10:00:00 --chdir="$WORKDIR" --output="$ARTROOT/slurm_logs/%x-%j.out" --error="$ARTROOT/slurm_logs/%x-%j.err" --container-image="$IMAGE" --container-mounts="$WORKDIR:$WORKDIR,$ARTROOT:$ARTROOT,$ARTROOT:/workspace" --container-workdir="$WORKDIR" --container-env=PYTHONPATH,HF_HOME,TRANSFORMERS_CACHE,HF_DATASETS_CACHE,HF_TOKEN,HUGGING_FACE_HUB_TOKEN --export=ALL,WORKDIR="$WORKDIR",ARTROOT="$ARTROOT",PYTHONPATH="$WORKDIR/src",HF_HOME="$HF_HOME",TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE",HF_DATASETS_CACHE="$HF_DATASETS_CACHE",PAIR_JSONL="$ARTROOT/data/derived/d4_human_llm_alignment_pairs_strat10k_v2/pairs.jsonl",CANDIDATE_SOURCE_DIR="$ARTROOT/artifacts/mechanistic/d4_j0_sae_merged_ontology_discovery_v1",REWARD_RUN_DIR="$ARTROOT/artifacts/reward/j0_anchor_v1_h100compact",OUT_DIR="$ARTROOT/artifacts/mechanistic/d4_j0_human_llm_candidate_alignment_strat10k_v2",MAX_PAIRS=0,MAX_CANDIDATES=900,MAX_FEATURES_PER_LAYER=90,RANDOM_CONTROLS_PER_LAYER=10,SOURCE_MIN_PAIRS=25,SCORE_BATCH_SIZE=4,SAE_BATCH_SIZE=4,SAE_TOKEN_CHUNK_SIZE=1024,MAX_LENGTH=512 cluster/lrz/d4_candidate_feature_pair_alignment.sbatch
```

Use `MAX_PAIRS=0` only after the capped run has completed and the output
manifest confirms stable pair counts and nonempty alignment rows.

Package and summarize the completed broad alignment run on LRZ:

```bash
cd "$WORKDIR" && python -m aisafety.scripts.inspect_d4_candidate_alignment --input "$ARTROOT/artifacts/mechanistic/d4_j0_human_llm_candidate_alignment_v1" --archive --no-print
```

After downloading the archive locally, extract and inspect with the same script:

```bash
python -m aisafety.scripts.inspect_d4_candidate_alignment --input artifacts/mechanistic/d4_j0_human_llm_candidate_alignment_v1.tar.gz --extract-to artifacts/mechanistic --top-k 40 --source-top-k 40
```

## Deferred LRZ Commands

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

Broad human-vs-LLM candidate alignment supports:

- frozen candidate features distinguish human from LLM text in paired corpora
- activation deltas align with J0 LLM-minus-human reward margins beyond Laurito
- source/domain stability and matched random-control comparisons are measured

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
- `artifacts/mechanistic/d4_j0_sae_merged_ontology_discovery_v1`
- `data/derived/d4_human_llm_alignment_pairs_v1`
- `artifacts/mechanistic/d4_j0_human_llm_candidate_alignment_v1`
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
