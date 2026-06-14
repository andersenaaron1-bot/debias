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
python -m aisafety.scripts.build_d4_bundle_candidate_registry --help
python -m aisafety.scripts.run_d4_feature_perturbation --help
python -m aisafety.scripts.build_d4_surface_counterfactual_pairs --help
python -m aisafety.scripts.run_d4_surface_counterfactual_audit --help
python -m aisafety.scripts.run_d4_readout_surface_nulling --help
python -m aisafety.scripts.build_d4_human_llm_stage_contrast_pairs --help
python -m aisafety.scripts.run_d4_human_llm_stage_contrast --help
python -m aisafety.scripts.summarize_d4_human_llm_stage_contrasts --help
python -m aisafety.scripts.build_d4_bt_stage_contrast_pairs --help
python -m aisafety.scripts.run_d4_bt_stage_contrast --help
python -m aisafety.scripts.build_judge_competence_pref_pairs --help
python -m aisafety.scripts.build_d4_invariance_style_groups --help
```

Judge-reasoning trajectories:

```bash
python -m aisafety.scripts.build_judge_reasoning_source_pack --help
python -m aisafety.scripts.build_judge_reasoning_pairs --help
python -m aisafety.scripts.build_judge_reasoning_suite --help
python -m aisafety.scripts.run_judge_reasoning_trajectories --help
python -m aisafety.scripts.analyze_judge_reasoning_trajectories --help
python -m aisafety.scripts.analyze_judge_reasoning_mode_contrasts --help
python -m aisafety.scripts.build_judge_deliberation_source_pack --help
python -m aisafety.scripts.run_judge_reasoning_budget_sweep --help
python -m aisafety.scripts.analyze_judge_reasoning_budget_sweep --help
python -m aisafety.scripts.build_helpsteer2_matched_criterion_suite --help
python -m aisafety.scripts.analyze_helpsteer2_matched_criterion --help
python -m aisafety.scripts.read_helpsteer2_matched_criterion --help
python -m aisafety.scripts.build_helpsteer2_criterion_switch_suite --help
python -m aisafety.scripts.build_helpsteer2_criterion_confirmation --help
python -m aisafety.scripts.run_judge_criterion_switch_behavior --help
python -m aisafety.scripts.analyze_judge_criterion_confirmation --help
python -m aisafety.scripts.read_judge_criterion_confirmation --help
python -m aisafety.scripts.analyze_judge_criterion_confirmation_activations --help
python -m aisafety.scripts.run_judge_criterion_confirmation_patching --help
python -m aisafety.scripts.read_judge_criterion_confirmation_mechanistic --help
python -m aisafety.scripts.analyze_judge_criterion_switch_behavior --help
python -m aisafety.scripts.run_judge_criterion_switch_activations --help
python -m aisafety.scripts.analyze_judge_criterion_switch_decoders --help
python -m aisafety.scripts.analyze_judge_criterion_switch_pairs --help
python -m aisafety.scripts.run_judge_criterion_switch_patching --help
python -m aisafety.scripts.run_judge_factual_readout_activations --help
python -m aisafety.scripts.read_judge_readout_calibration --help
python -m aisafety.scripts.analyze_judge_reasoning_fixed_decoders --help
python -m aisafety.scripts.run_judge_reasoning_interventions --help
python -m aisafety.scripts.summarize_judge_reasoning_suite --help
```

Run these help commands inside the container or with `PYTHONPATH=$WORKDIR/src`
when only imports from this repo are needed.

## LRZ Judge-Reasoning Trajectory Scout

This suite asks when and how open-weight judges form pairwise verdicts across
quality, authorship, moral, safety, truthfulness, and related dimensions. It
stores exact token prefixes for replay, uses pair-grouped probes, and separates
descriptive trajectory results from intervention evidence.

The recommended first scout submits the pair build, Qwen3 base and post-trained
trajectory jobs, dependent analyses, and a combined summary:

```bash
cd "$WORKDIR" && RUN_TAG=judge_reasoning_qwen3_scout_v1 MAX_PAIRS_PER_DATASET=80 MAX_PAIRS=60 BRANCHES_PER_COMPARISON=3 bash cluster/lrz/submit_judge_reasoning_qwen3_matrix.sh
```

On the local multi-GPU host, use the corresponding resumable wrapper:

```bash
cd "$WORKDIR" && RUN_TAG=judge_reasoning_qwen3_scout_v1 GPU_A=1 GPU_B=7 MAX_PAIRS_PER_DATASET=60 MAX_PAIRS=60 BRANCHES_PER_COMPARISON=3 bash cluster/local/run_judge_reasoning_qwen3_matrix.sh
```

Build every currently staged dataset in
`configs/datasets/judge_reasoning_suite_v1.json`. `SKIP_MISSING=1` permits a
scout from the available domains while recording missing optional sources:

```bash
cd "$WORKDIR" && sbatch --parsable --partition=lrz-cpu --qos=cpu --job-name=judge-reason-pairs --cpus-per-task=2 --mem=32G --time=00:30:00 --chdir="$WORKDIR" --output="$ARTROOT/slurm_logs/%x-%j.out" --error="$ARTROOT/slurm_logs/%x-%j.err" --container-image="$IMAGE" --container-mounts="$WORKDIR:$WORKDIR,$ARTROOT:$ARTROOT,$ARTROOT:/workspace" --container-workdir="$WORKDIR" --container-env=PYTHONPATH --export=ALL,WORKDIR="$WORKDIR",ARTROOT="$ARTROOT",PYTHONPATH="$WORKDIR/src",CONFIG="$WORKDIR/configs/datasets/judge_reasoning_suite_v1.json",OUT_DIR="$ARTROOT/data/derived/judge_reasoning_suite_v1",MAX_PAIRS_PER_DATASET=200,SKIP_MISSING=1 cluster/lrz/judge_reasoning_suite_pairs.sbatch
```

Capture a single-model scout after the pair job succeeds. Use explicit layers
for model-stage comparisons:

```bash
cd "$WORKDIR" && PARTS="lrz-hgx-h100-94x4,lrz-dgx-a100-80x8,lrz-hgx-a100-80x4" && sbatch --parsable --partition="$PARTS" --job-name=judge-reason-trace --gres=gpu:1 --cpus-per-task=8 --mem=160G --time=10:00:00 --chdir="$WORKDIR" --output="$ARTROOT/slurm_logs/%x-%j.out" --error="$ARTROOT/slurm_logs/%x-%j.err" --container-image="$IMAGE" --container-mounts="$WORKDIR:$WORKDIR,$ARTROOT:$ARTROOT,$ARTROOT:/workspace" --container-workdir="$WORKDIR" --container-env=PYTHONPATH,HF_HOME,TRANSFORMERS_CACHE,HF_DATASETS_CACHE,HF_TOKEN,HUGGING_FACE_HUB_TOKEN --export=ALL,WORKDIR="$WORKDIR",ARTROOT="$ARTROOT",PYTHONPATH="$WORKDIR/src",HF_HOME="$HF_HOME",TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE",HF_DATASETS_CACHE="$HF_DATASETS_CACHE",COMPARISONS_JSONL="$ARTROOT/data/derived/judge_reasoning_suite_v1/comparisons.jsonl",MODEL_ID="Qwen/Qwen3-8B",RUN_LABEL=qwen3_8b_it,REASONING_MODES=thinking:direct,SELECTED_LAYERS=4:8:12:16:20:24:28:32,BRANCHES_PER_COMPARISON=3,MAX_PAIRS=60,MAX_NEW_TOKENS_THINKING=256,OUT_DIR="$ARTROOT/artifacts/mechanistic/judge_reasoning_trajectories_qwen3_8b_it_scout_v1" cluster/lrz/judge_reasoning_trajectories.sbatch
```

Slurm parses commas in `--export` as variable separators. These wrappers accept
colon-separated `REASONING_MODES`, `SELECTED_LAYERS`, `PROBE_TARGETS`,
`ALPHAS`, and `RANDOM_CONTROL_SEEDS`, then convert them to comma-separated
Python arguments. Use colons for those values in direct `sbatch --export`
commands.

Trajectory capture writes complete shards and their metadata incrementally. To
resume the same output directory after preemption, resubmit with `RESUME=1`;
completed trace IDs are skipped. Without `RESUME=1`, an existing trace artifact
causes a fail-fast error instead of being overwritten.

Analyze a completed trace artifact on the CPU partition:

```bash
cd "$WORKDIR" && sbatch --parsable --partition=lrz-cpu --qos=cpu --job-name=judge-reason-analyze --cpus-per-task=8 --mem=96G --time=04:00:00 --chdir="$WORKDIR" --output="$ARTROOT/slurm_logs/%x-%j.out" --error="$ARTROOT/slurm_logs/%x-%j.err" --container-image="$IMAGE" --container-mounts="$WORKDIR:$WORKDIR,$ARTROOT:$ARTROOT,$ARTROOT:/workspace" --container-workdir="$WORKDIR" --container-env=PYTHONPATH --export=ALL,WORKDIR="$WORKDIR",ARTROOT="$ARTROOT",PYTHONPATH="$WORKDIR/src",TRACE_DIR="$ARTROOT/artifacts/mechanistic/judge_reasoning_trajectories_qwen3_8b_it_scout_v1",OUT_DIR="$ARTROOT/artifacts/mechanistic/judge_reasoning_analysis_qwen3_8b_it_scout_v1" cluster/lrz/judge_reasoning_analysis.sbatch
```

The analyzer uses stable pair-grouped folds across layers and trajectory
points. `decision_dynamics.csv` is derived only from out-of-fold probabilities.
Use colon-separated overrides in Slurm exports, for example
`GROUP_COLUMNS=comparison_dimension:source_dataset:validity_type:difficulty_tier:analysis_split`.
Existing trace artifacts can be re-analyzed without rerunning the model.

Run the intervention scout only after selecting a stable held-out probe
direction. The script automatically chooses the best matching successful probe
when `HIDDEN_LAYER=0` and `POINT_INDEX=-1`, but paper-facing runs should freeze
both explicitly:

```bash
cd "$WORKDIR" && PARTS="lrz-hgx-h100-94x4,lrz-dgx-a100-80x8,lrz-hgx-a100-80x4" && sbatch --parsable --partition="$PARTS" --job-name=judge-reason-intervene --gres=gpu:1 --cpus-per-task=8 --mem=160G --time=06:00:00 --chdir="$WORKDIR" --output="$ARTROOT/slurm_logs/%x-%j.out" --error="$ARTROOT/slurm_logs/%x-%j.err" --container-image="$IMAGE" --container-mounts="$WORKDIR:$WORKDIR,$ARTROOT:$ARTROOT,$ARTROOT:/workspace" --container-workdir="$WORKDIR" --container-env=PYTHONPATH,HF_HOME,TRANSFORMERS_CACHE,HF_DATASETS_CACHE,HF_TOKEN,HUGGING_FACE_HUB_TOKEN --export=ALL,WORKDIR="$WORKDIR",ARTROOT="$ARTROOT",PYTHONPATH="$WORKDIR/src",HF_HOME="$HF_HOME",TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE",HF_DATASETS_CACHE="$HF_DATASETS_CACHE",MODEL_ID="Qwen/Qwen3-8B",TRACE_DIR="$ARTROOT/artifacts/mechanistic/judge_reasoning_trajectories_qwen3_8b_it_scout_v1",ANALYSIS_DIR="$ARTROOT/artifacts/mechanistic/judge_reasoning_analysis_qwen3_8b_it_scout_v1",OUT_DIR="$ARTROOT/artifacts/mechanistic/judge_reasoning_interventions_qwen3_8b_it_scout_v1",PROBE_TARGET=final_choice,HIDDEN_LAYER=0,POINT_INDEX=-1,MAX_TRACES=64 cluster/lrz/judge_reasoning_interventions.sbatch
```

## Immediate LRZ Judge-Competence Adapter Pair Commands

The clean adapter contrast for judge-sway/invariance experiments is a fresh
BT reward adapter trained on the same backbone and same competence preference
data, with and without the D4 surface-realization invariance term:

- `jbt_pref_competence_v1`: preference stream plus zero-weight style stream
- `jbt_pref_competence_inv_v1`: identical stream mix plus invariance loss

This contrast should be trained from `google/gemma-2-9b-it`, not on top of an
existing J0 adapter. Build preference data from SHP-2 plus HelpSteer2-derived
high-utility-gap BT pairs. Build invariance groups from deterministic D4
surface counterfactuals; keep Laurito as evaluation, not training.

Obtain or refresh the public source datasets on LRZ if missing:

```bash
cd "$WORKDIR" && sbatch --parsable --partition=lrz-cpu --qos=cpu --job-name=jbt-src-data --cpus-per-task=2 --mem=48G --time=03:00:00 --chdir="$WORKDIR" --output="$ARTROOT/slurm_logs/%x-%j.out" --error="$ARTROOT/slurm_logs/%x-%j.err" --container-image="$IMAGE" --container-mounts="$WORKDIR:$WORKDIR,$ARTROOT:$ARTROOT,$ARTROOT:/workspace" --container-workdir="$WORKDIR" --container-env=PYTHONPATH,HF_HOME,TRANSFORMERS_CACHE,HF_DATASETS_CACHE,HF_TOKEN,HUGGING_FACE_HUB_TOKEN --export=ALL,WORKDIR="$WORKDIR",ARTROOT="$ARTROOT",PYTHONPATH="$WORKDIR/src",HF_HOME="$HF_HOME",TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE",HF_DATASETS_CACHE="$HF_DATASETS_CACHE" --wrap="python -m aisafety.scripts.build_pref_pairs_shp2 --out-dir '$ARTROOT/data/derived/pref_pairs_shp2' --max-train 220000 --max-val 20000 && python -m aisafety.scripts.build_helpsteer2_anchor --out-dir '$ARTROOT/data/derived/helpsteer2_anchor' --max-train 0 --max-val 0"
```

Build the merged competence preference file:

```bash
cd "$WORKDIR" && sbatch --parsable --partition=lrz-cpu --qos=cpu --job-name=jbt-pref --cpus-per-task=2 --mem=32G --time=00:45:00 --chdir="$WORKDIR" --output="$ARTROOT/slurm_logs/%x-%j.out" --error="$ARTROOT/slurm_logs/%x-%j.err" --container-image="$IMAGE" --container-mounts="$WORKDIR:$WORKDIR,$ARTROOT:$ARTROOT,$ARTROOT:/workspace" --container-workdir="$WORKDIR" --container-env=PYTHONPATH --export=ALL,WORKDIR="$WORKDIR",ARTROOT="$ARTROOT",PYTHONPATH="$WORKDIR/src" --wrap="python -m aisafety.scripts.build_judge_competence_pref_pairs --workspace-root '$WORKDIR' --shp-train-jsonl '$ARTROOT/data/derived/pref_pairs_shp2/pref_pairs_train.jsonl' --shp-val-jsonl '$ARTROOT/data/derived/pref_pairs_shp2/pref_pairs_val.jsonl' --helpsteer-train-jsonl '$ARTROOT/data/derived/helpsteer2_anchor/anchor_train.jsonl' --helpsteer-val-jsonl '$ARTROOT/data/derived/helpsteer2_anchor/anchor_val.jsonl' --out-dir '$ARTROOT/data/derived/judge_competence_pref_v1' --max-shp-train 120000 --max-shp-val 10000 --max-helpsteer-train 80000 --max-helpsteer-val 10000 --min-helpsteer-utility-gap 0.25 --max-helpsteer-pairs-per-prompt 3"
```

Build deterministic invariance counterfactuals and trainer style groups:

```bash
cd "$WORKDIR" && sbatch --parsable --partition=lrz-cpu --qos=cpu --job-name=jbt-inv-data --cpus-per-task=2 --mem=32G --time=01:00:00 --chdir="$WORKDIR" --output="$ARTROOT/slurm_logs/%x-%j.out" --error="$ARTROOT/slurm_logs/%x-%j.err" --container-image="$IMAGE" --container-mounts="$WORKDIR:$WORKDIR,$ARTROOT:$ARTROOT,$ARTROOT:/workspace" --container-workdir="$WORKDIR" --container-env=PYTHONPATH --export=ALL,WORKDIR="$WORKDIR",ARTROOT="$ARTROOT",PYTHONPATH="$WORKDIR/src" --wrap="python -m aisafety.scripts.build_d4_surface_counterfactual_pairs --workspace-root '$WORKDIR' --pair-jsonl '$ARTROOT/data/derived/d4_human_llm_alignment_pairs_strat10k_v3/pairs.jsonl' --out-dir '$ARTROOT/data/derived/d4_surface_counterfactual_pairs_invariance_train_v1' --axes structured_assistant_packaging,answer_likeness_packaging --max-pairs 0 && python -m aisafety.scripts.build_d4_invariance_style_groups --workspace-root '$WORKDIR' --counterfactual-jsonl '$ARTROOT/data/derived/d4_surface_counterfactual_pairs_invariance_train_v1/counterfactuals.jsonl' --out-dir '$ARTROOT/data/derived/d4_invariance_style_groups_v1' --axes structured_assistant_packaging,answer_likeness_packaging --group-mode per_axis --include-objective-bullets --max-variants-per-group 4 --val-frac 0.1"
```

Train the adapter pair after both data-build jobs complete:

```bash
cd "$WORKDIR" && PARTS="lrz-hgx-h100-94x4,lrz-dgx-a100-80x8,lrz-hgx-a100-80x4" && sbatch --parsable --partition="$PARTS" --job-name=jbt-pref --gres=gpu:1 --cpus-per-task=8 --mem=160G --time=08:00:00 --chdir="$WORKDIR" --output="$ARTROOT/slurm_logs/%x-%j.out" --error="$ARTROOT/slurm_logs/%x-%j.err" --container-image="$IMAGE" --container-mounts="$WORKDIR:$WORKDIR,$ARTROOT:$ARTROOT,$ARTROOT:/workspace" --container-workdir="$WORKDIR" --container-env=PYTHONPATH,HF_HOME,TRANSFORMERS_CACHE,HF_DATASETS_CACHE,HF_TOKEN,HUGGING_FACE_HUB_TOKEN --export=ALL,WORKDIR="$WORKDIR",ARTROOT="$ARTROOT",PYTHONPATH="$WORKDIR/src",HF_HOME="$HF_HOME",TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE",HF_DATASETS_CACHE="$HF_DATASETS_CACHE" --wrap="python -m aisafety.scripts.run_experiment_config --config '$WORKDIR/configs/experiments/jbt_pref_competence_v1.json' --workspace-root '$ARTROOT'"
```

```bash
cd "$WORKDIR" && PARTS="lrz-hgx-h100-94x4,lrz-dgx-a100-80x8,lrz-hgx-a100-80x4" && sbatch --parsable --partition="$PARTS" --job-name=jbt-pref-inv --gres=gpu:1 --cpus-per-task=8 --mem=160G --time=08:00:00 --chdir="$WORKDIR" --output="$ARTROOT/slurm_logs/%x-%j.out" --error="$ARTROOT/slurm_logs/%x-%j.err" --container-image="$IMAGE" --container-mounts="$WORKDIR:$WORKDIR,$ARTROOT:$ARTROOT,$ARTROOT:/workspace" --container-workdir="$WORKDIR" --container-env=PYTHONPATH,HF_HOME,TRANSFORMERS_CACHE,HF_DATASETS_CACHE,HF_TOKEN,HUGGING_FACE_HUB_TOKEN --export=ALL,WORKDIR="$WORKDIR",ARTROOT="$ARTROOT",PYTHONPATH="$WORKDIR/src",HF_HOME="$HF_HOME",TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE",HF_DATASETS_CACHE="$HF_DATASETS_CACHE" --wrap="python -m aisafety.scripts.run_experiment_config --config '$WORKDIR/configs/experiments/jbt_pref_competence_inv_v1.json' --workspace-root '$ARTROOT'"
```

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

Surface counterfactual auditing supports:

- deterministic cue-increase/cue-decrease text variants move J0 rewards under
  approximate content preservation
- reward deltas are measured with length/source stratification
- SAE bundle activation deltas move in the expected direction

Feature-card inspection supports:

- a feature is an interpretable candidate cue feature after top examples and
  close negatives are checked

Bundle graph construction supports:

- atom features form a bundle-level hypothesis

Readout-space nulling supports:

- a pooled-state surface-cue subspace can reduce counterfactual cue sensitivity
- human-vs-LLM margin changes and general preference retention are measured

Only intervention tests support:

- causal influence on judge behavior
- repair or mitigation claims

## Immediate LRZ Feature-Perturbation Scout

The first intervention scout perturbs the frozen
`formal_institutional_packaging` bundle by damping the eligible SAE features,
then compares J0 margin changes on high-vs-low bundle-occurrence prompt pairs
against layer-matched random-control feature damping.

```bash
cd "$WORKDIR" && PARTS="lrz-hgx-h100-94x4,lrz-dgx-a100-80x8,lrz-hgx-a100-80x4" && sbatch --parsable --partition="$PARTS" --job-name=d4-formal-pert --gres=gpu:1 --cpus-per-task=8 --mem=160G --time=06:00:00 --chdir="$WORKDIR" --output="$ARTROOT/slurm_logs/%x-%j.out" --error="$ARTROOT/slurm_logs/%x-%j.err" --container-image="$IMAGE" --container-mounts="$WORKDIR:$WORKDIR,$ARTROOT:$ARTROOT,$ARTROOT:/workspace" --container-workdir="$WORKDIR" --container-env=PYTHONPATH,HF_HOME,TRANSFORMERS_CACHE,HF_DATASETS_CACHE,HF_TOKEN,HUGGING_FACE_HUB_TOKEN --export=ALL,WORKDIR="$WORKDIR",ARTROOT="$ARTROOT",PYTHONPATH="$WORKDIR/src",HF_HOME="$HF_HOME",TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE",HF_DATASETS_CACHE="$HF_DATASETS_CACHE",PAIR_JSONL="$ARTROOT/data/derived/d4_human_llm_alignment_pairs_strat10k_v3/pairs.jsonl",BUNDLE_REGISTRY_DIR="$ARTROOT/artifacts/mechanistic/d4_j0_bundle_candidate_registry_v1",REWARD_RUN_DIR="$ARTROOT/artifacts/reward/j0_anchor_v1_h100compact",OUT_DIR="$ARTROOT/artifacts/mechanistic/d4_j0_formal_bundle_feature_perturbation_scout_v1",BUNDLE_ID=formal_institutional_packaging,MAX_PAIRS=1200,DAMPING_STRENGTH=0.5,HIGH_LOW_FRAC=0.25,SCORE_BATCH_SIZE=4,SAE_BATCH_SIZE=4,SAE_TOKEN_CHUNK_SIZE=1024,MAX_LENGTH=512,RANDOM_CONTROL_RANK=1 cluster/lrz/d4_feature_perturbation.sbatch
```

## Immediate LRZ Surface-Counterfactual And Readout-Nulling Commands

The active surface-cue counterfactual axes are
`structured_assistant_packaging`, `answer_likeness_packaging`,
`formal_institutional_packaging`, and `benefit_value_framing`. Prefer
axis-specific output directories for paper-facing comparisons. The current
high-value sequence is to rerun `structured_assistant_packaging` with the
broadened bidirectional paragraphize/listify transform, then run
`answer_likeness_packaging` separately to test direct-answer scaffolding beyond
list formatting.

Build deterministic surface-cue counterfactuals on CPU:

```bash
cd "$WORKDIR" && sbatch --parsable --partition=lrz-cpu --qos=cpu --job-name=d4-surf-cf --cpus-per-task=2 --mem=16G --time=00:30:00 --chdir="$WORKDIR" --output="$ARTROOT/slurm_logs/%x-%j.out" --error="$ARTROOT/slurm_logs/%x-%j.err" --container-image="$IMAGE" --container-mounts="$WORKDIR:$WORKDIR,$ARTROOT:$ARTROOT,$ARTROOT:/workspace" --container-workdir="$WORKDIR" --container-env=PYTHONPATH --export=ALL,WORKDIR="$WORKDIR",ARTROOT="$ARTROOT",PYTHONPATH="$WORKDIR/src",PAIR_JSONL="$ARTROOT/data/derived/d4_human_llm_alignment_pairs_strat10k_v3/pairs.jsonl",OUT_DIR="$ARTROOT/data/derived/d4_surface_counterfactual_pairs_v1",MAX_PAIRS=0 cluster/lrz/d4_surface_counterfactual_pairs.sbatch
```

Audit J0 reward deltas and matching SAE bundle activation deltas:

```bash
cd "$WORKDIR" && PARTS="lrz-hgx-h100-94x4,lrz-dgx-a100-80x8,lrz-hgx-a100-80x4" && sbatch --parsable --partition="$PARTS" --job-name=d4-surf-audit --gres=gpu:1 --cpus-per-task=8 --mem=160G --time=08:00:00 --chdir="$WORKDIR" --output="$ARTROOT/slurm_logs/%x-%j.out" --error="$ARTROOT/slurm_logs/%x-%j.err" --container-image="$IMAGE" --container-mounts="$WORKDIR:$WORKDIR,$ARTROOT:$ARTROOT,$ARTROOT:/workspace" --container-workdir="$WORKDIR" --container-env=PYTHONPATH,HF_HOME,TRANSFORMERS_CACHE,HF_DATASETS_CACHE,HF_TOKEN,HUGGING_FACE_HUB_TOKEN --export=ALL,WORKDIR="$WORKDIR",ARTROOT="$ARTROOT",PYTHONPATH="$WORKDIR/src",HF_HOME="$HF_HOME",TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE",HF_DATASETS_CACHE="$HF_DATASETS_CACHE",COUNTERFACTUAL_JSONL="$ARTROOT/data/derived/d4_surface_counterfactual_pairs_v1/counterfactuals.jsonl",BUNDLE_REGISTRY_DIR="$ARTROOT/artifacts/mechanistic/d4_j0_bundle_candidate_registry_v1",REWARD_RUN_DIR="$ARTROOT/artifacts/reward/j0_anchor_v1_h100compact",OUT_DIR="$ARTROOT/artifacts/mechanistic/d4_j0_surface_counterfactual_audit_v1",MAX_COUNTERFACTUALS=0,SCORE_BATCH_SIZE=4,SAE_BATCH_SIZE=4,SAE_TOKEN_CHUNK_SIZE=1024,MAX_LENGTH=512 cluster/lrz/d4_surface_counterfactual_audit.sbatch
```

Fit pooled-state surface directions and evaluate readout-space nulling:

```bash
cd "$WORKDIR" && PARTS="lrz-hgx-h100-94x4,lrz-dgx-a100-80x8,lrz-hgx-a100-80x4" && sbatch --parsable --partition="$PARTS" --job-name=d4-null --gres=gpu:1 --cpus-per-task=8 --mem=160G --time=08:00:00 --chdir="$WORKDIR" --output="$ARTROOT/slurm_logs/%x-%j.out" --error="$ARTROOT/slurm_logs/%x-%j.err" --container-image="$IMAGE" --container-mounts="$WORKDIR:$WORKDIR,$ARTROOT:$ARTROOT,$ARTROOT:/workspace" --container-workdir="$WORKDIR" --container-env=PYTHONPATH,HF_HOME,TRANSFORMERS_CACHE,HF_DATASETS_CACHE,HF_TOKEN,HUGGING_FACE_HUB_TOKEN --export=ALL,WORKDIR="$WORKDIR",ARTROOT="$ARTROOT",PYTHONPATH="$WORKDIR/src",HF_HOME="$HF_HOME",TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE",HF_DATASETS_CACHE="$HF_DATASETS_CACHE",COUNTERFACTUAL_JSONL="$ARTROOT/data/derived/d4_surface_counterfactual_pairs_v1/counterfactuals.jsonl",PAIR_JSONL="$ARTROOT/data/derived/d4_human_llm_alignment_pairs_strat10k_v3/pairs.jsonl",PREF_VAL_JSONL="$ARTROOT/data/derived/pref_pairs_shp2/pref_pairs_val.jsonl",REWARD_RUN_DIR="$ARTROOT/artifacts/reward/j0_anchor_v1_h100compact",OUT_DIR="$ARTROOT/artifacts/mechanistic/d4_j0_readout_surface_nulling_v1",MAX_COUNTERFACTUALS=0,MAX_PAIRS=3000,MAX_PREF_PAIRS=1000,FIT_FRAC=0.5,MIN_DIRECTION_ROWS=20,SCORE_BATCH_SIZE=4,ENCODE_BATCH_SIZE=4,MAX_LENGTH=512 cluster/lrz/d4_readout_surface_nulling.sbatch
```

## Immediate LRZ Human-vs-LLM Training-Stage Contrast

This diagnostic tests whether base-to-instruction post-training shifts
LLM-minus-human preference margins on the same broad paired corpus. It is not a
feature-localization claim and does not identify RLHF separately from SFT/DPO
unless the compared model family exposes matched intermediate checkpoints.

Recommended first-pass Tulu/Llama ladder:

- `llama31_base`: `meta-llama/Llama-3.1-8B`, plain forced-choice prompt
- `tulu3_sft`: `allenai/Llama-3.1-Tulu-3-8B-SFT`, chat-template forced choice
- `tulu3_dpo`: `allenai/Llama-3.1-Tulu-3-8B-DPO`, chat-template forced choice
- `tulu3_final`: `allenai/Llama-3.1-Tulu-3-8B`, chat-template forced choice
- `llama31_instruct`: `meta-llama/Llama-3.1-8B-Instruct`, chat-template forced
  choice, if gated access is available

Use the matrix submitter from the LRZ login node for the scout. It queues the
pair builder, all scoring jobs with dependencies, and the summary job. The
default includes the official Llama instruct comparison and response-likelihood
controls for base/SFT/DPO; set `INCLUDE_RESPONSE_LIKELIHOOD=0` to run only the
forced-choice ladder.

```bash
cd "$WORKDIR" && RUN_TAG=tulu_stage_scout_v1 PAIR_JSONL="$ARTROOT/data/derived/d4_human_llm_alignment_pairs_strat10k_v3/pairs.jsonl" MAX_SOURCE_PAIRS=1000 INCLUDE_META_INSTRUCT=1 INCLUDE_RESPONSE_LIKELIHOOD=1 bash cluster/lrz/submit_d4_human_llm_tulu_stage_matrix.sh
```

For the full broad pair file after the scout validates container imports and
nonempty outputs, rerun with `MAX_SOURCE_PAIRS=0` and a fresh `RUN_TAG`.

Temporary local-cluster fallback when LRZ is unavailable:

- use the A100s first: GPU `1` and `7` are 80GB, GPU `0` is 40GB
- avoid V100 and RTX 8000 for the default CausalLM path because it loads models
  in bf16
- keep Hugging Face caches on large storage
- export an HF token with accepted Llama 3.1 access before running Meta stages

Suggested local setup:

```bash
cd /path/to/AISafety
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch
python -m pip install -e .
export WORKDIR=$PWD
export ARTROOT=/path/to/large/debias-artifacts
export HF_HOME=$ARTROOT/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_TOKEN=<token-with-llama-access>
```

Run the local scout directly, without Slurm:

```bash
cd "$WORKDIR" && RUN_TAG=tulu_stage_scout_local_v1 PAIR_JSONL="$ARTROOT/data/derived/d4_human_llm_alignment_pairs_strat10k_v3/pairs.jsonl" MAX_SOURCE_PAIRS=1000 GPU_A=1 GPU_B=7 GPU_C=0 INCLUDE_META_INSTRUCT=1 INCLUDE_RESPONSE_LIKELIHOOD=0 bash cluster/local/run_d4_human_llm_tulu_stage_matrix.sh
```

If the forced-choice ladder completes cleanly, add the likelihood controls:

```bash
cd "$WORKDIR" && RUN_TAG=tulu_stage_likelihood_local_v1 PAIR_JSONL="$ARTROOT/data/derived/d4_human_llm_alignment_pairs_strat10k_v3/pairs.jsonl" MAX_SOURCE_PAIRS=1000 GPU_A=1 GPU_B=7 GPU_C=0 INCLUDE_META_INSTRUCT=1 INCLUDE_RESPONSE_LIKELIHOOD=1 bash cluster/local/run_d4_human_llm_tulu_stage_matrix.sh
```

Build order-swapped human-vs-LLM BT rows from the broad pair file:

```bash
cd "$WORKDIR" && sbatch --parsable --partition=lrz-cpu --qos=cpu --job-name=d4-hllm-stage-pairs --cpus-per-task=2 --mem=16G --time=00:20:00 --chdir="$WORKDIR" --output="$ARTROOT/slurm_logs/%x-%j.out" --error="$ARTROOT/slurm_logs/%x-%j.err" --container-image="$IMAGE" --container-mounts="$WORKDIR:$WORKDIR,$ARTROOT:$ARTROOT,$ARTROOT:/workspace" --container-workdir="$WORKDIR" --container-env=PYTHONPATH --export=ALL,WORKDIR="$WORKDIR",ARTROOT="$ARTROOT",PYTHONPATH="$WORKDIR/src",PAIR_JSONL="$ARTROOT/data/derived/d4_human_llm_alignment_pairs_strat10k_v3/pairs.jsonl",OUT_DIR="$ARTROOT/data/derived/d4_human_llm_stage_contrast_pairs_strat10k_v1",MAX_PAIRS=0,INCLUDE_ORDER_SWAPS=1 cluster/lrz/d4_human_llm_stage_contrast_pairs.sbatch
```

Score Gemma base and instruction checkpoints with order-debiased forced-choice
logprobs. Use `PROMPT_STYLE=plain` for base and run both `plain` and
`chat_template` for instruction-tuned checkpoints to separate checkpoint shift
from chat-template elicitation:

```bash
cd "$WORKDIR" && PARTS="lrz-hgx-h100-94x4,lrz-dgx-a100-80x8,lrz-hgx-a100-80x4" && sbatch --parsable --partition="$PARTS" --job-name=hllm-gemma2-base --gres=gpu:1 --cpus-per-task=8 --mem=160G --time=04:00:00 --chdir="$WORKDIR" --output="$ARTROOT/slurm_logs/%x-%j.out" --error="$ARTROOT/slurm_logs/%x-%j.err" --container-image="$IMAGE" --container-mounts="$WORKDIR:$WORKDIR,$ARTROOT:$ARTROOT,$ARTROOT:/workspace" --container-workdir="$WORKDIR" --container-env=PYTHONPATH,HF_HOME,TRANSFORMERS_CACHE,HF_DATASETS_CACHE,HF_TOKEN,HUGGING_FACE_HUB_TOKEN --export=ALL,WORKDIR="$WORKDIR",ARTROOT="$ARTROOT",PYTHONPATH="$WORKDIR/src",HF_HOME="$HF_HOME",TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE",HF_DATASETS_CACHE="$HF_DATASETS_CACHE",BT_PAIRS_JSONL="$ARTROOT/data/derived/d4_human_llm_stage_contrast_pairs_strat10k_v1/bt_pairs.jsonl",SCORING_MODE=forced_choice,STAGE_LABEL=gemma2_9b_base_forced_plain,MODEL_ID=google/gemma-2-9b,PROMPT_STYLE=plain,OUT_DIR="$ARTROOT/artifacts/mechanistic/d4_hllm_stage_gemma2_9b_base_forced_plain_v1",MAX_PAIRS=0,SCORE_BATCH_SIZE=4,MAX_LENGTH=2048 cluster/lrz/d4_human_llm_stage_contrast.sbatch
```

```bash
cd "$WORKDIR" && PARTS="lrz-hgx-h100-94x4,lrz-dgx-a100-80x8,lrz-hgx-a100-80x4" && sbatch --parsable --partition="$PARTS" --job-name=hllm-gemma2-it --gres=gpu:1 --cpus-per-task=8 --mem=160G --time=04:00:00 --chdir="$WORKDIR" --output="$ARTROOT/slurm_logs/%x-%j.out" --error="$ARTROOT/slurm_logs/%x-%j.err" --container-image="$IMAGE" --container-mounts="$WORKDIR:$WORKDIR,$ARTROOT:$ARTROOT,$ARTROOT:/workspace" --container-workdir="$WORKDIR" --container-env=PYTHONPATH,HF_HOME,TRANSFORMERS_CACHE,HF_DATASETS_CACHE,HF_TOKEN,HUGGING_FACE_HUB_TOKEN --export=ALL,WORKDIR="$WORKDIR",ARTROOT="$ARTROOT",PYTHONPATH="$WORKDIR/src",HF_HOME="$HF_HOME",TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE",HF_DATASETS_CACHE="$HF_DATASETS_CACHE",BT_PAIRS_JSONL="$ARTROOT/data/derived/d4_human_llm_stage_contrast_pairs_strat10k_v1/bt_pairs.jsonl",SCORING_MODE=forced_choice,STAGE_LABEL=gemma2_9b_it_forced_chat,MODEL_ID=google/gemma-2-9b-it,PROMPT_STYLE=chat_template,OUT_DIR="$ARTROOT/artifacts/mechanistic/d4_hllm_stage_gemma2_9b_it_forced_chat_v1",MAX_PAIRS=0,SCORE_BATCH_SIZE=4,MAX_LENGTH=2048 cluster/lrz/d4_human_llm_stage_contrast.sbatch
```

Run response-likelihood as a base-model-friendly familiarity control. This is
not a judge-preference measure; it asks whether the model assigns higher
conditional likelihood to the LLM response than the human response:

```bash
cd "$WORKDIR" && PARTS="lrz-hgx-h100-94x4,lrz-dgx-a100-80x8,lrz-hgx-a100-80x4" && sbatch --parsable --partition="$PARTS" --job-name=hllm-gemma2-base-like --gres=gpu:1 --cpus-per-task=8 --mem=160G --time=04:00:00 --chdir="$WORKDIR" --output="$ARTROOT/slurm_logs/%x-%j.out" --error="$ARTROOT/slurm_logs/%x-%j.err" --container-image="$IMAGE" --container-mounts="$WORKDIR:$WORKDIR,$ARTROOT:$ARTROOT,$ARTROOT:/workspace" --container-workdir="$WORKDIR" --container-env=PYTHONPATH,HF_HOME,TRANSFORMERS_CACHE,HF_DATASETS_CACHE,HF_TOKEN,HUGGING_FACE_HUB_TOKEN --export=ALL,WORKDIR="$WORKDIR",ARTROOT="$ARTROOT",PYTHONPATH="$WORKDIR/src",HF_HOME="$HF_HOME",TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE",HF_DATASETS_CACHE="$HF_DATASETS_CACHE",BT_PAIRS_JSONL="$ARTROOT/data/derived/d4_human_llm_stage_contrast_pairs_strat10k_v1/bt_pairs.jsonl",SCORING_MODE=response_likelihood,STAGE_LABEL=gemma2_9b_base_response_likelihood,MODEL_ID=google/gemma-2-9b,PROMPT_STYLE=plain,OUT_DIR="$ARTROOT/artifacts/mechanistic/d4_hllm_stage_gemma2_9b_base_response_likelihood_v1",MAX_PAIRS=0,SCORE_BATCH_SIZE=2,MAX_LENGTH=2048 cluster/lrz/d4_human_llm_stage_contrast.sbatch
```

Score J0 or a fresh reward adapter with independent scalar scores:

```bash
cd "$WORKDIR" && PARTS="lrz-hgx-h100-94x4,lrz-dgx-a100-80x8,lrz-hgx-a100-80x4" && sbatch --parsable --partition="$PARTS" --job-name=hllm-j0 --gres=gpu:1 --cpus-per-task=8 --mem=160G --time=04:00:00 --chdir="$WORKDIR" --output="$ARTROOT/slurm_logs/%x-%j.out" --error="$ARTROOT/slurm_logs/%x-%j.err" --container-image="$IMAGE" --container-mounts="$WORKDIR:$WORKDIR,$ARTROOT:$ARTROOT,$ARTROOT:/workspace" --container-workdir="$WORKDIR" --container-env=PYTHONPATH,HF_HOME,TRANSFORMERS_CACHE,HF_DATASETS_CACHE,HF_TOKEN,HUGGING_FACE_HUB_TOKEN --export=ALL,WORKDIR="$WORKDIR",ARTROOT="$ARTROOT",PYTHONPATH="$WORKDIR/src",HF_HOME="$HF_HOME",TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE",HF_DATASETS_CACHE="$HF_DATASETS_CACHE",BT_PAIRS_JSONL="$ARTROOT/data/derived/d4_human_llm_stage_contrast_pairs_strat10k_v1/bt_pairs.jsonl",SCORING_MODE=reward_scalar,STAGE_LABEL=j0_reward_scalar_prompt_response,MODEL_ID=google/gemma-2-9b-it,REWARD_RUN_DIR="$ARTROOT/artifacts/reward/j0_anchor_v1_h100compact",REWARD_INPUT_FORMAT=prompt_response,OUT_DIR="$ARTROOT/artifacts/mechanistic/d4_hllm_stage_j0_reward_scalar_prompt_response_v1",MAX_PAIRS=0,SCORE_BATCH_SIZE=4,MAX_LENGTH=512 cluster/lrz/d4_human_llm_stage_contrast.sbatch
```

Summarize paired stage deltas on CPU. `RUNS` and `CONTRASTS` use
colon-separated lists because commas in Slurm `--export` split environment
variables:

```bash
cd "$WORKDIR" && sbatch --parsable --partition=lrz-cpu --qos=cpu --job-name=hllm-stage-sum --cpus-per-task=2 --mem=8G --time=00:15:00 --chdir="$WORKDIR" --output="$ARTROOT/slurm_logs/%x-%j.out" --error="$ARTROOT/slurm_logs/%x-%j.err" --container-image="$IMAGE" --container-mounts="$WORKDIR:$WORKDIR,$ARTROOT:$ARTROOT,$ARTROOT:/workspace" --container-workdir="$WORKDIR" --container-env=PYTHONPATH --export=ALL,WORKDIR="$WORKDIR",ARTROOT="$ARTROOT",PYTHONPATH="$WORKDIR/src",RUNS="gemma2_base=$ARTROOT/artifacts/mechanistic/d4_hllm_stage_gemma2_9b_base_forced_plain_v1:gemma2_it=$ARTROOT/artifacts/mechanistic/d4_hllm_stage_gemma2_9b_it_forced_chat_v1:j0=$ARTROOT/artifacts/mechanistic/d4_hllm_stage_j0_reward_scalar_prompt_response_v1",CONTRASTS="gemma2_it_minus_base=gemma2_it-gemma2_base:j0_minus_it=j0-gemma2_it",OUT_DIR="$ARTROOT/artifacts/mechanistic/d4_human_llm_stage_contrast_summary_gemma2_j0_v1" cluster/lrz/d4_human_llm_stage_contrast_summary.sbatch
```

Build Bradley-Terry stage-contrast rows from a deterministic counterfactual
JSONL before scoring base, instruction-tuned, or reward-stage models:

```bash
cd "$WORKDIR" && sbatch --parsable --partition=lrz-cpu --qos=cpu --job-name=d4-bt-pairs --cpus-per-task=2 --mem=16G --time=00:20:00 --chdir="$WORKDIR" --output="$ARTROOT/slurm_logs/%x-%j.out" --error="$ARTROOT/slurm_logs/%x-%j.err" --container-image="$IMAGE" --container-mounts="$WORKDIR:$WORKDIR,$ARTROOT:$ARTROOT,$ARTROOT:/workspace" --container-workdir="$WORKDIR" --container-env=PYTHONPATH --export=ALL,WORKDIR="$WORKDIR",ARTROOT="$ARTROOT",PYTHONPATH="$WORKDIR/src",COUNTERFACTUAL_JSONL="$ARTROOT/data/derived/d4_surface_counterfactual_pairs_answer_likeness_v1/counterfactuals.jsonl",OUT_DIR="$ARTROOT/data/derived/d4_bt_stage_contrast_pairs_answer_likeness_v1",AXES=answer_likeness_packaging,MAX_COUNTERFACTUALS=0,INCLUDE_ORDER_SWAPS=1 cluster/lrz/d4_bt_stage_contrast_pairs.sbatch
```

Score a BT stage on GPU. Use `STAGE=base_lm` with
`MODEL_ID=google/gemma-2-9b` and `PROMPT_STYLE=plain`; use `STAGE=it_lm` with
`MODEL_ID=google/gemma-2-9b-it` and `PROMPT_STYLE=chat_template`; use
`STAGE=reward_j0` with the J0 reward directory and `PROMPT_STYLE=plain`.

```bash
cd "$WORKDIR" && PARTS="lrz-hgx-h100-94x4,lrz-dgx-a100-80x8,lrz-hgx-a100-80x4" && sbatch --parsable --partition="$PARTS" --job-name=d4-bt-stage --gres=gpu:1 --cpus-per-task=8 --mem=160G --time=04:00:00 --chdir="$WORKDIR" --output="$ARTROOT/slurm_logs/%x-%j.out" --error="$ARTROOT/slurm_logs/%x-%j.err" --container-image="$IMAGE" --container-mounts="$WORKDIR:$WORKDIR,$ARTROOT:$ARTROOT,$ARTROOT:/workspace" --container-workdir="$WORKDIR" --container-env=PYTHONPATH,HF_HOME,TRANSFORMERS_CACHE,HF_DATASETS_CACHE,HF_TOKEN,HUGGING_FACE_HUB_TOKEN --export=ALL,WORKDIR="$WORKDIR",ARTROOT="$ARTROOT",PYTHONPATH="$WORKDIR/src",HF_HOME="$HF_HOME",TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE",HF_DATASETS_CACHE="$HF_DATASETS_CACHE",BT_PAIRS_JSONL="$ARTROOT/data/derived/d4_bt_stage_contrast_pairs_answer_likeness_v1/bt_pairs.jsonl",STAGE=it_lm,MODEL_ID=google/gemma-2-9b-it,PROMPT_STYLE=chat_template,REWARD_RUN_DIR="$ARTROOT/artifacts/reward/j0_anchor_v1_h100compact",OUT_DIR="$ARTROOT/artifacts/mechanistic/d4_bt_stage_contrast_answer_likeness_it_v1",MAX_PAIRS=0,SCORE_BATCH_SIZE=4,MAX_LENGTH=2048 cluster/lrz/d4_bt_stage_contrast.sbatch
```

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
- `data/derived/d4_human_llm_stage_contrast_pairs_v1`
- `data/derived/d4_surface_counterfactual_pairs_v1`
- `artifacts/mechanistic/d4_j0_human_llm_candidate_alignment_v1`
- `artifacts/mechanistic/d4_hllm_stage_<model>_<mode>_v1`
- `artifacts/mechanistic/d4_hllm_stage_tulu_stage_scout_v1`
- `artifacts/mechanistic/d4_human_llm_stage_contrast_summary_<models>_v1`
- `artifacts/mechanistic/d4_j0_bundle_graph_v1`
- `artifacts/mechanistic/d4_j0_intervention_scout_v1`
- `artifacts/mechanistic/d4_j0_surface_counterfactual_audit_v1`
- `artifacts/mechanistic/d4_j0_readout_surface_nulling_v1`

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
11. Treat SAE features as detectors unless text counterfactual or readout-space
    interventions show reward impact and retention.
