# Deliberation Under Criterion Uncertainty

This directory is the persistent research track for the question:

> When does inference-time deliberation improve LLM judgment, and when does
> the absence of a verifiable stopping criterion produce overthinking,
> instability, or rationalization?

The track is adjacent to the D4 atom-to-bundle work. It reuses the
judge-reasoning data and activation infrastructure, but it has a distinct
claim contract and experiment order.

Files:

- `EXPERIMENT_CHARTA.md`: experiment sequence, claim gates, estimands, and
  stopping rules.
- `paper_story.tex`: loose-prose manuscript that tracks the argument and
  evidence status in parallel with experiments.
- `references.bib`: focused bibliography for this track.

The first claim-critical run is the Qwen3-8B token-budget sweep. The activation
analysis follows only after the behavioral budget curves and criterion
manipulations have been completed. Larger-model and cross-family replications
come after the 8B design is frozen.

The source progression is materialized with:

```bash
python -m aisafety.scripts.build_judge_deliberation_source_pack --help
python -m aisafety.scripts.build_judge_reasoning_suite --help
```

The behavioral budget sweep and its analysis are:

```bash
python -m aisafety.scripts.run_judge_reasoning_budget_sweep --help
python -m aisafety.scripts.analyze_judge_reasoning_budget_sweep --help
python -m aisafety.scripts.build_helpsteer2_matched_criterion_suite --help
python -m aisafety.scripts.analyze_helpsteer2_matched_criterion --help
python -m aisafety.scripts.read_helpsteer2_matched_criterion --help
python -m aisafety.scripts.build_helpsteer2_criterion_switch_suite --help
python -m aisafety.scripts.build_helpsteer2_criterion_confirmation --help
python -m aisafety.scripts.run_judge_criterion_switch_behavior --help
python -m aisafety.scripts.analyze_judge_criterion_switch_behavior --help
```

The later activation pass uses:

```bash
python -m aisafety.scripts.run_judge_reasoning_trajectories --help
python -m aisafety.scripts.analyze_judge_reasoning_fixed_decoders --help
python -m aisafety.scripts.run_judge_criterion_switch_activations --help
python -m aisafety.scripts.analyze_judge_criterion_switch_decoders --help
python -m aisafety.scripts.run_judge_criterion_switch_patching --help
```

LRZ scout submission:

```bash
cd "$WORKDIR" && RUN_TAG=judge_deliberation_qwen3_8b_budget_scout_v1 MAX_PAIRS_PER_DATASET=30 BRANCHES_PER_COMPARISON=3 bash cluster/lrz/submit_judge_deliberation_qwen3_8b_budget.sh
```

ipe-monster scout:

```bash
cd "$WORKDIR" && RUN_TAG=judge_deliberation_qwen3_8b_budget_scout_v1 GPU=7 MAX_PAIRS_PER_DATASET=30 BRANCHES_PER_COMPARISON=3 bash cluster/local/run_judge_deliberation_qwen3_8b_budget.sh
```

Targeted incremental scouts can set `INCLUDE_DATASETS` to a comma-separated
allowlist of dataset IDs from the progression config. Each targeted run must
use a distinct `RUN_TAG`.

The first within-pair criterion evidence collector uses the same HelpSteer2
response pairs under overall, correctness, helpfulness, coherence, and fixed
weighted rules. It permits `C` for tied or underdetermined decisions, keeps
both response orders together, and runs two balanced shards on GPUs 0 and 1:

```bash
cd "$WORKDIR" && RUN_TAG=helpsteer2_matched_criterion_qwen3_8b_scout_v1 GPU_0=0 GPU_1=1 MAX_PAIRS_PER_STRATUM=8 BRANCHES_PER_COMPARISON=2 BUDGET_TOKENS=0,128,512,1024 bash cluster/local/run_helpsteer2_matched_criterion_qwen3_8b.sh
```

This collector estimates compliance with an explicit decision rule on fixed
texts. It does not establish a universal ground-truth ranking for evaluative
answers.

The held-out staged criterion-switch scout reuses the same first-stage
reasoning across stable, reminder, switch, placebo, and delayed-rule
conditions. It uses the larger HelpSteer2 training split because the prior
matched-criterion scout used validation and validation contains too few
choice-to-choice criterion conflicts. It also excludes any prior response
pairs by content signature when that artifact is available:

```bash
cd "$WORKDIR" && RUN_TAG=judge_criterion_switch_qwen3_8b_scout_v1 SOURCE_SPLIT=train GPU_0=0 GPU_1=1 MAX_PAIRS_PER_TRANSITION=8 BRANCHES_PER_EPISODE=3 bash cluster/local/run_judge_criterion_switch_qwen3_8b_behavior.sh
```

Build the locked 24-pair operationalization confirmation and its 96 blinded
human-audit prompts without loading the model:

```bash
cd "$WORKDIR" && ARTROOT="$ARTROOT" BUILD_ONLY=1 bash cluster/local/run_judge_criterion_confirmation_qwen3_8b.sh
```

Run the frozen 408-trace confirmation on GPU 7:

```bash
cd "$WORKDIR" && ARTROOT="$ARTROOT" RUN_ONLY=1 GPU=7 bash cluster/local/run_judge_criterion_confirmation_qwen3_8b.sh
```

After inspecting and freezing the behavioral artifact, capture exact-prefix
activations and fit pair-held-out multiclass decoders:

```bash
cd "$WORKDIR" && RUN_TAG=judge_criterion_switch_qwen3_8b_scout_v1 GPU_0=0 GPU_1=1 bash cluster/local/run_judge_criterion_switch_qwen3_8b_mech.sh
```

Set `RUN_PATCHING=1` only after the decoder selection is accepted. The
patching stage applies within-pair switch-minus-reminder differences at the
selected criterion layer and includes negative, shuffled-pair, and
same-target controls.

When the scout and non-overlapping extension are both complete, capture only
the reminder, switch, and placebo conditions and fit a combined decoder:

```bash
cd "$WORKDIR" && RUN_TAG=judge_criterion_switch_qwen3_8b_combined_v1 GPU_0=1 GPU_1=7 bash cluster/local/run_judge_criterion_switch_qwen3_8b_combined_mech.sh
```

After the scout validates the design, the confirmation run is:

```bash
cd "$WORKDIR" && RUN_TAG=judge_deliberation_qwen3_8b_budget_confirm_v1 MAX_PAIRS_PER_DATASET=60 BRANCHES_PER_COMPARISON=5 bash cluster/lrz/submit_judge_deliberation_qwen3_8b_budget.sh
```

The corresponding ipe-monster confirmation command is:

```bash
cd "$WORKDIR" && RUN_TAG=judge_deliberation_qwen3_8b_budget_confirm_v1 GPU=7 MAX_PAIRS_PER_DATASET=60 BRANCHES_PER_COMPARISON=5 bash cluster/local/run_judge_deliberation_qwen3_8b_budget.sh
```

Submit activation capture only after freezing the confirmation suite:

```bash
cd "$WORKDIR" && SUITE_DIR="$ARTROOT/data/derived/judge_deliberation_suite_judge_deliberation_qwen3_8b_budget_confirm_v1" RUN_TAG=judge_deliberation_qwen3_8b_activations_v1 bash cluster/lrz/submit_judge_deliberation_qwen3_8b_activations.sh
```

On ipe-monster:

```bash
cd "$WORKDIR" && SUITE_DIR="$ARTROOT/data/derived/judge_deliberation_suite_judge_deliberation_qwen3_8b_budget_confirm_v1" RUN_TAG=judge_deliberation_qwen3_8b_activations_v1 GPU=7 bash cluster/local/run_judge_deliberation_qwen3_8b_activations.sh
```
