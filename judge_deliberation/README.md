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
```

The later activation pass uses:

```bash
python -m aisafety.scripts.run_judge_reasoning_trajectories --help
python -m aisafety.scripts.analyze_judge_reasoning_fixed_decoders --help
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
