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
python -m aisafety.scripts.analyze_judge_criterion_confirmation --help
python -m aisafety.scripts.read_judge_criterion_confirmation --help
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

Analyze the completed run. The analyzer automatically uses
`audit_prompts_for_judging_completed.csv` when it is present:

```bash
cd "$WORKDIR" && python -m aisafety.scripts.analyze_judge_criterion_confirmation --workspace-root "$WORKDIR" --run-dir "$ARTROOT/artifacts/mechanistic/judge_criterion_confirmation_qwen3_8b_v1/behavior" --suite-dir "$ARTROOT/data/derived/helpsteer2_criterion_confirmation_judge_criterion_confirmation_qwen3_8b_v1" --out-dir "$ARTROOT/artifacts/mechanistic/judge_criterion_confirmation_qwen3_8b_v1/analysis" && python -m aisafety.scripts.read_judge_criterion_confirmation --workspace-root "$WORKDIR" --analysis-dir "$ARTROOT/artifacts/mechanistic/judge_criterion_confirmation_qwen3_8b_v1/analysis"
```

Capture all 408 confirmation traces at the exact forced-decision boundary,
run the frozen-layer pair-held-out analysis, and patch criterion, score
evidence, and explicit-target state differences into criterion-only
recipients:

```bash
cd "$WORKDIR" && ARTROOT="$ARTROOT" GPU=7 bash cluster/local/run_judge_criterion_confirmation_qwen3_8b_mech.sh
```

The paper-facing layers are frozen from the development calibration:
criterion at layer 20, target and final choice at layer 32, current choice at
layer 28, and presentation order at layer 12. Patching uses branch zero in
both response orders and includes sign reversal, norm-matched random
orthogonal, same-target shuffled, opposite-target shuffled, and
same-target-transition controls.

Run the matched structured-CoT follow-up on the same locked 24 pairs. This
compares free CoT, a generic length-matched scaffold, a criterion-operational
scaffold, score evidence, an explicit-target ceiling, and a cached direct
answer baseline:

```bash
cd "$WORKDIR" && ARTROOT="$ARTROOT" GPU=7 bash cluster/local/run_judge_structured_cot_qwen3_8b_overnight.sh
```

The default run contains 408 CoT traces, 48 direct evaluations, and a
branch-zero activation replay of the 192 main-condition traces. The primary
effect is `criterion_scaffold - free_cot`; specificity requires the same
improvement against `generic_scaffold`. Success requires higher
order-consistent target adoption without a corresponding increase in invalid
or budget-saturated outputs. Fixed development-selected layers test whether
the scaffold also strengthens criterion-target and choice readouts. Set
`RUN_ACTIVATIONS=0` for behavior only. The complete readout is written to:

```bash
$ARTROOT/artifacts/mechanistic/judge_structured_cot_qwen3_8b_v1/readout.txt
```

Measure whether the phase-one rationales actually followed the requested
scaffold without rerunning the model:

```bash
cd "$WORKDIR" && PYTHONPATH="$WORKDIR/src" "$WORKDIR/.venv/bin/python" -m aisafety.scripts.analyze_judge_structured_cot_adherence --workspace-root "$WORKDIR" --run-dir "$ARTROOT/artifacts/mechanistic/judge_structured_cot_qwen3_8b_v1/behavior" --phase1-budget 128 --endpoint-budget 384 --bootstrap 5000 --audit-sample 32 --out-dir "$ARTROOT/artifacts/mechanistic/judge_structured_cot_qwen3_8b_v1/adherence_analysis"
```

The analyzer distinguishes numbered-step adherence from substantive criterion
operationalization, checks grounding against both displayed responses, and
reports pair-clustered associations with forced and order-consistent target
adoption. It also writes `adherence_audit_sample.csv` with empty human-review
columns. Compliance is an observational trace property: compliant versus
noncompliant differences are diagnostic, not causal effects.

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
