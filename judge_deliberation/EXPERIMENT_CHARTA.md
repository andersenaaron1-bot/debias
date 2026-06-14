# Experiment Charta: Deliberation Under Criterion Uncertainty

## Working Claim

Inference-time deliberation has positive returns when decisions have
verifiable criteria, but increasingly produces nontermination, instability,
or weakly productive computation as judgment criteria become underspecified or
plural.

This is not yet an established claim. The existing Qwen3-8B scout motivates
it:

- thinking improved ARC Challenge and TruthfulQA decisions;
- thinking consumed more tokens and timed out more often on moral,
  tradeoff-preference, and human-vs-LLM judgments;
- the evaluative domains did not show reliable gains in branch agreement;
- every invalid thinking trace exhausted the available token budget.

## Competing Explanations

1. **Useful-compute account:** evaluative tasks simply require more tokens than
   the current cap.
2. **Criterion-uncertainty account:** computation becomes less productive when
   the decision rule does not determine a unique answer.
3. **Task-difficulty account:** the apparent progression is ordinary
   difficulty, unrelated to whether a criterion is verifiable.
4. **Elicitation artifact:** Qwen3's thinking template or verdict parser causes
   the effect.
5. **Small-model weakness:** larger reasoning models resolve the ambiguity
   cleanly.
6. **Verbalization artifact:** visible chain-of-thought length changes without
   corresponding changes in internal decision formation.

The experiment sequence is designed to separate these accounts.

## Task Progression

| Class | Dataset/condition | Target | Criterion determinacy |
|---|---|---|---|
| Formal verification | GSM8K corrupted-final-answer pairs | Exact | Exact |
| Formal verification | MATH-500 corrupted-final-answer pairs | Exact | Exact |
| Logical verification | BBH logical deduction | Exact | Exact |
| Factual | ARC Challenge | Gold answer | Explicit |
| Misleading factual | TruthfulQA | Gold answer | Explicit |
| Consensus normative | ETHICS Commonsense | Consensus label | Moderate |
| Dominated preference | HelpSteer2 Pareto-dominated pairs | Preference proxy | Ordered |
| Criterion-resolved tradeoff | HelpSteer2 by correctness/helpfulness/coherence | Attribute label | Explicit proxy |
| Plural tradeoff | HelpSteer2 overall-quality tradeoff | No total order | Underspecified |
| Ecological judgment | D4 human-vs-LLM quality | No gold | Underspecified |

The HelpSteer2 and D4 responses are repeated under explicit criterion
interventions. The comparison texts remain fixed; only the decision rule
changes. This within-pair manipulation is the main test of the
criterion-uncertainty account.

### Matched HelpSteer2 Evidence Collector

The first criterion test removes the cross-dataset and answer-format
confounds. It selects the same HelpSteer2 response pairs from four strata:

- Pareto dominance;
- one-attribute separation;
- multi-attribute tradeoff;
- annotation near-tie.

Every pair is judged under overall, correctness, helpfulness, coherence, and a
fixed weighted rubric, in both presentation orders. The verdict set is A/B/C,
where C means tied or underdetermined under the stated rule. The primary
estimands are:

- criterion-specific target compliance after requiring order consistency;
- whether verdicts switch when the criterion-specific target switches;
- justified-tie recall and unjustified-tie rate;
- direct-to-thinking changes in those measures;
- branch agreement, revisions, natural termination, and forced confidence.

The target is a rule-compliance proxy derived from HelpSteer2 annotations. For
overall tradeoffs and near-ties, C is correct because the prompt explicitly
defines those cases as underdetermined, not because HelpSteer2 supplies a
universal preference label.

## Phase 1: Qwen3-8B Token-Budget Curves

### Conditions

- Same Qwen3-8B checkpoint in native chat serialization.
- Direct/no-thinking forced verdict.
- Thinking prefixes at 0, 128, 256, 512, 1024, and 2048 generated tokens.
- Both response orders.
- Five sampled branches for the confirmation run.
- At least 60 source pairs per condition after the scout validates runtime.

One maximal thinking path is generated per branch. At each budget prefix the
reasoning block is closed and the next-token A/B verdict is scored. This
isolates the marginal information in additional reasoning while avoiding
independent resampling at every budget.

### Primary Outcomes

- unconditional target success;
- forced-prefix target success;
- natural verdict rate by budget;
- net wrong-to-correct minus correct-to-wrong transitions;
- order-consistent majority rate;
- branch agreement and branch entropy;
- decision revision count;
- first stable forced verdict;
- A/B confidence and confidence change per added token;
- generated-token and budget-saturation rates.

### Primary Test

Estimate the interaction between token budget and task/criterion class. Report
pair-bootstrap confidence intervals. Do not reduce the primary result to one
hand-assigned determinacy scalar.

### Claim Gate

Advance the working claim only if:

1. formal/factual controls improve or plateau with added computation;
2. underspecified judgments consume more computation or terminate less often;
3. extra computation does not comparably improve agreement or a defensible
   target on those judgments; and
4. explicit criteria rescue at least part of the evaluative-task failure on
   the same response pairs.

If only token length differs, the result is an elicitation observation, not a
reasoning-quality claim.

The matched HelpSteer2 collector is the first test of item 4. A positive result
requires better criterion compliance or justified ties, not merely higher
confidence or longer rationales.

## Phase 2: Fixed-Decoder Activation Dynamics

The current point-specific probe probabilities are not calibrated across time.
The corrected analysis therefore:

1. trains endpoint decoders on `fit` pairs;
2. selects layer and regularization using `selection` pairs;
3. freezes the decoder;
4. applies that decoder to every earlier trajectory state on `intervention`
   pairs.

Measure:

- final-choice evidence emergence;
- first persistent commitment;
- sign reversals and confidence drops;
- evidence velocity per generated token;
- target/criterion evidence relative to choice evidence;
- trajectory path length and efficiency;
- cross-branch endpoint convergence.

### Activation Claim Gate

Use language such as "choice information becomes recoverable before criterion
information" unless an intervention changes the verdict. Linear
recoverability alone is not causal evidence and visible chain-of-thought is not
assumed faithful.

## Criterion Use: Deliberation Versus Rationalization

The next claim-critical experiment asks whether the model represents an active
criterion, constructs the target implied by that criterion, and allows that
target to control its verdict. These are separate questions.

### Operational distinction

Deliberation requires:

1. criterion identity to update when the rule changes;
2. criterion-conditioned target evidence to update;
3. final-choice evidence and the emitted verdict to follow that target; and
4. confidence to increase primarily when criterion compliance improves.

The behavior is rationalization-like when criterion or target information is
recoverable, but the final choice remains anchored and becomes more confident
or stable. Visible rationale text alone is not sufficient evidence.

### Held-out HelpSteer2 switch suite

Use new HelpSteer2 training-split pairs that do not overlap the validation-split
first matched-criterion scout:

- choice-to-choice conflicts, where two criteria prefer opposite responses;
- tie-to-choice conflicts, where one criterion is tied and another selects a
  response;
- same-target controls, where both criteria select the same response.

Correctness, helpfulness, and coherence are the primary criteria. Every pair
is presented in both orders. Pair-level analysis splits are frozen before any
activation modeling.

### Staged behavioral protocol

Each sampled branch has a 128-token first stage and a 384-token second stage.
The first-stage analysis is reused where conditions share the same initial
criterion.

1. **Stable:** keep the initial criterion without restating it.
2. **Reminder:** restate the initial criterion.
3. **Switch:** replace the initial criterion with a criterion whose target is
   known to differ or remain the same according to the pair stratum.
4. **Placebo:** provide a length-matched neutral review instruction.
5. **Delayed criterion:** first compare without a decision rule, then reveal
   the criterion.

Forced A/B/C readouts are collected before reasoning, during stage one, at the
update boundary, and during stage two. Primary behavioral outcomes are target
adoption, anchoring to the pre-update verdict, revision latency, semantic order
invariance, natural verdict validity, and confidence change.

### Point-aligned activation artifact

Replay the exact sampled token sequences and capture selected residual streams
at:

- stage-one prompt end;
- 64 and 128 stage-one tokens;
- stage-two prompt end, immediately after the update;
- 32, 128, and 384 stage-two tokens.

Every point records its active criterion and criterion-implied semantic target.
The trace also records the final semantic choice. This point-level labeling is
required because criterion and target can change within one trace.

The paper-facing artifact additionally captures the exact forced-decision
readout state after the reasoning prefix is closed and the prompt ends in
`FINAL:`. Raw generated-prefix states and decision-readout states are reported
separately. This distinguishes information present during free generation from
information available at the actual behavioral measurement boundary.

### Fixed held-out decoders

Fit pair-split endpoint decoders for:

- active criterion identity;
- criterion-conditioned semantic target;
- final semantic choice;
- presentation order as a nuisance control.

Select layer and regularization only on selection pairs, then freeze the
decoder and apply it to all time points on intervention pairs. The central
descriptive quantities are criterion-update latency, target-update latency,
choice-update latency, and the target-to-choice utilization gap.

Hyperparameters are selected on the frozen fit and selection pairs. Reported
temporal metrics are then cross-fitted by source pair within the untouched
intervention split. Order swaps, conditions, and sampled branches from one pair
remain in the same fold. Confidence intervals resample pairs rather than
traces. A switch-minus-reminder analysis applies the same decoder protocol to
within-pair activation differences. A pooled fit-plus-intervention result may
be emitted as an explicitly exploratory sensitivity analysis, not as the
confirmatory estimate.

### Factual readout calibration

Before interpreting HelpSteer2 decoder magnitude, replay the completed
Qwen3-8B budget traces for ARC Challenge, BBH logical deduction, GSM8K
verification, MATH-500 verification, and TruthfulQA. Capture the exact
`FINAL:` readout states at 0, 128, 512, and 2048 generated tokens and run the
same pair-grouped temporal analysis for criterion, target, current choice,
final choice, and presentation order.

This baseline does not define a universal mechanistic threshold. It provides a
within-model reference for how strongly verifiable targets and evolving
decisions are linearly recoverable under the same readout construction,
layers, folds, and token budgets.

### Same-pair criterion patching

For each pair, order, and branch, compare the reminder and switch conditions.
At a fixed point and layer, compute the within-pair state difference:

```text
switch state - reminder state
```

Add this difference to the reminder replay and measure whether the continuation
moves toward the switched criterion target. Required controls are the negative
direction, a same-target placebo difference, and a shuffled-pair criterion
difference. Test prompt-end, 128-token, and late-stage points.

A criterion-state intervention supports a scoped causal-control claim only
when it redirects held-out verdicts more than these controls. If criterion or
target information is decodable but patching does not affect choice, the result
supports representation without demonstrated control.

### Locked operationalization confirmation

The 72 pairs from the scout and extension runs are development data. Freeze
prompts, checkpoints, target definitions, and later patch settings before
opening a new 24-pair HelpSteer2 confirmation set:

- 9 choice-to-choice conflicts;
- 9 tie-to-choice conflicts, always directed from an initial tie to an updated
  A/B target;
- 6 same-target controls.

Each pair is audited under both active criteria and both presentation orders,
yielding 96 blinded human target checks. The main behavioral experiment crosses
criterion timing with criterion-specific score evidence:

| Condition | Criterion timing | Additional evidence |
|---|---|---|
| early criterion | before 128-token analysis | none |
| late criterion | after 128-token old-criterion analysis | none |
| early evidence | before 128-token analysis | HelpSteer criterion scores |
| late evidence | after 128-token old-criterion analysis | HelpSteer criterion scores |

The evidence gives the two criterion-specific 0--4 scores without stating an
A/B/C verdict. Twelve conflict pairs additionally receive a late explicit
criterion target as a downstream-control ceiling. Both orders and two branches
produce 384 main traces; the one-branch ceiling adds 24 traces.

The primary confirmatory estimands are:

1. operationalization rescue from score evidence;
2. the penalty for presenting identical criterion information after an
   initial commitment;
3. the timing-by-evidence interaction;
4. whether a directly supplied target overcomes the prior verdict;
5. order-consistent target adoption, revision latency, confidence change,
   natural validity, and budget saturation.

Do not call the result rationalization merely because the criterion is
decodable. A rationalization-like claim requires a commitment-dependent loss
of target adoption when equivalent criterion or target information is supplied
late, followed by controlled target or choice interventions that localize the
blocked stage.

The first locked behavioral run instead shows a timing-invariant
operationalization gap: criterion-only target adoption is approximately
0.59 early and 0.53 late, while criterion-specific score evidence raises both
to approximately 0.99. The explicit target ceiling is 1.00. The current claim
is therefore awareness or criterion representation without reliable
operationalization, not commitment-driven rationalization.

The frozen mechanistic adjudication has three matched differences:

1. early-criterion minus late-criterion at the phase-one 128-token readout,
   patched at the development-selected criterion layer 20;
2. late-evidence minus late-criterion at the phase-two zero-token readout,
   patched at target layer 32;
3. late-explicit-target minus late-criterion at the same phase-two readout and
   layer.

Each patch is evaluated on conflict pairs in both orders using direct A/B/C
readout logits. Required controls are sign reversal, a magnitude-matched
random orthogonal vector, a shuffled same-target vector, a shuffled
opposite-target vector, and, where available, a vector from a same-target
transition. Pair-bootstrap effects on target probability, target logit margin,
discrete target adoption, and order consistency determine whether the
represented criterion/evidence update is sufficient to control the verdict.

## Phase 3: Replication

Freeze the 8B design before replication.

1. **Qwen3-30B-A3B:** preferred scale replication because it has a native
   thinking switch and low active-parameter inference cost.
2. **Qwen3-32B:** dense same-family replication if cluster time permits.
3. **Phi-4-reasoning-plus:** cross-family reasoning-model replication.
4. **Gemma 3 27B IT:** optional prompted-CoT robustness condition, not a clean
   native thinking/non-thinking contrast.

The first larger-model pass is behavioral only:

- 30-40 source pairs per selected class;
- both orders;
- three branches;
- direct plus 256, 1024, and 2048 token prefixes.

Run activation capture only after the behavioral pattern replicates.

## Phase 4: Causal Tests

Only stable held-out fixed decoders are intervention candidates. Required
controls:

- fitted decision-direction steering;
- sign-reversed steering;
- magnitude-matched random orthogonal directions;
- order-swap controls;
- factual competence retention;
- criterion-specific retention.

A successful intervention supports a scoped causal-influence claim at the
tested layer, point, model, and task. It does not establish a universal
reasoning mechanism.

## Optional Human Reference

For 100-200 HelpSteer2 tradeoff and D4 pairs:

- collect three blinded ratings per pair;
- allow indifference;
- record confidence;
- collect overall and criterion-specific choices.

This separates movement toward human consensus from increased model
self-consistency.

## Paper-Level Minimum

The minimum complete package is:

1. Qwen3-8B token-budget curves;
2. within-pair criterion rescue;
3. one approximately 30B behavioral replication;
4. corrected fixed-decoder trajectory analysis.

The criterion-use extension refines this package:

1. factual-to-evaluative budget curves establish the external behavioral
   progression;
2. matched HelpSteer2 results isolate criterion sensitivity on fixed texts;
3. staged criterion switches distinguish representation, target construction,
   and verdict control;
4. same-pair patching tests whether criterion-induced states causally redirect
   decisions.

Cross-family replication, human ratings, and causal interventions strengthen
the paper but should not delay the minimum package.

## Falsification Outcomes

- If evaluative tasks improve monotonically at larger budgets, report a
  compute-requirement result rather than overthinking.
- If criterion prompts do not rescue behavior, criterion uncertainty is not
  isolated.
- If effects vanish with forced-prefix verdicts, nontermination is primarily
  output-format behavior.
- If effects vanish in Qwen3-30B-A3B, small-model weakness remains viable.
- If fixed decoders show stable early choices on every task, long reasoning may
  be post-hoc elaboration; this requires intervention before stronger wording.
