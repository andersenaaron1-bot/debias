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
