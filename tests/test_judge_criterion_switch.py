import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from aisafety.scripts.analyze_judge_criterion_switch_pairs import (
    _candidate_selection,
    _cross_fitted_predictions,
)
from aisafety.scripts.analyze_judge_criterion_switch_decoders import (
    _fit,
    _metrics,
)
from aisafety.scripts.analyze_judge_criterion_confirmation import (
    PAIR_METRICS,
    paired_effects,
)
from aisafety.scripts.analyze_judge_criterion_confirmation_activations import (
    difference_rows,
)
from aisafety.scripts.analyze_judge_structured_cot import (
    _checkpoint_pair_metrics,
    _paired_effects,
)
from aisafety.scripts.analyze_judge_structured_cot_adherence import (
    _outcome_associations,
    branch_pair_rows,
    score_adherence,
)
from aisafety.scripts.analyze_judge_structured_cot_enforced import (
    pair_metrics as enforced_pair_metrics,
    trace_rows as enforced_trace_rows,
)
from aisafety.scripts.build_helpsteer2_criterion_switch_suite import (
    _pair_signature,
    _transition_candidates,
    build_episodes,
    build_switch_pairs,
)
from aisafety.scripts.build_helpsteer2_criterion_confirmation import (
    TRANSITION_QUOTAS,
    build_confirmation_episodes,
    score_evidence_text,
    select_confirmation_pairs,
    write_audit_bundle,
)
from aisafety.scripts.build_helpsteer2_structured_cot_suite import (
    CRITERION_SCAFFOLD,
    GENERIC_SCAFFOLD,
    build_structured_cot_episodes,
)
from aisafety.scripts.build_helpsteer2_enforced_structure_suite import (
    build_episodes as build_enforced_episodes,
)
from aisafety.scripts.build_judge_reasoning_source_pack import ATTRIBUTE_NAMES
from aisafety.scripts.run_judge_criterion_switch_activations import (
    filter_traces,
    point_forced_choices,
    point_labels,
    point_specs,
    point_step_indices,
    point_token_sequences,
)
from aisafety.scripts.run_judge_criterion_switch_behavior import (
    _phase1_key,
    _semantic_verdict,
    phase1_user_content,
    phase2_update_content,
)
from aisafety.scripts.run_judge_structured_cot_enforced import (
    _read_jsonl_if_exists,
    final_content as enforced_final_content,
    long_analysis_content,
    stage_contents,
)
from aisafety.scripts.run_judge_criterion_switch_patching import _paired
from aisafety.scripts.run_judge_criterion_confirmation_patching import (
    _select_control,
    matched_condition_rows,
    patch_effects,
    summarize_patch_rows,
)
from aisafety.scripts.read_judge_criterion_confirmation import _read


def _row(prompt: str, response: str, values: tuple[float, ...]) -> dict:
    return {
        "prompt": prompt,
        "response": response,
        **dict(zip(ATTRIBUTE_NAMES, values, strict=True)),
    }


def _source_rows() -> list[dict]:
    rows: list[dict] = []
    for index in range(5):
        rows.extend(
            [
                _row(f"choice-{index}", "helpful", (4, 2, 3, 2, 2)),
                _row(f"choice-{index}", "correct", (2, 4, 3, 2, 2)),
                _row(f"tie-{index}", "helpful", (4, 3, 3, 2, 2)),
                _row(f"tie-{index}", "tied", (2, 3, 3, 2, 2)),
                _row(f"same-{index}", "strong", (4, 4, 4, 2, 2)),
                _row(f"same-{index}", "weak", (2, 2, 2, 2, 2)),
            ]
        )
    return rows


def _confirmation_candidates() -> list[dict]:
    rows: list[dict] = []
    criteria = ("correctness", "helpfulness", "coherence")
    for transition_type in TRANSITION_QUOTAS:
        for index in range(12):
            initial = criteria[index % len(criteria)]
            updated = criteria[(index + 1) % len(criteria)]
            if transition_type == "choice_to_choice":
                initial_target = "A" if index % 2 == 0 else "B"
                updated_target = "B" if initial_target == "A" else "A"
            elif transition_type == "tie_to_choice":
                initial_target = "C"
                updated_target = "A" if index % 2 == 0 else "B"
            else:
                initial_target = updated_target = "A" if index % 2 == 0 else "B"
            targets = {
                criterion: "C" for criterion in criteria
            }
            targets[initial] = initial_target
            targets[updated] = updated_target
            gaps = {
                criterion: (
                    0.0
                    if targets[criterion] == "C"
                    else 2.0
                    if targets[criterion] == "A"
                    else -2.0
                )
                for criterion in criteria
            }
            rows.append(
                {
                    "pair_id": f"{transition_type}-{index}",
                    "pair_signature": f"signature-{transition_type}-{index}",
                    "prompt": f"prompt-{transition_type}-{index}",
                    "option_a_text": "response a",
                    "option_b_text": "response b",
                    "option_a_attributes": {
                        "helpfulness": 4.0,
                        "correctness": 4.0,
                        "coherence": 4.0,
                        "complexity": 2.0,
                        "verbosity": 2.0,
                    },
                    "option_b_attributes": {
                        "helpfulness": 2.0,
                        "correctness": 2.0,
                        "coherence": 2.0,
                        "complexity": 2.0,
                        "verbosity": 2.0,
                    },
                    "criterion_gaps_a_minus_b": gaps,
                    "criterion_targets": targets,
                    "transition_type": transition_type,
                    "initial_criterion_id": initial,
                    "updated_criterion_id": updated,
                    "initial_target_semantic": initial_target,
                    "updated_target_semantic": updated_target,
                    "source_split": "train",
                }
            )
    return rows


class CriterionSwitchSuiteTests(unittest.TestCase):
    def test_transition_types(self) -> None:
        choice = _transition_candidates(
            {"helpfulness": "A", "correctness": "B", "coherence": "C"},
            {"helpfulness": 2.0, "correctness": -2.0, "coherence": 0.0},
            min_choice_gap=1.0,
        )
        self.assertTrue(choice["choice_to_choice"])
        self.assertTrue(choice["tie_to_choice"])
        same = _transition_candidates(
            {"helpfulness": "A", "correctness": "A", "coherence": "A"},
            {"helpfulness": 2.0, "correctness": 2.0, "coherence": 2.0},
            min_choice_gap=1.0,
        )
        self.assertTrue(same["same_target"])

    def test_content_signature_ignores_response_order(self) -> None:
        self.assertEqual(
            _pair_signature("prompt", "left", "right"),
            _pair_signature("prompt", "right", "left"),
        )
        excluded = _pair_signature("choice-0", "helpful", "correct")
        pairs = build_switch_pairs(
            _source_rows(),
            excluded_pair_signatures={excluded},
            max_pairs_per_transition=4,
            min_pairs_per_transition=4,
            min_choice_gap=1.0,
            seed=1234,
        )
        self.assertNotIn(excluded, {row["pair_signature"] for row in pairs})

    def test_pair_splits_and_episode_conditions(self) -> None:
        pairs = build_switch_pairs(
            _source_rows(),
            excluded_pair_signatures=set(),
            max_pairs_per_transition=5,
            min_pairs_per_transition=5,
            min_choice_gap=1.0,
            seed=1234,
        )
        self.assertEqual(len(pairs), 15)
        for transition in (
            "choice_to_choice",
            "tie_to_choice",
            "same_target",
        ):
            values = [
                row for row in pairs if row["transition_type"] == transition
            ]
            self.assertEqual(
                [row["analysis_split"] for row in values].count("fit"), 3
            )
            self.assertEqual(
                [row["analysis_split"] for row in values].count("selection"),
                1,
            )
            self.assertEqual(
                [row["analysis_split"] for row in values].count(
                    "intervention"
                ),
                1,
            )
        for pair in pairs:
            pair["source_split"] = "train"
        episodes = build_episodes(pairs[:1])
        self.assertEqual(len(episodes), 10)
        self.assertEqual({row["split"] for row in episodes}, {"train"})
        self.assertEqual(
            {row["condition_id"] for row in episodes},
            {"stable", "reminder", "switch", "placebo", "delayed"},
        )
        self.assertEqual(
            {row["presentation_order"] for row in episodes},
            {"original", "swapped"},
        )

    def test_semantic_verdict_swap(self) -> None:
        self.assertEqual(_semantic_verdict("A", "original"), "A")
        self.assertEqual(_semantic_verdict("A", "swapped"), "B")
        self.assertEqual(_semantic_verdict("C", "swapped"), "C")

    def test_confirmation_design_counts_and_audit_bundle(self) -> None:
        pairs = select_confirmation_pairs(
            _confirmation_candidates(),
            quotas=TRANSITION_QUOTAS,
            seed=1234,
        )
        self.assertEqual(len(pairs), 24)
        self.assertEqual(
            pd.Series([row["transition_type"] for row in pairs])
            .value_counts()
            .to_dict(),
            TRANSITION_QUOTAS,
        )
        episodes, ceiling_ids = build_confirmation_episodes(
            pairs,
            main_branches=2,
            ceiling_pairs_per_conflict_transition=6,
            seed=1234,
        )
        self.assertEqual(len(episodes), 216)
        self.assertEqual(
            sum(int(row["branches_per_episode"]) for row in episodes),
            408,
        )
        self.assertEqual(len(ceiling_ids), 12)
        self.assertTrue(
            all(
                row["initial_target_semantic"] == "C"
                for row in pairs
                if row["transition_type"] == "tie_to_choice"
            )
        )
        with tempfile.TemporaryDirectory() as raw:
            audit = write_audit_bundle(Path(raw), pairs)
            self.assertEqual(audit["n_audit_prompts"], 96)
            self.assertEqual(
                len(list((Path(raw) / "human_audit" / "prompts").glob("*.txt"))),
                96,
            )

    def test_confirmation_prompt_variants_and_cache_sharing(self) -> None:
        pair = _confirmation_candidates()[0]
        pairs = select_confirmation_pairs(
            [pair, *_confirmation_candidates()[1:]],
            quotas=TRANSITION_QUOTAS,
            seed=1234,
        )
        episodes, _ = build_confirmation_episodes(
            pairs,
            main_branches=2,
            ceiling_pairs_per_conflict_transition=6,
            seed=1234,
        )
        late = next(
            row for row in episodes if row["condition_id"] == "late_criterion"
        )
        late_evidence = next(
            row
            for row in episodes
            if row["pair_id"] == late["pair_id"]
            and row["presentation_order"] == late["presentation_order"]
            and row["condition_id"] == "late_evidence"
        )
        explicit = next(
            row
            for row in episodes
            if row["condition_id"] == "late_explicit_target"
        )
        early_evidence = next(
            row for row in episodes if row["condition_id"] == "early_evidence"
        )
        self.assertEqual(
            _phase1_key(late, branch_index=0),
            _phase1_key(late_evidence, branch_index=0),
        )
        self.assertIn(
            "annotation evidence",
            phase1_user_content(early_evidence),
        )
        self.assertIn(
            "annotation evidence",
            phase2_update_content(late_evidence),
        )
        self.assertIn(
            f"implies Option {explicit['phase2_target_option']}",
            phase2_update_content(explicit),
        )
        self.assertIn(
            "Option A",
            score_evidence_text(
                pair,
                criterion_id=str(pair["updated_criterion_id"]),
                presentation_order="original",
            ),
        )

    def test_confirmation_effect_signs(self) -> None:
        values = {
            "early_criterion": 0.5,
            "early_evidence": 0.8,
            "late_criterion": 0.4,
            "late_evidence": 0.5,
            "late_explicit_target": 0.9,
        }
        rows = []
        for pair_id in ("p1", "p2", "p3"):
            for condition, value in values.items():
                row = {
                    "pair_id": pair_id,
                    "condition_id": condition,
                    "transition_type": "choice_to_choice",
                    "n_traces": 4,
                }
                row.update({metric: value for metric in PAIR_METRICS})
                rows.append(row)
        effects = paired_effects(
            pd.DataFrame(rows),
            bootstrap=100,
            seed=1234,
        )
        target = effects[
            effects["metric"].eq("forced_target_adoption")
            & effects["transition_type"].eq("all")
        ].set_index("contrast")
        self.assertAlmostEqual(
            target.loc["early_operationalization_rescue", "mean"],
            0.3,
        )
        self.assertAlmostEqual(
            target.loc["late_operationalization_rescue", "mean"],
            0.1,
        )
        self.assertAlmostEqual(
            target.loc["timing_by_evidence_interaction", "mean"],
            0.2,
        )
        self.assertAlmostEqual(
            target.loc["explicit_target_vs_late_criterion", "mean"],
            0.5,
        )

    def test_confirmation_reader_accepts_empty_optional_csv(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "audit_pair_rows.csv").write_text("", encoding="utf-8")
            self.assertTrue(_read(root, "audit_pair_rows.csv").empty)

    def test_structured_cot_design_counts_and_prompt_controls(self) -> None:
        pairs = select_confirmation_pairs(
            _confirmation_candidates(),
            quotas=TRANSITION_QUOTAS,
            seed=1234,
        )
        conflict_ids = {
            str(row["pair_id"])
            for row in pairs
            if row["transition_type"] != "same_target"
        }
        ceiling_ids = set(sorted(conflict_ids)[:12])
        episodes = build_structured_cot_episodes(
            pairs,
            ceiling_pair_ids=ceiling_ids,
            main_branches=2,
            ceiling_branches=1,
            include_explicit_target=True,
        )
        self.assertEqual(len(episodes), 216)
        self.assertEqual(
            sum(int(row["branches_per_episode"]) for row in episodes),
            408,
        )
        direct_keys = {
            (
                row["pair_id"],
                row["presentation_order"],
                row["direct_criterion_ids"][0],
            )
            for row in episodes
        }
        self.assertEqual(len(direct_keys), 48)
        criterion = next(
            row
            for row in episodes
            if row["condition_id"] == "criterion_scaffold"
        )
        generic = next(
            row
            for row in episodes
            if row["pair_id"] == criterion["pair_id"]
            and row["presentation_order"]
            == criterion["presentation_order"]
            and row["condition_id"] == "generic_scaffold"
        )
        self.assertEqual(
            criterion["phase1_reasoning_instructions"],
            CRITERION_SCAFFOLD,
        )
        self.assertEqual(
            generic["phase1_reasoning_instructions"],
            GENERIC_SCAFFOLD,
        )
        self.assertNotIn(
            "implies Option",
            criterion["phase1_reasoning_instructions"],
        )
        self.assertLess(
            abs(
                len(CRITERION_SCAFFOLD.split())
                - len(GENERIC_SCAFFOLD.split())
            ),
            30,
        )

    def test_structured_cot_prompt_extension_points(self) -> None:
        episode = {
            "condition_id": "criterion_scaffold",
            "prompt": "question",
            "option_a_text": "left",
            "option_b_text": "right",
            "phase1_criterion_text": "Use correctness.",
            "phase2_criterion_text": "Use correctness.",
            "phase1_evidence_text": "",
            "phase2_evidence_text": "",
            "phase1_reasoning_instructions": "Apply the rule step by step.",
            "phase2_update_override": "Use the prior structured analysis.",
        }
        phase1 = phase1_user_content(episode)
        self.assertIn("Reasoning procedure", phase1)
        self.assertIn("Apply the rule step by step.", phase1)
        self.assertIn(
            "Use the prior structured analysis.",
            phase2_update_content(episode),
        )

    def test_structured_cot_adherence_scoring(self) -> None:
        compliant = score_adherence(
            {
                "phase1_criterion_id": "correctness",
                "option_a_text": (
                    "Paris is the capital of France and has a large "
                    "population."
                ),
                "option_b_text": (
                    "Lyon is the capital of France and has a smaller "
                    "population."
                ),
                "phase1_response_text": (
                    "1. The correctness test is whether the claimed capital "
                    "is factually correct.\n"
                    "2. Option A says Paris is the capital of France, which "
                    "passes that factual test.\n"
                    "3. Option B says Lyon is the capital of France, which "
                    "fails the same factual test.\n"
                    "4. Comparing the criterion-specific evidence, Option A "
                    "is better."
                ),
            }
        )
        self.assertEqual(compliant["explicit_step_count"], 4)
        self.assertTrue(compliant["content_compliant"])
        self.assertTrue(compliant["format_compliant"])
        self.assertTrue(compliant["strict_compliant"])

        superficial = score_adherence(
            {
                "phase1_criterion_id": "correctness",
                "option_a_text": "Paris is the capital of France.",
                "option_b_text": "Lyon is the capital of France.",
                "phase1_response_text": (
                    "Option A and Option B are both presented clearly. "
                    "There are several differences."
                ),
            }
        )
        self.assertFalse(superficial["content_compliant"])
        self.assertFalse(superficial["strict_compliant"])

    def test_enforced_structure_design_and_option_isolation(self) -> None:
        pairs = select_confirmation_pairs(
            _confirmation_candidates(),
            quotas=TRANSITION_QUOTAS,
            seed=1234,
        )
        episodes = build_enforced_episodes(pairs, branches=1)
        self.assertEqual(len(episodes), 192)
        self.assertEqual(
            {row["condition_id"] for row in episodes},
            {
                "free_long",
                "prompted_long",
                "enforced_generic",
                "enforced_criterion",
            },
        )
        episode = next(
            row
            for row in episodes
            if row["condition_id"] == "enforced_criterion"
        )
        name, stage1 = stage_contents(episode, [])
        self.assertEqual(name, "criterion_tests")
        self.assertNotIn(episode["option_a_text"], stage1)
        self.assertNotIn(episode["option_b_text"], stage1)

        artifacts = [
            {
                "stage_name": name,
                "response_text": "TESTS: factual accuracy",
            }
        ]
        name, stage2 = stage_contents(episode, artifacts)
        self.assertEqual(name, "option_a_assessment")
        self.assertIn(episode["option_a_text"], stage2)
        self.assertNotIn(episode["option_b_text"], stage2)

        artifacts.append(
            {
                "stage_name": name,
                "response_text": "A passes.",
            }
        )
        name, stage3 = stage_contents(episode, artifacts)
        self.assertEqual(name, "option_b_assessment")
        self.assertIn(episode["option_b_text"], stage3)
        self.assertNotIn(episode["option_a_text"], stage3)

        artifacts.append(
            {
                "stage_name": name,
                "response_text": "B fails.",
            }
        )
        name, stage4 = stage_contents(episode, artifacts)
        self.assertEqual(name, "criterion_comparison")
        self.assertIn("A passes.", stage4)
        self.assertIn("B fails.", stage4)
        artifacts.append(
            {
                "stage_name": name,
                "response_text": "A is supported.",
            }
        )
        final = enforced_final_content(episode, artifacts)
        self.assertIn("FINAL: A", final)
        self.assertIn("factual accuracy", final)

    def test_enforced_structure_long_prompt_control(self) -> None:
        episode = {
            "condition_id": "prompted_long",
            "criterion_id": "correctness",
            "criterion_text": "Judge correctness.",
            "prompt": "Question",
            "option_a_text": "A text",
            "option_b_text": "B text",
        }
        prompted = long_analysis_content(episode)
        self.assertIn(CRITERION_SCAFFOLD, prompted)
        episode["condition_id"] = "free_long"
        free = long_analysis_content(episode)
        self.assertNotIn(CRITERION_SCAFFOLD, free)

    def test_enforced_structure_fresh_jsonl_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing.jsonl"
            self.assertEqual(_read_jsonl_if_exists(path), [])


class CriterionSwitchActivationTests(unittest.TestCase):
    def test_activation_filter_selects_conditions_and_branches(self) -> None:
        rows = [
            {
                "trace_id": f"{condition}-{branch}",
                "condition_id": condition,
                "branch_index": branch,
            }
            for condition in ("free_cot", "criterion_scaffold")
            for branch in (0, 1)
        ]
        selected = filter_traces(
            rows,
            include_conditions={"criterion_scaffold"},
            include_branches={0},
        )
        self.assertEqual(
            [row["trace_id"] for row in selected],
            ["criterion_scaffold-0"],
        )

    def test_point_prefixes_and_labels(self) -> None:
        row = {
            "phase1_prompt_token_ids": [1, 2],
            "phase1_response_token_ids": list(range(200)),
            "phase2_prompt_token_ids": [3, 4, 5],
            "phase2_response_token_ids": list(range(500)),
            "phase1_criterion_id": "correctness",
            "phase2_criterion_id": "helpfulness",
            "phase1_target_semantic": "A",
            "phase2_target_semantic": "B",
        }
        sequences = point_token_sequences(row)
        self.assertEqual(len(sequences), 7)
        self.assertEqual(len(sequences[1]), 66)
        self.assertEqual(len(sequences[5]), 131)
        criteria, targets = point_labels(row)
        self.assertEqual(
            criteria,
            ["correctness"] * 3 + ["helpfulness"] * 4,
        )
        self.assertEqual(targets, ["A"] * 3 + ["B"] * 4)

    def test_readout_points_end_at_forced_decision_boundary(self) -> None:
        class Tokenizer:
            def decode(self, token_ids, skip_special_tokens=False):
                del skip_special_tokens
                return "".join(chr(65 + int(value)) for value in token_ids)

            def __call__(
                self,
                text,
                *,
                add_special_tokens,
                truncation,
                max_length,
            ):
                self.last_text = text
                values = [1] + [ord(value) % 251 for value in text]
                return {"input_ids": values[:max_length]}

        row = {
            "phase1_prompt_text": "P1:",
            "phase1_response_token_ids": [0, 1, 2],
            "phase1_generated_tokens": 3,
            "phase2_prompt_text": "P2:",
            "phase2_response_token_ids": [3, 4, 5, 6],
            "phase2_generated_tokens": 4,
            "phase1_criterion_id": "correctness",
            "phase2_criterion_id": "helpfulness",
            "phase1_target_semantic": "A",
            "phase2_target_semantic": "B",
            "phase1_checkpoints": [
                {"budget_tokens": 0, "forced_choice_semantic": "A"},
                {"budget_tokens": 64, "forced_choice_semantic": "B"},
                {"budget_tokens": 128, "forced_choice_semantic": "B"},
            ],
            "phase2_checkpoints": [
                {"budget_tokens": 0, "forced_choice_semantic": "A"},
                {"budget_tokens": 32, "forced_choice_semantic": "B"},
                {"budget_tokens": 128, "forced_choice_semantic": "B"},
                {"budget_tokens": 384, "forced_choice_semantic": "C"},
            ],
        }
        tokenizer = Tokenizer()
        sequences = point_token_sequences(
            row,
            tokenizer=tokenizer,
            point_mode="readout",
            max_score_length=128,
        )
        self.assertEqual(len(sequences), 7)
        self.assertTrue(tokenizer.last_text.endswith("\nFINAL:"))
        self.assertEqual(
            [name for name, _stage, _budget in point_specs("readout")],
            [
                "phase1_readout_0",
                "phase1_readout_64",
                "phase1_readout_128",
                "phase2_readout_0",
                "phase2_readout_32",
                "phase2_readout_128",
                "phase2_readout_384",
            ],
        )
        self.assertEqual(
            point_forced_choices(row, point_mode="readout"),
            ["A", "B", "B", "A", "B", "B", "C"],
        )
        np.testing.assert_array_equal(
            point_step_indices(row, point_mode="readout"),
            np.asarray([0, 3, 3, 3, 7, 7, 7], dtype=np.int32),
        )

    def test_multiclass_decoder_helpers(self) -> None:
        x = np.asarray(
            [
                [3.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
                [0.0, 0.0, 2.0],
            ],
            dtype=np.float32,
        )
        labels = np.asarray(["a", "a", "b", "b", "c", "c"])
        center, model = _fit(x, labels, c_value=10.0, seed=1234)
        metrics = _metrics(model, center, x, labels)
        self.assertGreaterEqual(metrics["balanced_accuracy"], 0.99)

    def test_confirmation_difference_rows_align_conditions(self) -> None:
        point_names = [
            "phase1_readout_0",
            "phase1_readout_64",
            "phase1_readout_128",
            "phase2_readout_0",
            "phase2_readout_32",
            "phase2_readout_128",
            "phase2_readout_384",
        ]
        conditions = [
            "early_criterion",
            "late_criterion",
            "late_evidence",
            "late_explicit_target",
        ]
        frame = pd.DataFrame(
            [
                {
                    "trace_id": condition,
                    "pair_id": "pair-1",
                    "condition_id": condition,
                    "presentation_order": "original",
                    "branch_index": 0,
                    "transition_type": "choice_to_choice",
                    "point_names": point_names,
                    "phase1_criterion_id": (
                        "helpfulness"
                        if condition == "early_criterion"
                        else "correctness"
                    ),
                    "phase2_criterion_id": "helpfulness",
                    "phase1_target_semantic": (
                        "B"
                        if condition == "early_criterion"
                        else "A"
                    ),
                    "phase2_target_semantic": "B",
                    "point_forced_choices_semantic": ["A"] * 7,
                    "decoder_final_choice_semantic": "B",
                }
                for condition in conditions
            ]
        )

        class Artifact:
            def __init__(self):
                self.frame = frame

            def layer_states(self, layer):
                values = np.zeros((len(frame), 7, 3), dtype=np.float32)
                for index in range(len(frame)):
                    values[index] = float(index + layer)
                return values

        differences, states = difference_rows(
            Artifact(),
            target_layers={
                "active_criterion": 20,
                "criterion_target": 32,
                "current_choice": 28,
                "final_choice": 32,
                "presentation_order": 12,
            },
        )
        self.assertEqual(
            set(differences["difference_type"]),
            {
                "criterion_update",
                "evidence_operationalization",
                "explicit_target",
            },
        )
        self.assertEqual(len(differences), 6)
        self.assertEqual(states[20].shape, (6, 3))
        criterion = differences[
            differences["difference_type"].eq("criterion_update")
        ].iloc[0]
        self.assertEqual(criterion["active_criterion"], "helpfulness")
        self.assertEqual(criterion["criterion_target"], "B")

    def test_confirmation_patch_matching_and_controls(self) -> None:
        rows = []
        for pair_id, transition, target in (
            ("p1", "choice_to_choice", "A"),
            ("p2", "choice_to_choice", "A"),
            ("p3", "choice_to_choice", "B"),
            ("p4", "same_target", "A"),
        ):
            for condition in ("late_criterion", "late_evidence"):
                rows.append(
                    {
                        "trace_id": f"{pair_id}-{condition}",
                        "pair_id": pair_id,
                        "condition_id": condition,
                        "presentation_order": "original",
                        "branch_index": 0,
                        "transition_type": transition,
                        "phase1_target_option": "B",
                        "phase1_target_semantic": "B",
                        "phase2_target_option": target,
                        "phase2_target_semantic": target,
                    }
                )
        matches = matched_condition_rows(
            rows,
            donor_condition="late_evidence",
            recipient_condition="late_criterion",
            branch_index=0,
            include_orders={"original"},
        )
        self.assertEqual(len(matches), 4)
        current = next(row for row in matches if row["pair_id"] == "p1")
        same, quality = _select_control(
            current,
            matches,
            kind="shuffled_same_target",
            seed=1234,
            patch_type="evidence_operationalization",
        )
        self.assertEqual(same["pair_id"], "p2")
        self.assertEqual(quality, "same_order_target_transition")
        opposite, _ = _select_control(
            current,
            matches,
            kind="shuffled_opposite_target",
            seed=1234,
            patch_type="evidence_operationalization",
        )
        self.assertEqual(opposite["pair_id"], "p3")
        stable, _ = _select_control(
            current,
            matches,
            kind="same_target_donor",
            seed=1234,
            patch_type="evidence_operationalization",
        )
        self.assertEqual(stable["pair_id"], "p4")

    def test_confirmation_patch_effects_are_pair_grouped(self) -> None:
        rows = []
        for pair_id in ("p1", "p2"):
            for order in ("original", "swapped"):
                for setting, value, alpha in (
                    ("baseline", 0.25, 0.0),
                    ("matched_delta", 0.75, 1.0),
                    ("random_orthogonal", 0.30, 1.0),
                ):
                    rows.append(
                        {
                            "patch_type": "evidence_operationalization",
                            "pair_id": pair_id,
                            "presentation_order": order,
                            "branch_index": 0,
                            "setting": setting,
                            "alpha": alpha,
                            "predicted_semantic": "A",
                            "target_selected": value >= 0.5,
                            "target_probability": value,
                            "target_logit_margin": value - 0.5,
                            "choice_confidence": abs(value - 0.5),
                        }
                    )
        frame = pd.DataFrame(rows)
        summary = summarize_patch_rows(frame)
        self.assertEqual(summary["n_pairs"].max(), 2)
        effects = patch_effects(frame, bootstrap=100, seed=1234)
        target = effects[
            effects["left_setting"].eq("matched_delta")
            & effects["right_setting"].eq("baseline")
            & effects["metric"].eq("target_probability")
        ].iloc[0]
        self.assertAlmostEqual(target["mean"], 0.5)
        self.assertEqual(target["n_pairs"], 2)

    def test_structured_cot_checkpoint_effects_are_pair_matched(self) -> None:
        rows = []
        for pair_id in ("p1", "p2", "p3", "p4"):
            target = "A"
            for condition in ("free_cot", "criterion_scaffold"):
                for branch_index in (0, 1):
                    for order in ("original", "swapped"):
                        correct = (
                            condition == "criterion_scaffold"
                            or pair_id in {"p1", "p2"}
                        )
                        choice = target if correct else "B"
                        rows.append(
                            {
                                "trace_id": (
                                    f"{pair_id}-{condition}-"
                                    f"{branch_index}-{order}"
                                ),
                                "pair_id": pair_id,
                                "condition_id": condition,
                                "transition_type": "choice_to_choice",
                                "branch_index": branch_index,
                                "presentation_order": order,
                                "stage": "phase2",
                                "budget_tokens": 384,
                                "phase1_target_semantic": target,
                                "phase2_target_semantic": target,
                                "forced_choice_semantic": choice,
                                "forced_choice_confidence": 0.8,
                            }
                        )
        pair_metrics = _checkpoint_pair_metrics(pd.DataFrame(rows))
        effects = _paired_effects(
            pair_metrics,
            metrics=("forced_target_adoption",),
            bootstrap=100,
            seed=1234,
            extra_index=["stage", "budget_tokens"],
        )
        target = effects[
            effects["contrast"].eq("criterion_scaffold_rescue")
            & effects["transition_type"].eq("all")
        ].iloc[0]
        self.assertAlmostEqual(target["mean"], 0.5)
        self.assertEqual(target["n_pairs"], 4)

    def test_structured_cot_adherence_association_uses_pair_variation(
        self,
    ) -> None:
        rows = []
        for pair_index in range(8):
            for branch_index, score in enumerate((0.25, 1.0)):
                correct = float(score == 1.0)
                for order in ("original", "swapped"):
                    rows.append(
                        {
                            "pair_id": f"p{pair_index}",
                            "condition_id": "criterion_scaffold",
                            "transition_type": "choice_to_choice",
                            "branch_index": branch_index,
                            "presentation_order": order,
                            "criterion_procedure_score": score,
                            "content_compliant": bool(correct),
                            "strict_compliant": bool(correct),
                            "forced_choice_semantic": (
                                "A" if correct else "B"
                            ),
                            "phase2_target_semantic": "A",
                            "forced_target_adoption": bool(correct),
                        }
                    )
        pairs = branch_pair_rows(pd.DataFrame(rows))
        associations = _outcome_associations(
            pairs,
            bootstrap=100,
            seed=1234,
        )
        slope = associations[
            associations["association"].eq("within_pair_score_slope")
            & associations["outcome"].eq("forced_target_adoption")
        ].iloc[0]
        self.assertGreater(slope["estimate"], 1.0)
        compliant = associations[
            associations["association"].eq(
                "both_orders_content_compliant_minus_other"
            )
            & associations["outcome"].eq(
                "order_consistent_target_adoption"
            )
        ].iloc[0]
        self.assertAlmostEqual(compliant["estimate"], 1.0)

    def test_enforced_structure_pair_metrics_require_order_agreement(
        self,
    ) -> None:
        traces = []
        for pair_id in ("p1", "p2"):
            for condition in ("free_long", "enforced_criterion"):
                for order in ("original", "swapped"):
                    correct = condition == "enforced_criterion"
                    choice = (
                        "A"
                        if correct or order == "original"
                        else "B"
                    )
                    traces.append(
                        {
                            "trace_id": (
                                f"{pair_id}-{condition}-{order}"
                            ),
                            "pair_id": pair_id,
                            "condition_id": condition,
                            "transition_type": "choice_to_choice",
                            "presentation_order": order,
                            "branch_index": 0,
                            "target_semantic": "A",
                            "valid_choice": True,
                            "final_target_semantic_selected": correct,
                            "analysis_budget_saturation_rate": 0.0,
                            "analysis_generated_tokens": 1024,
                            "verdict_budget_saturated": False,
                            "decision_checkpoint": {
                                "forced_choice_semantic": choice,
                                "forced_choice_confidence": 0.8,
                                "forced_target_semantic_selected": (
                                    choice == "A"
                                ),
                            },
                        }
                    )
        pairs = enforced_pair_metrics(enforced_trace_rows(traces))
        enforced = pairs[
            pairs["condition_id"].eq("enforced_criterion")
        ]
        free = pairs[pairs["condition_id"].eq("free_long")]
        self.assertTrue(
            enforced["order_consistent_target_adoption"].eq(1.0).all()
        )
        self.assertTrue(
            free["order_consistent_target_adoption"].eq(0.0).all()
        )
        self.assertTrue(free["order_consistent_rate"].eq(0.0).all())

    def test_pair_cross_fit_keeps_branches_together(self) -> None:
        rows = []
        states = []
        for pair_index in range(30):
            split = (
                "fit"
                if pair_index < 18
                else "selection"
                if pair_index < 24
                else "intervention"
            )
            label = "A" if pair_index % 2 == 0 else "B"
            value = 2.0 if label == "A" else -2.0
            for branch_index in range(2):
                rows.append(
                    {
                        "trace_id": f"{pair_index}-{branch_index}",
                        "pair_id": f"pair-{pair_index}",
                        "analysis_split": split,
                        "point_names": ["p0", "p1"],
                        "point_forced_choices_semantic": [label, label],
                        "condition_id": "switch",
                        "transition_type": "choice_to_choice",
                        "presentation_order": "original",
                        "branch_index": branch_index,
                    }
                )
                states.append(
                    [
                        [value, float(branch_index)],
                        [value * 1.5, float(branch_index)],
                    ]
                )
        frame = pd.DataFrame(rows)
        point_mask = np.ones((len(frame), 2), dtype=bool)
        layer_states = {4: np.asarray(states, dtype=np.float32)}
        _candidates, selected = _candidate_selection(
            frame=frame,
            point_mask=point_mask,
            layer_states=layer_states,
            targets=["current_choice"],
            c_values=[0.1],
            fit_split="fit",
            selection_split="selection",
            min_fit_rows=4,
            min_selection_rows=2,
            seed=1234,
        )
        predictions = _cross_fitted_predictions(
            frame=frame,
            point_mask=point_mask,
            layer_states=layer_states,
            selected=selected,
            estimation_splits={"fit", "intervention"},
            cv_folds=3,
            min_fold_train_rows=4,
            seed=1234,
        )
        self.assertFalse(predictions.empty)
        self.assertTrue(
            predictions.groupby("pair_id")["cv_fold"].nunique().eq(1).all()
        )

    def test_patching_pairs_preserve_placebo_control(self) -> None:
        common = {
            "pair_id": "pair",
            "presentation_order": "original",
            "branch_index": 0,
            "analysis_split": "intervention",
        }
        rows = [
            {**common, "condition_id": condition, "trace_id": condition}
            for condition in ("reminder", "switch", "placebo")
        ]
        pairs = _paired(rows, analysis_split="intervention")
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["placebo"]["trace_id"], "placebo")


if __name__ == "__main__":
    unittest.main()
