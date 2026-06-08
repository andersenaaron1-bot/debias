import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from aisafety.mech.judge_reasoning import (
    OneShotLastTokenSteeringHook,
    direction_angle_degrees,
    first_persistent_threshold,
    parse_final_choice,
    random_orthogonal_direction,
    resample_trajectory,
)
from aisafety.scripts.analyze_judge_reasoning_trajectories import (
    TraceArtifact,
    _binary_target,
    analyze,
    decision_dynamics_rows,
    descriptive_mediation_row,
    grouped_probe,
    grouped_probe_oof,
)
from aisafety.scripts.build_judge_reasoning_pairs import build_comparisons
from aisafety.scripts.build_judge_reasoning_suite import build_suite
from aisafety.scripts.run_judge_reasoning_interventions import select_probe_direction
from aisafety.scripts.run_judge_reasoning_trajectories import (
    TraceShardWriter,
    _cap_comparisons,
)


class JudgeReasoningSuiteTests(unittest.TestCase):
    def test_preference_pairs_preserve_targets_and_swap_order(self) -> None:
        rows = [
            {
                "pair_id": "pair-1",
                "prompt": "Choose.",
                "chosen": "Preferred answer.",
                "rejected": "Rejected answer.",
                "source_dataset": "prefs",
                "extra_field": "kept",
            }
        ]
        comparisons = build_comparisons(
            rows,
            input_format="preference",
            source_label="prefs",
            task_type="ordered_preference",
            comparison_dimension="human_preference",
        )
        self.assertEqual(len(comparisons), 2)
        self.assertEqual(comparisons[0]["target_option"], "A")
        self.assertEqual(comparisons[1]["target_option"], "B")
        self.assertEqual(comparisons[0]["target_kind"], "preference")
        self.assertEqual(comparisons[0]["presentation_order"], "original")
        self.assertEqual(comparisons[1]["presentation_order"], "swapped")
        self.assertEqual(comparisons[1]["metadata"]["extra_field"], "kept")

    def test_binary_items_build_non_ordered_moral_comparisons(self) -> None:
        rows = [
            {"input": "Return a lost wallet.", "label": 1, "category": "commonsense"},
            {"input": "Keep a lost wallet.", "label": 0, "category": "commonsense"},
        ]
        comparisons = build_comparisons(
            rows,
            input_format="binary_items",
            source_label="moral",
            task_type="non_ordered_attribute",
            comparison_dimension="moral_acceptability",
            binary_group_key="category",
            binary_question="Which action is more morally acceptable?",
        )
        self.assertEqual(len(comparisons), 2)
        self.assertEqual(comparisons[0]["target_option"], "A")
        self.assertEqual(comparisons[1]["target_option"], "B")
        self.assertEqual(comparisons[0]["target_kind"], "objective")
        self.assertEqual(comparisons[0]["presentation_order"], "positive_first")
        self.assertEqual(comparisons[1]["presentation_order"], "negative_first")

    def test_multi_dataset_suite_skips_optional_missing_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "prefs.jsonl"
            source.write_text(
                json.dumps({"pair_id": "p", "chosen": "Good", "rejected": "Bad"}) + "\n",
                encoding="utf-8",
            )
            config = {
                "datasets": [
                    {
                        "dataset_id": "prefs",
                        "input_jsonl": "prefs.jsonl",
                        "input_format": "preference",
                        "comparison_dimension": "human_preference",
                        "validity_type": "preference",
                        "difficulty_tier": "mixed",
                    },
                    {
                        "dataset_id": "optional",
                        "input_jsonl": "missing.jsonl",
                        "input_format": "binary_items",
                        "required": False,
                    },
                ]
            }
            comparisons, datasets = build_suite(
                config,
                workspace_root=root,
                input_root=root,
                skip_missing=False,
                max_pairs_per_dataset=0,
                seed=1234,
            )
            self.assertEqual(len(comparisons), 2)
            self.assertEqual([row["status"] for row in datasets], ["ok", "missing"])
            self.assertEqual(comparisons[0]["metadata"]["validity_type"], "preference")
            self.assertEqual(comparisons[0]["metadata"]["difficulty_tier"], "mixed")

    def test_choice_parser_requires_explicit_tail_verdict(self) -> None:
        self.assertEqual(parse_final_choice("Option A is stronger.\nFINAL: B"), "B")
        self.assertEqual(parse_final_choice("Reasoning only mentions A and B."), "")
        self.assertEqual(parse_final_choice("analysis\nA"), "A")

    def test_trajectory_resampling_and_commitment(self) -> None:
        states = np.arange(7 * 2 * 3, dtype=np.float32).reshape(7, 2, 3)
        sampled, indices, positions = resample_trajectory(states, n_points=4)
        self.assertEqual(sampled.shape, (4, 2, 3))
        self.assertEqual(indices.tolist(), [0, 2, 4, 6])
        self.assertTrue(np.allclose(positions, [0.0, 1 / 3, 2 / 3, 1.0]))
        self.assertEqual(
            first_persistent_threshold([0.5, 0.76, 0.8, 0.9], threshold=0.75, persistence=3),
            1,
        )

    def test_trajectory_cap_round_robins_across_sources(self) -> None:
        rows = [
            {"pair_id": f"a{index}", "source_dataset": "a"}
            for index in range(5)
        ] + [
            {"pair_id": f"b{index}", "source_dataset": "b"}
            for index in range(5)
        ]
        selected = _cap_comparisons(
            rows,
            max_pairs=4,
            seed=1234,
            strategy="source_round_robin",
        )
        counts = pd.Series([row["source_dataset"] for row in selected]).value_counts()
        self.assertEqual(counts.to_dict(), {"a": 2, "b": 2})

    def test_random_control_is_unit_and_orthogonal(self) -> None:
        direction = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
        control = random_orthogonal_direction(direction, seed=2101)
        self.assertAlmostEqual(float(np.linalg.norm(control)), 1.0, places=6)
        self.assertAlmostEqual(float(np.dot(control, direction)), 0.0, places=5)
        self.assertAlmostEqual(
            float(direction_angle_degrees(direction, control)),
            90.0,
            places=4,
        )

    def test_one_shot_hook_edits_only_first_call_and_preserves_tuple(self) -> None:
        hook = OneShotLastTokenSteeringHook(np.asarray([1.0, -1.0]), alpha=2.0)
        hidden = torch.zeros((1, 3, 2))
        first = hook(None, (), (hidden, "cache"))
        second = hook(None, (), (hidden, "cache"))
        self.assertTrue(torch.equal(first[0][0, -1], torch.tensor([2.0, -2.0])))
        self.assertEqual(first[1], "cache")
        self.assertTrue(torch.equal(second[0], hidden))

    def test_trace_shard_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            writer = TraceShardWriter(
                root,
                n_points=3,
                hidden_layers=[1, 2],
                shard_size=2,
                compress=False,
            )
            states = np.ones((2, 2, 4), dtype=np.float32)
            shard, row_index = writer.add(
                states=states,
                step_indices=np.asarray([0, 3]),
                positions=np.asarray([0.0, 1.0], dtype=np.float32),
                label_margins=np.asarray([0.2, -0.1], dtype=np.float32),
            )
            writer.flush()
            trace = {
                "trace_id": "t1",
                "pair_id": "p1",
                "comparison_id": "c1",
                "reasoning_mode": "thinking",
                "comparison_dimension": "quality",
                "trajectory_shard": shard,
                "trajectory_shard_row": row_index,
            }
            (root / "traces.jsonl").write_text(json.dumps(trace) + "\n", encoding="utf-8")
            artifact = TraceArtifact(root)
            self.assertEqual(artifact.hidden_layers, [1, 2])
            self.assertEqual(artifact.step_indices[0].tolist(), [0, 3, -1])
            self.assertEqual(artifact.layer_states(2).shape, (1, 3, 4))
            self.assertTrue(np.allclose(artifact.layer_states(2)[0, :2], 1.0))

    def test_grouped_probe_and_descriptive_mediation(self) -> None:
        rng = np.random.default_rng(1234)
        groups = [f"g{index}" for index in range(60)]
        y = np.asarray([index % 2 for index in range(60)], dtype=int)
        x = rng.normal(size=(60, 8)).astype(np.float32)
        x[:, 0] += (2 * y - 1) * 3.0
        metrics, direction = grouped_probe(
            x,
            y,
            groups,
            folds=5,
            c_value=0.1,
            seed=1234,
            salt="test",
        )
        self.assertEqual(metrics["status"], "ok")
        self.assertGreater(metrics["roc_auc"], 0.9)
        self.assertIsNotNone(direction)
        outcome = y.copy()
        mediation = descriptive_mediation_row(
            x=x,
            direction=direction,
            mediator=y,
            outcome=outcome,
        )
        self.assertIsNotNone(mediation)
        self.assertGreater(mediation["condition_projection_gap"], 0.0)

    def test_grouped_probe_returns_cross_fitted_trace_probabilities(self) -> None:
        rng = np.random.default_rng(8)
        groups = [f"g{index // 2}" for index in range(80)]
        y = np.asarray([(index // 2) % 2 for index in range(80)], dtype=int)
        x = rng.normal(scale=0.2, size=(80, 5)).astype(np.float32)
        x[:, 0] += (2 * y - 1) * 2.0
        metrics, direction, probabilities, fold_ids = grouped_probe_oof(
            x,
            y,
            groups,
            folds=5,
            c_value=0.1,
            seed=1234,
            salt="stable",
        )
        self.assertEqual(metrics["status"], "ok")
        self.assertIsNotNone(direction)
        self.assertTrue(np.isfinite(probabilities).all())
        for group in set(groups):
            indices = [index for index, value in enumerate(groups) if value == group]
            self.assertEqual(len(set(fold_ids[indices].tolist())), 1)

    def test_decision_dynamics_uses_token_timing_and_target_gap(self) -> None:
        rows = []
        for point, token in enumerate([0, 4, 9]):
            common = {
                "trace_id": "t1",
                "pair_id": "p1",
                "comparison_id": "c1",
                "run_label": "model",
                "model_id": "model-id",
                "reasoning_mode": "thinking",
                "source_dataset": "arc_easy",
                "comparison_dimension": "factual_correctness",
                "task_type": "ordered_factual_judgment",
                "target_kind": "objective",
                "target_selected": True,
                "target_option": "A",
                "final_choice": "A",
                "valid_choice": True,
                "analysis_group": "all",
                "analysis_group_type": "all",
                "analysis_group_value": "all",
                "hidden_layer": 8,
                "point_index": point,
                "position": point / 2,
                "generated_tokens_before_state": token,
            }
            rows.append(
                {
                    **common,
                    "probe_target": "final_choice",
                    "prob_positive": [0.55, 0.92, 0.95][point],
                }
            )
            rows.append(
                {
                    **common,
                    "probe_target": "target_option",
                    "prob_positive": [0.52, 0.65, 0.90][point],
                }
            )
        dynamics = decision_dynamics_rows(
            pd.DataFrame(rows),
            choice_confidence_threshold=0.8,
            target_confidence_threshold=0.8,
            persistence=1,
        )
        self.assertEqual(len(dynamics), 1)
        row = dynamics.iloc[0]
        self.assertEqual(row["choice_commitment_generated_tokens"], 4.0)
        self.assertEqual(row["target_emergence_generated_tokens"], 9.0)
        self.assertEqual(row["shortcut_gap_tokens"], 5.0)
        self.assertTrue(bool(row["premature_commitment"]))

    def test_binary_target_is_encoded_within_analysis_scope(self) -> None:
        frame = pd.DataFrame(
            {
                "presentation_order": [
                    "original",
                    "swapped",
                    "positive_first",
                    "negative_first",
                ]
            }
        )
        encoded = _binary_target(
            frame,
            target="presentation_order",
            positive_condition_label="",
            scope=np.asarray([False, False, True, True]),
        )
        self.assertIsNotNone(encoded)
        valid, values, positive = encoded
        self.assertEqual(valid.tolist(), [False, False, True, True])
        self.assertEqual(positive, "positive_first")
        self.assertEqual(values.tolist(), [0, 0, 1, 0])

    def test_target_selected_handles_missing_values_without_coercion(self) -> None:
        frame = pd.DataFrame({"target_selected": [True, False, None]})
        encoded = _binary_target(
            frame,
            target="target_selected",
            positive_condition_label="",
        )
        self.assertIsNotNone(encoded)
        valid, values, positive = encoded
        self.assertEqual(valid.tolist(), [True, True, False])
        self.assertEqual(values.tolist(), [1, 0, 0])
        self.assertEqual(positive, "true")

    def test_probe_direction_selection_uses_best_matching_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = pd.DataFrame(
                [
                    {
                        "status": "ok",
                        "reasoning_mode": "thinking",
                        "analysis_group": "all",
                        "probe_target": "final_choice",
                        "positive_label": "A",
                        "hidden_layer": 8,
                        "point_index": 3,
                        "mean_position": 0.5,
                        "roc_auc": 0.8,
                        "balanced_accuracy": 0.7,
                    },
                    {
                        "status": "ok",
                        "reasoning_mode": "thinking",
                        "analysis_group": "all",
                        "probe_target": "final_choice",
                        "positive_label": "A",
                        "hidden_layer": 16,
                        "point_index": 5,
                        "mean_position": 0.75,
                        "roc_auc": 0.9,
                        "balanced_accuracy": 0.8,
                    },
                ]
            )
            metrics.to_csv(root / "probe_metrics.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "reasoning_mode": row.reasoning_mode,
                        "analysis_group": row.analysis_group,
                        "probe_target": row.probe_target,
                        "positive_label": row.positive_label,
                        "hidden_layer": row.hidden_layer,
                        "point_index": row.point_index,
                        "mean_position": row.mean_position,
                        "direction_index": index,
                    }
                    for index, row in metrics.iterrows()
                ]
            ).to_csv(root / "probe_direction_index.csv", index=False)
            np.savez(
                root / "probe_directions.npz",
                directions=np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32),
            )
            direction, metadata = select_probe_direction(
                root,
                reasoning_mode="thinking",
                analysis_group="all",
                probe_target="final_choice",
                hidden_layer=0,
                point_index=-1,
            )
            self.assertEqual(metadata["hidden_layer"], 16)
            self.assertTrue(np.allclose(direction, [0.0, 1.0]))

    def test_end_to_end_synthetic_analysis_emits_core_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            writer = TraceShardWriter(
                root,
                n_points=3,
                hidden_layers=[1],
                shard_size=16,
                compress=False,
            )
            rows = []
            rng = np.random.default_rng(7)
            for index in range(80):
                label = index % 2
                signal = 1.0 if label else -1.0
                states = rng.normal(scale=0.05, size=(3, 1, 6)).astype(np.float32)
                states[:, 0, 0] += np.asarray([0.1, 1.0, 3.0]) * signal
                shard, shard_row = writer.add(
                    states=states,
                    step_indices=np.asarray([0, 2, 4]),
                    positions=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
                    label_margins=np.asarray([0.0, signal, 2.0 * signal], dtype=np.float32),
                )
                rows.append(
                    {
                        "trace_id": f"t{index}",
                        "pair_id": f"p{index}",
                        "comparison_id": f"c{index}",
                        "reasoning_mode": "thinking",
                        "comparison_dimension": "moral_acceptability",
                        "condition_label": "cue_plus" if label else "cue_minus",
                        "presentation_order": "original" if label else "swapped",
                        "final_choice": "A" if label else "B",
                        "target_option": "A" if label else "B",
                        "target_selected": bool(label),
                        "source_dataset": "synthetic",
                        "metadata": {
                            "validity_type": "objective",
                            "difficulty_tier": "easy",
                            "analysis_split": "fit",
                        },
                        "trajectory_shard": shard,
                        "trajectory_shard_row": shard_row,
                    }
                )
            writer.flush()
            (root / "traces.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            outputs = analyze(
                TraceArtifact(root),
                probe_targets=[
                    "final_choice",
                    "target_option",
                    "condition_label",
                    "presentation_order",
                ],
                group_columns=[
                    "comparison_dimension",
                    "source_dataset",
                    "validity_type",
                    "difficulty_tier",
                    "analysis_split",
                ],
                positive_condition_label="cue_plus",
                cv_folds=5,
                probe_c=0.1,
                min_probe_rows=20,
                min_dimension_rows=20,
                commitment_auc=0.75,
                commitment_persistence=2,
                seed=1234,
            )
            self.assertFalse(outputs["probe_metrics"].empty)
            self.assertFalse(outputs["commitment_summary"].empty)
            self.assertFalse(outputs["trajectory_mediation_screen"].empty)
            self.assertFalse(outputs["probe_oof_predictions"].empty)
            self.assertFalse(outputs["decision_dynamics"].empty)
            self.assertFalse(outputs["decision_dynamics_summary"].empty)
            self.assertFalse(outputs["trajectory_geometry"].empty)
            self.assertGreater(
                float(outputs["probe_metrics"].query("status == 'ok'")["roc_auc"].max()),
                0.9,
            )


if __name__ == "__main__":
    unittest.main()
