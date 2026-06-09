import unittest

from aisafety.scripts.analyze_judge_reasoning_mode_contrasts import (
    analyze_mode_contrasts,
    commitment_threshold_sweep,
)


def _row(
    *,
    pair: str,
    comparison: str,
    order: str,
    mode: str,
    branch: int,
    choice: str,
    target: str,
    option_a: str,
    option_b: str,
    generated_tokens: int,
) -> dict:
    return {
        "pair_id": pair,
        "comparison_id": comparison,
        "source_dataset": "arc_easy",
        "presentation_order": order,
        "reasoning_mode": mode,
        "branch_index": branch,
        "final_choice": choice,
        "valid_choice": bool(choice),
        "target_option": target,
        "target_kind": "objective",
        "option_a_text": option_a,
        "option_b_text": option_b,
        "generated_tokens": generated_tokens,
        "response_text": f"Reasoning. FINAL: {choice}" if choice else "Reasoning only.",
        "metadata": {
            "validity_type": "objective",
            "difficulty_tier": "easy",
            "analysis_split": "fit",
        },
    }


class JudgeReasoningModeContrastTests(unittest.TestCase):
    def test_mode_analysis_tracks_corrections_order_and_budget(self) -> None:
        rows = []
        for branch in range(3):
            rows.extend(
                [
                    _row(
                        pair="p1",
                        comparison="p1-original",
                        order="original",
                        mode="direct",
                        branch=branch,
                        choice="B",
                        target="A",
                        option_a="correct",
                        option_b="wrong",
                        generated_tokens=8,
                    ),
                    _row(
                        pair="p1",
                        comparison="p1-swapped",
                        order="swapped",
                        mode="direct",
                        branch=branch,
                        choice="A",
                        target="B",
                        option_a="wrong",
                        option_b="correct",
                        generated_tokens=8,
                    ),
                    _row(
                        pair="p1",
                        comparison="p1-original",
                        order="original",
                        mode="thinking",
                        branch=branch,
                        choice="A" if branch < 2 else "",
                        target="A",
                        option_a="correct",
                        option_b="wrong",
                        generated_tokens=80 if branch < 2 else 100,
                    ),
                    _row(
                        pair="p1",
                        comparison="p1-swapped",
                        order="swapped",
                        mode="thinking",
                        branch=branch,
                        choice="B",
                        target="B",
                        option_a="wrong",
                        option_b="correct",
                        generated_tokens=80,
                    ),
                ]
            )
        outputs = analyze_mode_contrasts(
            rows,
            manifest={
                "max_new_tokens_direct": 16,
                "max_new_tokens_thinking": 100,
            },
            direct_mode="direct",
            deliberative_mode="thinking",
            bootstrap=50,
            seed=1234,
        )
        traces = outputs["trace_outcomes"]
        thinking = traces[traces["reasoning_mode"].eq("thinking")].iloc[0]
        self.assertAlmostEqual(float(thinking["valid_rate"]), 5 / 6)
        self.assertAlmostEqual(float(thinking["budget_saturation_rate"]), 1 / 6)

        matched = outputs["matched_comparison_summary"].iloc[0]
        self.assertGreater(float(matched["majority_wrong_to_correct_rate"]), 0.0)
        self.assertEqual(float(matched["majority_correct_to_wrong_rate"]), 0.0)

        order = outputs["order_invariance"]
        thinking_pair = order[order["reasoning_mode"].eq("thinking")].iloc[0]
        self.assertEqual(float(thinking_pair["order_consistent_majority"]), 1.0)

        effects = outputs["pair_mode_effects"].iloc[0]
        self.assertGreater(
            float(effects["delta_mean_trace_target_success_rate"]),
            0.0,
        )
        self.assertFalse(outputs["bootstrap_mode_effects"].empty)

    def test_commitment_threshold_sweep_tracks_thresholds(self) -> None:
        rows = []
        for point, probability in enumerate([0.55, 0.75, 0.9]):
            rows.append(
                {
                    "trace_id": "trace",
                    "pair_id": "pair",
                    "comparison_id": "comparison",
                    "run_label": "it",
                    "model_id": "model",
                    "reasoning_mode": "thinking",
                    "source_dataset": "arc_easy",
                    "comparison_dimension": "factual_correctness",
                    "task_type": "ordered_factual_judgment",
                    "target_kind": "objective",
                    "target_selected": True,
                    "target_option": "A",
                    "final_choice": "A",
                    "valid_choice": True,
                    "analysis_group": "source_dataset:arc_easy",
                    "analysis_group_type": "source_dataset",
                    "analysis_group_value": "arc_easy",
                    "hidden_layer": 8,
                    "point_index": point,
                    "position": point / 2,
                    "generated_tokens_before_state": point * 4,
                    "probe_target": "final_choice",
                    "prob_positive": probability,
                }
            )
        sweep = commitment_threshold_sweep(
            __import__("pandas").DataFrame(rows),
            thresholds=[0.4, 0.8],
            persistence=1,
        )
        self.assertEqual(sorted(sweep["confidence_threshold"].unique()), [0.4, 0.8])


if __name__ == "__main__":
    unittest.main()
