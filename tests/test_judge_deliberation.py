import unittest

import numpy as np

from aisafety.mech.judge_reasoning import comparison_prompt_content
from aisafety.scripts.analyze_judge_reasoning_budget_sweep import (
    analyze_budget_sweep,
)
from aisafety.scripts.analyze_judge_reasoning_fixed_decoders import (
    select_anchor_decoder,
)
from aisafety.scripts.build_judge_deliberation_source_pack import (
    build_bbh_logical_pairs,
    build_criterion_variants,
    build_gsm8k_pairs,
    build_math500_pairs,
    perturb_answer,
)


class JudgeDeliberationSourceTests(unittest.TestCase):
    def test_criterion_text_is_rendered_as_decision_rule(self) -> None:
        prompt = comparison_prompt_content(
            {
                "comparison_dimension": "correctness",
                "prompt": "Question",
                "option_a_text": "A response",
                "option_b_text": "B response",
                "metadata": {
                    "criterion_text": "Ignore style and judge factual correctness."
                },
            },
            reasoning_mode="thinking",
        )
        self.assertIn("Decision rule:", prompt)
        self.assertIn("Ignore style", prompt)

    def test_numeric_perturbation(self) -> None:
        self.assertEqual(perturb_answer("41"), "42")
        self.assertEqual(perturb_answer("3/5"), "4/5")
        self.assertEqual(perturb_answer("x = 7"), "x = 8")

    def test_gsm8k_corrupts_only_final_answer(self) -> None:
        rows = [
            {
                "question": "What is 20 plus 21?",
                "answer": "Add the values. 20 + 21 = 41.\n#### 41",
            }
        ]
        pairs = build_gsm8k_pairs(rows, seed=1234)
        self.assertEqual(len(pairs), 1)
        self.assertIn("#### 41", pairs[0]["option_a_text"])
        self.assertIn("#### 42", pairs[0]["option_b_text"])
        self.assertEqual(pairs[0]["target_option"], "A")
        self.assertEqual(pairs[0]["criterion_determinacy"], "exact")

    def test_math500_corrupts_boxed_answer(self) -> None:
        rows = [
            {
                "problem": "Compute 6 times 7.",
                "solution": "Multiplication gives $6\\cdot7=42$, so $\\boxed{42}$.",
                "subject": "Algebra",
                "level": "1",
            }
        ]
        pairs = build_math500_pairs(rows, seed=1234)
        self.assertEqual(len(pairs), 1)
        self.assertIn(r"\boxed{42}", pairs[0]["option_a_text"])
        self.assertIn(r"\boxed{43}", pairs[0]["option_b_text"])

    def test_bbh_extracts_correct_and_distractor_choices(self) -> None:
        rows = [
            {
                "input": (
                    "Ana is left of Bo. Which ordering is valid?\n"
                    "(A) Ana, Bo\n(B) Bo, Ana\n(C) They are tied"
                ),
                "target": "(A)",
            }
        ]
        pairs = build_bbh_logical_pairs(
            rows,
            subset="logical_deduction_three_objects",
            seed=1234,
        )
        self.assertEqual(len(pairs), 1)
        self.assertTrue(pairs[0]["option_a_text"].startswith("(A)"))
        self.assertEqual(pairs[0]["target_option"], "A")

    def test_helpsteer_criterion_variants_assign_proxy_targets(self) -> None:
        row = {
            "pair_id": "pair",
            "prompt": "Explain.",
            "option_a_text": "A",
            "option_b_text": "B",
            "target_option": "",
            "target_kind": "none",
            "option_a_attributes": {
                "helpfulness": 4,
                "correctness": 2,
                "coherence": 4,
                "complexity": 2,
                "verbosity": 2,
            },
            "option_b_attributes": {
                "helpfulness": 2,
                "correctness": 4,
                "coherence": 2,
                "complexity": 2,
                "verbosity": 2,
            },
        }
        variants = build_criterion_variants(
            [row],
            source_prefix="helpsteer2_tradeoff",
            criteria=["overall", "correctness", "helpfulness", "weighted"],
            seed=1234,
        )
        self.assertEqual(variants["overall"][0]["target_option"], "")
        self.assertEqual(variants["correctness"][0]["target_option"], "B")
        self.assertEqual(variants["helpfulness"][0]["target_option"], "A")
        self.assertEqual(variants["weighted"][0]["target_option"], "A")


def _score_row(
    *,
    pair: str,
    comparison: str,
    order: str,
    mode: str,
    branch: int,
    budget: int,
    choice: str,
    target: str = "A",
) -> dict:
    option_a = "correct"
    option_b = "wrong"
    if order == "swapped":
        option_a, option_b = option_b, option_a
        target = "B"
    selected = option_a if choice == "A" else option_b
    return {
        "budget_eval_id": f"{pair}:{comparison}:{mode}:{branch}:{budget}",
        "trace_id": f"{pair}:{comparison}:{mode}:{branch}",
        "pair_id": pair,
        "origin_pair_id": pair,
        "comparison_id": comparison,
        "source_dataset": "arc_challenge",
        "criterion_id": "factual_correctness",
        "criterion_family": "",
        "criterion_determinacy": "explicit_gold",
        "determinacy_level": "factual_explicit",
        "presentation_order": order,
        "reasoning_mode": mode,
        "branch_index": branch,
        "budget_tokens": budget,
        "forced_choice": choice,
        "target_option": target,
        "forced_selected_text_hash": selected,
        "forced_choice_confidence": 0.8,
        "forced_margin_a_minus_b": 2.0 if choice == "A" else -2.0,
        "natural_valid_at_budget": budget >= 128 or mode == "direct",
        "full_generated_tokens": 256,
        "max_budget_saturated": False,
    }


class JudgeDeliberationAnalysisTests(unittest.TestCase):
    def test_budget_analysis_tracks_direct_to_thinking_correction(self) -> None:
        rows = []
        for order, comparison in (("original", "p-original"), ("swapped", "p-swapped")):
            rows.append(
                _score_row(
                    pair="p",
                    comparison=comparison,
                    order=order,
                    mode="direct",
                    branch=-1,
                    budget=0,
                    choice="B" if order == "original" else "A",
                )
            )
            for branch in range(3):
                for budget in (0, 128, 256):
                    corrected = budget >= 128
                    choice = (
                        ("A" if order == "original" else "B")
                        if corrected
                        else ("B" if order == "original" else "A")
                    )
                    rows.append(
                        _score_row(
                            pair="p",
                            comparison=comparison,
                            order=order,
                            mode="thinking",
                            branch=branch,
                            budget=budget,
                            choice=choice,
                        )
                    )
        outputs = analyze_budget_sweep(rows, bootstrap=20, seed=1234)
        effects = outputs["budget_pair_effects"]
        final = effects[effects["budget_tokens"].eq(256)].iloc[0]
        self.assertGreater(
            float(final["delta_mean_majority_target_selected"]),
            0.0,
        )
        self.assertFalse(outputs["bootstrap_budget_effects"].empty)

    def test_fixed_decoder_selection_prefers_signal_layer(self) -> None:
        rng = np.random.default_rng(1234)
        y = np.asarray([0, 1] * 40, dtype=int)
        fit_mask = np.zeros(80, dtype=bool)
        fit_mask[:50] = True
        selection_mask = ~fit_mask
        noise = rng.normal(size=(80, 4)).astype(np.float32)
        signal = noise.copy()
        signal[:, 0] += (2 * y - 1) * 3.0
        table, selected = select_anchor_decoder(
            layer_endpoints={4: noise, 8: signal},
            y=y,
            fit_mask=fit_mask,
            selection_mask=selection_mask,
            c_values=[0.1, 1.0],
            seed=1234,
        )
        self.assertFalse(table.empty)
        self.assertEqual(int(selected["hidden_layer"]), 8)


if __name__ == "__main__":
    unittest.main()
