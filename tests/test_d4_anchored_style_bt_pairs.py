import unittest

from aisafety.scripts.build_d4_anchored_style_bt_pairs import build_anchored_bt_rows
from aisafety.scripts.build_d4_fixed_reference_bt_controls import build_fixed_reference_rows


PAIR = {
    "pair_id": "p1",
    "source_dataset": "hc3",
    "subset": "finance",
    "split": "test",
    "item_type": "qa",
    "question": "What is the tradeoff?",
    "human_text": "Original human answer.",
    "llm_text": "Original model answer.",
}


class D4AnchoredStyleBTPairTests(unittest.TestCase):
    def test_generated_human_rewrite_uses_original_llm_reference(self) -> None:
        rows, summary = build_anchored_bt_rows(
            [
                {
                    "counterfactual_id": "cf1",
                    "pair_id": "p1",
                    "role": "human",
                    "axis": "generated_assistant_style",
                    "direction": "increase",
                    "transform_id": "generated_style",
                    "prompt": "What is the tradeoff?",
                    "base_text": "Plain rewritten human answer.",
                    "variant_text": "Overall, styled rewritten human answer.",
                }
            ],
            [PAIR],
            include_order_swaps=True,
        )
        self.assertEqual(summary["n_counterfactuals_used"], 1)
        self.assertEqual(len(rows), 2)
        first = next(row for row in rows if row["presentation_order"] == "candidate_first")
        swapped = next(row for row in rows if row["presentation_order"] == "reference_first")
        self.assertEqual(first["reference_text"], "Original model answer.")
        self.assertEqual(first["option_a_text"], "Overall, styled rewritten human answer.")
        self.assertEqual(first["option_b_text"], "Original model answer.")
        self.assertEqual(first["cue_minus_text"], "Plain rewritten human answer.")
        self.assertEqual(swapped["cue_plus_option"], "B")

    def test_llm_rewrite_uses_original_human_reference(self) -> None:
        rows, _ = build_anchored_bt_rows(
            [
                {
                    "counterfactual_id": "cf2",
                    "pair_id": "p1",
                    "role": "llm",
                    "axis": "assistant_answer_label",
                    "direction": "increase",
                    "base_text": "Original model answer.",
                    "variant_text": "Answer: Original model answer.",
                }
            ],
            [PAIR],
            include_order_swaps=False,
        )
        self.assertEqual(rows[0]["reference_text"], "Original human answer.")

    def test_hllm_retention_control_keeps_candidate_when_neutralized(self) -> None:
        rows, summary = build_fixed_reference_rows(
            [PAIR],
            mode="human_llm",
            max_pairs=0,
            include_order_swaps=True,
            seed=1234,
        )
        self.assertEqual(summary["n_pairs_used"], 1)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["cue_plus_text"] == row["cue_minus_text"] for row in rows))
        self.assertTrue(all(row["reference_text"] == "Original human answer." for row in rows))

    def test_preference_retention_control_uses_chosen_as_candidate(self) -> None:
        rows, _ = build_fixed_reference_rows(
            [{"pair_id": "pref1", "prompt": "Choose.", "chosen": "Better.", "rejected": "Worse."}],
            mode="preference",
            max_pairs=0,
            include_order_swaps=False,
            seed=1234,
        )
        self.assertEqual(rows[0]["cue_plus_text"], "Better.")
        self.assertEqual(rows[0]["reference_text"], "Worse.")


if __name__ == "__main__":
    unittest.main()
