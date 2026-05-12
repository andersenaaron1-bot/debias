import unittest

from aisafety.scripts.build_d4_bt_stage_contrast_pairs import build_bt_rows


class D4BTStageContrastTests(unittest.TestCase):
    def test_build_bt_rows_maps_increase_to_cue_plus_variant_and_swaps_order(self) -> None:
        rows, summary = build_bt_rows(
            [
                {
                    "counterfactual_id": "cf1",
                    "pair_id": "p1",
                    "source_dataset": "hc3",
                    "subset": "finance",
                    "split": "test",
                    "item_type": "hc3",
                    "role": "llm",
                    "axis": "answer_likeness_packaging",
                    "direction": "increase",
                    "transform_id": "answer_likeness_scaffold_v1",
                    "prompt": "What is the answer?",
                    "base_text": "The conclusion follows from the facts.",
                    "variant_text": "Answer: The conclusion follows from the facts.",
                }
            ],
            include_order_swaps=True,
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(summary["n_counterfactuals_used"], 1)
        plus_first = next(row for row in rows if row["presentation_order"] == "plus_first")
        minus_first = next(row for row in rows if row["presentation_order"] == "minus_first")
        self.assertEqual(plus_first["cue_plus_option"], "A")
        self.assertEqual(minus_first["cue_plus_option"], "B")
        self.assertIn("Answer:", plus_first["cue_plus_text"])
        self.assertNotIn("Answer:", plus_first["cue_minus_text"])

    def test_build_bt_rows_maps_decrease_to_cue_plus_base(self) -> None:
        rows, _ = build_bt_rows(
            [
                {
                    "counterfactual_id": "cf2",
                    "pair_id": "p2",
                    "source_dataset": "hc3",
                    "role": "human",
                    "axis": "structured_assistant_packaging",
                    "direction": "decrease",
                    "transform_id": "structured_paragraphize_v1",
                    "base_text": "Key points: first answer. second answer.",
                    "variant_text": "first answer. second answer.",
                }
            ],
            include_order_swaps=False,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["cue_plus_source"], "base")
        self.assertEqual(rows[0]["cue_plus_option"], "A")
        self.assertIn("Key points", rows[0]["cue_plus_text"])


if __name__ == "__main__":
    unittest.main()
