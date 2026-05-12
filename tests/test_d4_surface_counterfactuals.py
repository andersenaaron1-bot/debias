import unittest

import pandas as pd

from aisafety.mech.counterfactuals import (
    CounterfactualVariant,
    answer_likeness_decrease,
    answer_likeness_increase,
    build_counterfactual_variants,
    structured_decrease,
    structured_increase,
)
from aisafety.scripts.build_d4_surface_counterfactual_pairs import build_counterfactual_rows


class D4SurfaceCounterfactualTests(unittest.TestCase):
    def test_structured_increase_and_decrease_round_trip_shape(self) -> None:
        text = (
            "The answer explains the main tradeoff. It then gives a concrete example. "
            "Finally, it states the practical implication."
        )
        increased, flags, reason = structured_increase(text)
        self.assertIsNone(reason)
        self.assertIn("- The answer explains", increased)
        self.assertIn("added_list_structure", flags)

        decreased, flags, reason = structured_decrease(increased)
        self.assertIsNone(reason)
        self.assertNotIn("- ", decreased)
        self.assertIn("paragraphized", flags)

    def test_structured_decrease_handles_labels_and_numbering(self) -> None:
        text = (
            "Answer:\n"
            "1. The first point explains the main tradeoff.\n"
            "2. The second point gives a concrete implication."
        )
        decreased, flags, reason = structured_decrease(text)
        self.assertIsNone(reason)
        self.assertNotIn("Answer:", decreased)
        self.assertNotIn("1.", decreased)
        self.assertIn("main tradeoff", decreased)
        self.assertIn("removed_assistant_packaging", flags)

    def test_answer_likeness_increase_and_decrease_are_content_preserving_shape(self) -> None:
        text = (
            "The answer states the conclusion. It explains why the conclusion follows. "
            "It adds a caveat."
        )
        increased, flags, reason = answer_likeness_increase(text)
        self.assertIsNone(reason)
        self.assertIn("Answer:", increased)
        self.assertIn("Details:", increased)
        self.assertIn("added_answer_frame", flags)

        decreased, flags, reason = answer_likeness_decrease(increased)
        self.assertIsNone(reason)
        self.assertNotIn("Answer:", decreased)
        self.assertNotIn("Details:", decreased)
        self.assertIn("conclusion", decreased)
        self.assertIn("removed_answer_frame", flags)

    def test_variant_builder_records_skip_and_valid_variant(self) -> None:
        outputs = build_counterfactual_variants(
            "This is a first sentence with enough detail. This is a second sentence with enough detail.",
            axes={"structured_assistant_packaging", "formal_institutional_packaging"},
            min_tokens=5,
        )
        self.assertTrue(any(isinstance(item, CounterfactualVariant) for item in outputs))
        skip_reasons = {getattr(item, "reason", "") for item in outputs}
        self.assertIn("no_formal_markers", skip_reasons)

    def test_build_counterfactual_rows_emits_stable_metadata(self) -> None:
        pair_df = pd.DataFrame(
            [
                {
                    "pair_id": "p1",
                    "source_dataset": "hc3",
                    "bundle_creation_role": "discovery_core",
                    "group_id": "g1",
                    "split": "test",
                    "item_type": "hc3",
                    "subset": "finance",
                    "title": "",
                    "question": "What is the tradeoff?",
                    "llm_generator": "gpt",
                    "human_text": (
                        "The human answer states the main point. It then adds one supporting detail. "
                        "It ends with a limitation."
                    ),
                    "llm_text": (
                        "The model answer states the main point. It then adds one supporting detail. "
                        "It ends with a limitation."
                    ),
                }
            ]
        )
        rows, summary = build_counterfactual_rows(
            pair_df,
            axes={"structured_assistant_packaging"},
            min_tokens=5,
            min_length_ratio=0.5,
            max_length_ratio=2.0,
        )
        self.assertGreaterEqual(len(rows), 2)
        first = rows[0]
        self.assertIn("counterfactual_id", first)
        self.assertEqual(first["source_dataset"], "hc3")
        self.assertEqual(first["axis"], "structured_assistant_packaging")
        self.assertIn(first["role"], {"human", "llm"})
        self.assertEqual(summary["emitted_by_role"]["human"], 1)
        self.assertEqual(summary["emitted_by_role"]["llm"], 1)


if __name__ == "__main__":
    unittest.main()
