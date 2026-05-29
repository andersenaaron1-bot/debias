import unittest

import pandas as pd

from aisafety.scripts.export_d4_decision_manifold_top_texts import export_top_texts


class D4DecisionManifoldTopTextsTests(unittest.TestCase):
    def test_exports_counterfactual_texts(self) -> None:
        top = pd.DataFrame(
            [
                {
                    "method": "pca",
                    "component": "pc2",
                    "unit_id": "cf1",
                    "score": 2.0,
                    "abs_score": 2.0,
                    "source_dataset": "hc3",
                    "item_type": "qa",
                    "axis": "answer_likeness_packaging",
                    "role": "llm",
                }
            ]
        )
        rows = [
            {
                "counterfactual_id": "cf1",
                "pair_id": "p1",
                "source_dataset": "hc3",
                "item_type": "qa",
                "axis": "answer_likeness_packaging",
                "direction": "increase",
                "role": "llm",
                "prompt": "Question?",
                "base_text": "Base answer.",
                "variant_text": "Direct answer: base answer.",
                "transform_id": "direct_answer",
            }
        ]
        out = export_top_texts(
            top,
            rows,
            method="pca",
            components=["pc2"],
            top_k=5,
            preview_chars=100,
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out["cue_plus_text"].iloc[0], "Direct answer: base answer.")
        self.assertEqual(out["cue_minus_text"].iloc[0], "Base answer.")


if __name__ == "__main__":
    unittest.main()
