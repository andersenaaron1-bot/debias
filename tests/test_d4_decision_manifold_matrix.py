import tempfile
import unittest
from pathlib import Path

import pandas as pd

from aisafety.scripts.build_d4_decision_manifold_matrix import build_matrix


class D4DecisionManifoldMatrixTests(unittest.TestCase):
    def test_builds_stage_template_and_interaction_features(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            stage_dir = root / "stage"
            stage_dir.mkdir()
            pd.DataFrame(
                [
                    {
                        "run_label": "base",
                        "pair_id": "p1",
                        "source_dataset": "laurito",
                        "item_type": "movie",
                        "mean_llm_margin": 0.25,
                    },
                    {
                        "run_label": "sft",
                        "pair_id": "p1",
                        "source_dataset": "laurito",
                        "item_type": "movie",
                        "mean_llm_margin": 0.75,
                    },
                ]
            ).to_csv(stage_dir / "stage_pair_summary_long.csv", index=False)

            template_dir = root / "template"
            template_dir.mkdir()
            pd.DataFrame(
                [
                    {
                        "run_label": "sft",
                        "pair_id": "p1",
                        "template_contrast": "standard_minus_minimal",
                        "template_delta_llm_margin": 0.5,
                    }
                ]
            ).to_csv(template_dir / "template_sensitivity_pair_deltas.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "pair_id": "p1",
                        "stage_contrast": "sft_minus_base",
                        "template_contrast": "standard_minus_minimal",
                        "stage_template_interaction_llm_margin": 0.4,
                    }
                ]
            ).to_csv(template_dir / "template_stage_interaction_pair_deltas.csv", index=False)

            long_df, wide_df, summary_df, manifest = build_matrix(
                [("stage_in", stage_dir), ("template_in", template_dir)],
                stage_contrasts=["sft_minus_base=sft-base"],
                prefix_input=False,
            )

            self.assertEqual(manifest["n_units"], 1)
            self.assertIn("stage_in__hllm_margin__base", set(long_df["feature_name"]))
            self.assertIn("stage_in__hllm_stage_delta__sft_minus_base", set(long_df["feature_name"]))
            self.assertIn(
                "template_in__hllm_stage_template_interaction__sft_minus_base__standard_minus_minimal",
                set(long_df["feature_name"]),
            )
            stage_delta = long_df[
                long_df["feature_name"] == "stage_in__hllm_stage_delta__sft_minus_base"
            ]["feature_value"].iloc[0]
            self.assertAlmostEqual(float(stage_delta), 0.5)
            self.assertEqual(len(wide_df), 1)
            self.assertGreaterEqual(len(summary_df), 4)


if __name__ == "__main__":
    unittest.main()
