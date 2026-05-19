import tempfile
import unittest
from pathlib import Path

import pandas as pd

from aisafety.scripts.summarize_d4_human_llm_template_sensitivity import (
    load_template_summary,
    stage_template_interaction_summary,
    template_delta_summary,
)


class D4HumanLlmTemplateSensitivityTests(unittest.TestCase):
    def test_template_delta_and_stage_interaction(self) -> None:
        rows = []
        for template, base, sft in (
            ("standard", 0.2, 0.8),
            ("minimal", 0.1, 0.2),
        ):
            for run_label, margin in (("base", base), ("sft", sft)):
                rows.append(
                    {
                        "template_label": template,
                        "run_label": run_label,
                        "pair_id": "p1",
                        "source_dataset": "hc3",
                        "subset": "all",
                        "item_type": "qa",
                        "scoring_mode": "forced_choice",
                        "mean_llm_margin": margin,
                        "llm_preferred": margin > 0.0,
                    }
                )
        pair_df = pd.DataFrame(rows)

        template_pairs, template_summary, _ = template_delta_summary(
            pair_df,
            ["standard_minus_minimal=standard-minimal"],
        )
        interactions, interaction_summary, _ = stage_template_interaction_summary(
            template_pairs,
            ["sft_minus_base=sft-base"],
        )

        base_delta = template_summary[template_summary["run_label"] == "base"]
        sft_delta = template_summary[template_summary["run_label"] == "sft"]
        self.assertAlmostEqual(float(base_delta["mean_template_delta_llm_margin"].iloc[0]), 0.1)
        self.assertAlmostEqual(float(sft_delta["mean_template_delta_llm_margin"].iloc[0]), 0.6)
        self.assertEqual(len(interactions), 1)
        self.assertAlmostEqual(float(interaction_summary["mean_interaction_llm_margin"].iloc[0]), 0.5)

    def test_load_template_summary_filters_likelihood_rows(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            pd.DataFrame(
                [
                    {
                        "run_label": "base",
                        "pair_id": "p1",
                        "source_dataset": "hc3",
                        "subset": "all",
                        "item_type": "qa",
                        "scoring_mode": "forced_choice",
                        "mean_llm_margin": 0.1,
                    },
                    {
                        "run_label": "base_like",
                        "pair_id": "p1",
                        "source_dataset": "hc3",
                        "subset": "all",
                        "item_type": "qa",
                        "scoring_mode": "response_likelihood",
                        "mean_llm_margin": 0.4,
                    },
                ]
            ).to_csv(out / "stage_pair_summary_long.csv", index=False)

            loaded = load_template_summary("standard", out)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded["template_label"].iloc[0], "standard")
            self.assertEqual(loaded["comparison_template"].iloc[0], "standard")


if __name__ == "__main__":
    unittest.main()
