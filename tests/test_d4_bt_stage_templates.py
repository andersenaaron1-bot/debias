import tempfile
import unittest
from pathlib import Path

import pandas as pd

from aisafety.scripts.summarize_d4_bt_stage_templates import (
    _load_run,
    _pair_level,
    stage_contrast_summary,
    stage_template_interaction_summary,
    template_sensitivity_summary,
)


class D4BTStageTemplateSummaryTests(unittest.TestCase):
    def test_stage_template_interaction_for_surface_cues(self) -> None:
        rows = []
        for template, base, sft in (
            ("standard", 0.2, 0.9),
            ("minimal", 0.1, 0.3),
        ):
            for run_label, margin in (("base", base), ("sft", sft)):
                rows.append(
                    {
                        "template_label": template,
                        "run_label": run_label,
                        "counterfactual_id": "cf1",
                        "pair_id": "p1",
                        "source_dataset": "hc3",
                        "subset": "all",
                        "item_type": "qa",
                        "role": "human",
                        "axis": "structured_assistant_packaging",
                        "direction": "increase",
                        "transform_id": "structured_listify_v1",
                        "presentation_order": "plus_first",
                        "scoring_mode": "forced_choice",
                        "cue_plus_margin": margin,
                        "cue_plus_preferred": margin > 0.0,
                    }
                )
        pair_df = _pair_level(pd.DataFrame(rows))
        _, stage_summary, _ = stage_contrast_summary(pair_df, ["sft_minus_base=sft-base"])
        template_pairs, template_summary, _ = template_sensitivity_summary(
            pair_df,
            ["standard_minus_minimal=standard-minimal"],
        )
        _, interaction_summary, _ = stage_template_interaction_summary(
            template_pairs,
            ["sft_minus_base=sft-base"],
        )

        standard_stage = stage_summary[stage_summary["template_label"] == "standard"]
        self.assertAlmostEqual(float(standard_stage["mean_delta_cue_plus_margin"].iloc[0]), 0.7)
        base_template = template_summary[template_summary["run_label"] == "base"]
        sft_template = template_summary[template_summary["run_label"] == "sft"]
        self.assertAlmostEqual(float(base_template["mean_template_delta_cue_plus_margin"].iloc[0]), 0.1)
        self.assertAlmostEqual(float(sft_template["mean_template_delta_cue_plus_margin"].iloc[0]), 0.6)
        self.assertAlmostEqual(float(interaction_summary["mean_interaction_cue_plus_margin"].iloc[0]), 0.5)

    def test_load_run_from_bt_score_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_dir = root / "run"
            run_dir.mkdir()
            pd.DataFrame(
                [
                    {
                        "counterfactual_id": "cf1",
                        "pair_id": "p1",
                        "source_dataset": "hc3",
                        "role": "llm",
                        "axis": "answer_likeness_packaging",
                        "direction": "increase",
                        "presentation_order": "plus_first",
                        "cue_plus_margin": 0.25,
                    }
                ]
            ).to_csv(run_dir / "bt_stage_scores.csv", index=False)

            loaded = _load_run("standard", "base", run_dir)
            self.assertEqual(loaded["template_label"].iloc[0], "standard")
            self.assertEqual(loaded["run_label"].iloc[0], "base")
            self.assertAlmostEqual(float(loaded["cue_plus_margin"].iloc[0]), 0.25)

    def test_empty_template_interactions_are_allowed_for_single_template_summary(self) -> None:
        empty_pairs = pd.DataFrame()

        interaction_pairs, interaction_summary, interaction_groups = stage_template_interaction_summary(
            empty_pairs,
            ["sft_minus_base=sft-base"],
        )

        self.assertTrue(interaction_pairs.empty)
        self.assertTrue(interaction_summary.empty)
        self.assertTrue(interaction_groups.empty)


if __name__ == "__main__":
    unittest.main()
