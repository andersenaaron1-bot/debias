from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aisafety.scripts.summarize_judge_suite import (
    format_bundle_contrast,
    format_core_eval_summary,
    format_top_bundles,
    load_bundle_effects,
    load_core_eval_summary,
)


class SummarizeJudgeSuiteTests(unittest.TestCase):
    def test_load_and_format_suite_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            eval_dir = root / "artifacts" / "reward" / "j0_anchor_v1_h100compact" / "eval"
            eval_dir.mkdir(parents=True)
            (eval_dir / "reward_benchmarks").mkdir(parents=True)

            (eval_dir / "pref_retention.json").write_text(
                json.dumps(
                    {
                        "pairwise_acc": 0.61,
                        "separation_auc": 0.64,
                        "mean_margin": 0.12,
                    }
                ),
                encoding="utf-8",
            )
            (eval_dir / "style_sensitivity.json").write_text(
                json.dumps({"summary": [{"style_axis": "formality", "mean_d": 0.07}]}),
                encoding="utf-8",
            )
            (eval_dir / "laurito_bias.json").write_text(
                json.dumps(
                    {
                        "raw": {
                            "overall": {"prop_llm_chosen": 0.52},
                            "paper": {"prop_llm_chosen": 0.60},
                            "product": {"prop_llm_chosen": 0.55},
                            "movie": {"prop_llm_chosen": 0.41},
                        }
                    }
                ),
                encoding="utf-8",
            )
            (eval_dir / "reward_benchmarks" / "summary.csv").write_text(
                "run,benchmark,n_examples,accuracy\nj0,arc_challenge,100,0.62\nj0,boolq,100,0.74\n",
                encoding="utf-8",
            )

            d3_dir = root / "artifacts" / "style_groups" / "d3_j0_anchor_v1_h100compact"
            d3_dir.mkdir(parents=True)
            (d3_dir / "bundle_effects.tsv").write_text(
                "\t".join(["name", "signed_effect_z", "signed_effect_ci_95_low", "signed_effect_ci_95_high", "auc_llm_choice", "d2_status"]) + "\n"
                + "\t".join(["academic_formality", "-0.123", "-0.200", "-0.030", "0.420", "exploratory"]) + "\n"
                + "\t".join(["template_packaging", "0.080", "0.010", "0.150", "0.540", "exploratory"]) + "\n",
                encoding="utf-8",
            )

            core = load_core_eval_summary(root, "j0_anchor_v1_h100compact")
            bundles = load_bundle_effects(root, "j0_anchor_v1_h100compact")

            core_lines = format_core_eval_summary("j0_anchor_v1_h100compact", core)
            bundle_lines = format_top_bundles("j0_anchor_v1_h100compact", bundles, top_n=2)
            contrast_lines = format_bundle_contrast(
                "jrepair_all_v1",
                "j0_anchor_v1_h100compact",
                reference_rows={"academic_formality": {"signed_effect_z": "-0.050"}},
                current_rows=bundles,
                top_n=2,
            )

            self.assertTrue(any("pref_acc=0.6100" in line for line in core_lines))
            self.assertTrue(any("academic_formality" in line for line in bundle_lines))
            self.assertTrue(any("academic_formality" in line for line in contrast_lines))


if __name__ == "__main__":
    unittest.main()
