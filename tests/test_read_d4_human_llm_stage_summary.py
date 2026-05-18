from pathlib import Path
import tempfile
import unittest

import pandas as pd

from aisafety.scripts.read_d4_human_llm_stage_summary import print_summary


class ReadD4HumanLlmStageSummaryTests(unittest.TestCase):
    def test_print_summary_reads_expected_csvs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            pd.DataFrame(
                [
                    {
                        "run_label": "base",
                        "group_type": "all",
                        "group_value": "all",
                        "n_pairs": 2,
                        "mean_llm_margin": -0.1,
                        "median_llm_margin": -0.1,
                        "llm_preference_rate": 0.0,
                        "mean_llm_prob": 0.47,
                    }
                ]
            ).to_csv(out / "stage_summary.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "contrast": "sft-base",
                        "n_pairs": 2,
                        "mean_right_llm_margin": -0.1,
                        "mean_left_llm_margin": 0.2,
                        "mean_delta_llm_margin": 0.3,
                        "right_llm_preference_rate": 0.0,
                        "left_llm_preference_rate": 1.0,
                        "pref_flip_to_llm_rate": 1.0,
                        "pref_flip_from_llm_rate": 0.0,
                    }
                ]
            ).to_csv(out / "stage_contrast_deltas.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "contrast": "sft-base",
                        "group_type": "source_dataset",
                        "group_value": "hc3",
                        "n_pairs": 2,
                        "mean_delta_llm_margin": 0.3,
                        "right_llm_preference_rate": 0.0,
                        "left_llm_preference_rate": 1.0,
                    }
                ]
            ).to_csv(out / "stage_contrast_group_deltas.csv", index=False)

            print_summary(out)


if __name__ == "__main__":
    unittest.main()
