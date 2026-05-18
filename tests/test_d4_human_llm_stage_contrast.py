import tempfile
import unittest
from pathlib import Path

import pandas as pd

from aisafety.scripts.build_d4_human_llm_stage_contrast_pairs import build_hllm_bt_rows
from aisafety.scripts.build_d4_human_llm_surface_control_pairs import build_surface_control_pairs
from aisafety.scripts.run_d4_human_llm_stage_contrast import _comparison_content
from aisafety.scripts.summarize_d4_human_llm_stage_contrasts import _contrast_summary, _pair_level


class D4HumanLlmStageContrastTests(unittest.TestCase):
    def test_build_hllm_bt_rows_orders_llm_target(self) -> None:
        rows, summary = build_hllm_bt_rows(
            [
                {
                    "pair_id": "p1",
                    "source_dataset": "hc3",
                    "subset": "finance",
                    "split": "test",
                    "item_type": "qa",
                    "group_id": "g1",
                    "question": "How should I save money?",
                    "human_text": "Spend less than you earn and track purchases.",
                    "llm_text": "You can build a budget, reduce expenses, and save automatically.",
                    "llm_generator": "chatgpt",
                }
            ],
            include_order_swaps=True,
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(summary["n_source_pairs_used"], 1)
        llm_first = next(row for row in rows if row["presentation_order"] == "llm_first")
        human_first = next(row for row in rows if row["presentation_order"] == "human_first")
        self.assertEqual(llm_first["llm_option"], "A")
        self.assertEqual(human_first["llm_option"], "B")
        self.assertEqual(llm_first["prompt"], "How should I save money?")
        self.assertGreater(llm_first["token_delta_llm_minus_human"], 0)

    def test_stage_summary_computes_order_debiased_contrast(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "run_label": "base",
                    "pair_id": "p1",
                    "source_dataset": "hc3",
                    "subset": "finance",
                    "item_type": "qa",
                    "presentation_order": "llm_first",
                    "llm_margin": -0.2,
                    "llm_prob": 0.45,
                    "llm_preferred": False,
                },
                {
                    "run_label": "base",
                    "pair_id": "p1",
                    "source_dataset": "hc3",
                    "subset": "finance",
                    "item_type": "qa",
                    "presentation_order": "human_first",
                    "llm_margin": 0.0,
                    "llm_prob": 0.50,
                    "llm_preferred": False,
                },
                {
                    "run_label": "it",
                    "pair_id": "p1",
                    "source_dataset": "hc3",
                    "subset": "finance",
                    "item_type": "qa",
                    "presentation_order": "llm_first",
                    "llm_margin": 0.4,
                    "llm_prob": 0.60,
                    "llm_preferred": True,
                },
                {
                    "run_label": "it",
                    "pair_id": "p1",
                    "source_dataset": "hc3",
                    "subset": "finance",
                    "item_type": "qa",
                    "presentation_order": "human_first",
                    "llm_margin": 0.2,
                    "llm_prob": 0.55,
                    "llm_preferred": True,
                },
            ]
        )

        pair_df = _pair_level(df)
        contrasts, grouped = _contrast_summary(pair_df, ["it_minus_base=it-base"])

        self.assertEqual(len(pair_df), 2)
        self.assertAlmostEqual(
            float(pair_df[pair_df["run_label"] == "base"]["mean_llm_margin"].iloc[0]),
            -0.1,
        )
        self.assertAlmostEqual(float(contrasts["mean_delta_llm_margin"].iloc[0]), 0.4)
        self.assertEqual(grouped[grouped["group_type"] == "source_dataset"]["group_value"].iloc[0], "hc3")

    def test_summarizer_loads_score_csvs_from_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_dir = root / "run"
            run_dir.mkdir()
            pd.DataFrame(
                [
                    {
                        "pair_id": "p1",
                        "source_dataset": "hc3",
                        "subset": "finance",
                        "item_type": "qa",
                        "presentation_order": "llm_first",
                        "stage_label": "base",
                        "scoring_mode": "forced_choice",
                        "model_id": "m",
                        "prompt_style": "plain",
                        "llm_margin": 0.1,
                        "llm_prob": 0.52,
                        "llm_preferred": True,
                    }
                ]
            ).to_csv(run_dir / "hllm_stage_scores.csv", index=False)

            # Import inside the test to keep the public functions above simple to exercise.
            from aisafety.scripts.summarize_d4_human_llm_stage_contrasts import _load_run

            loaded = _load_run("base", run_dir)
            self.assertEqual(loaded["run_label"].iloc[0], "base")
            self.assertAlmostEqual(float(loaded["llm_margin"].iloc[0]), 0.1)

    def test_comparison_templates_change_instruction_surface(self) -> None:
        row = {
            "prompt": "How should I save money?",
            "option_a_text": "Track expenses.",
            "option_b_text": "Make a budget.",
        }

        standard = _comparison_content(row, comparison_template="standard")
        minimal = _comparison_content(row, comparison_template="minimal")
        substance = _comparison_content(row, comparison_template="substance_only")

        self.assertIn("Which response is better?", standard)
        self.assertIn("Better response?", minimal)
        self.assertIn("Do not prefer a response because it is longer", substance)

    def test_surface_control_pairs_flatten_assistant_packaging(self) -> None:
        rows = [
            {
                "pair_id": "p1",
                "source_dataset": "hc3",
                "subset": "finance",
                "question": "How should I save money?",
                "human_text": (
                    "People can save money by tracking what they spend, setting a realistic budget, "
                    "and moving a small amount into savings each month."
                ),
                "llm_text": (
                    "Answer:\n"
                    "Saving money works best when it is routine.\n\n"
                    "Details:\n"
                    "- Track what you spend every week.\n"
                    "- Set a realistic monthly budget.\n"
                    "- Move a small amount into savings each month."
                ),
            }
        ]

        controlled, summary = build_surface_control_pairs(
            rows,
            mode="surface_minimized",
            min_response_tokens=5,
            min_transform_length_ratio=0.4,
            require_changed=True,
        )

        self.assertEqual(len(controlled), 1)
        self.assertEqual(summary["n_surface_control_pairs"], 1)
        self.assertTrue(controlled[0]["llm_surface_changed"])
        self.assertNotIn("Answer:", controlled[0]["llm_text"])
        self.assertNotIn("\n", controlled[0]["llm_text"])
        self.assertIn("surface_control_mode", controlled[0])


if __name__ == "__main__":
    unittest.main()
