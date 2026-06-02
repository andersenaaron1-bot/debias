import unittest
from argparse import Namespace

import pandas as pd

from aisafety.scripts.run_d4_openrouter_judge_screen import (
    _cap_rows,
    _effective_reasoning_effort,
    _estimate_rows,
    contrast_summaries,
    parse_choice,
    summarize_scores,
)


class D4OpenRouterJudgeScreenTests(unittest.TestCase):
    def test_parse_choice_accepts_short_verdicts(self) -> None:
        self.assertEqual(parse_choice("A"), "A")
        self.assertEqual(parse_choice("Answer: b"), "B")
        self.assertEqual(parse_choice("I prefer A because it is clearer."), "")
        self.assertEqual(parse_choice("unclear"), "")

    def test_cap_rows_keeps_order_swaps_together(self) -> None:
        rows = [
            {"pair_id": "p1", "bt_pair_id": "p1a"},
            {"pair_id": "p1", "bt_pair_id": "p1b"},
            {"pair_id": "p2", "bt_pair_id": "p2a"},
            {"pair_id": "p2", "bt_pair_id": "p2b"},
        ]
        capped = _cap_rows(rows, max_sources=1, seed=1234, dataset="hllm")
        self.assertEqual(len(capped), 2)
        self.assertEqual(len({row["pair_id"] for row in capped}), 1)

    def test_estimate_uses_catalog_prices(self) -> None:
        estimated = _estimate_rows(
            [{"prompt": "a" * 40}, {"prompt": "b" * 40}],
            models=[("small", "provider/model")],
            catalog={"provider/model": {"pricing": {"prompt": "0.1", "completion": "0.2"}}},
            chars_per_token=4.0,
            completion_tokens=2,
        )
        self.assertEqual(int(estimated.iloc[0]["estimated_prompt_tokens"]), 20)
        self.assertAlmostEqual(float(estimated.iloc[0]["estimated_cost_usd"]), 2.8)

    def test_reasoning_effort_is_only_sent_to_supported_models(self) -> None:
        args = Namespace(reasoning_effort="none")
        self.assertEqual(_effective_reasoning_effort(args, {"supported_parameters": ["reasoning"]}), "none")
        self.assertEqual(_effective_reasoning_effort(args, {"supported_parameters": ["max_tokens"]}), "")

    def test_summary_and_model_contrast_average_order_swaps(self) -> None:
        scores = pd.DataFrame(
            [
                self._score("base", "p1", True),
                self._score("base", "p1", True),
                self._score("it", "p1", True),
                self._score("it", "p1", False),
            ]
        )
        pair_df, summary = summarize_scores(scores)
        base = summary[summary["model_label"] == "base"].iloc[0]
        it = summary[summary["model_label"] == "it"].iloc[0]
        self.assertEqual(float(base["mean_target_preference_share"]), 1.0)
        self.assertEqual(float(it["mean_target_preference_share"]), 0.5)
        contrast = contrast_summaries(pair_df, ["it_minus_base=it-base"])
        self.assertEqual(float(contrast.iloc[0]["mean_delta_target_preference_share"]), -0.5)

    @staticmethod
    def _score(model_label: str, source_id: str, preferred: bool) -> dict:
        return {
            "dataset": "hllm",
            "model_label": model_label,
            "model_id": f"provider/{model_label}",
            "source_id": source_id,
            "valid_choice": True,
            "target_preferred": preferred,
            "prompt_tokens": 10,
            "completion_tokens": 1,
            "cost_usd": 0.01,
        }


if __name__ == "__main__":
    unittest.main()
