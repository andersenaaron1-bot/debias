import unittest

from aisafety.scripts.build_d4_human_llm_matched_pairs import build_matched_pairs


class D4HumanLlmMatchedPairsTests(unittest.TestCase):
    def test_matched_pairs_filter_large_length_delta_and_balance_sources(self) -> None:
        rows = []
        for dataset in ("hc3", "hape"):
            for idx in range(4):
                rows.append(
                    {
                        "pair_id": f"{dataset}-good-{idx}",
                        "source_dataset": dataset,
                        "subset": "qa",
                        "question": "Explain the plan.",
                        "human_text": "The plan has two steps and should be checked carefully.",
                        "llm_text": "The plan has three steps and should be checked carefully.",
                    }
                )
            rows.append(
                {
                    "pair_id": f"{dataset}-bad",
                    "source_dataset": dataset,
                    "subset": "qa",
                    "question": "Explain the plan.",
                    "human_text": "short answer",
                    "llm_text": " ".join(["long"] * 60),
                }
            )

        matched, summary = build_matched_pairs(
            rows,
            min_response_tokens=3,
            max_response_tokens=100,
            max_abs_token_delta=5,
            max_length_ratio=1.5,
            max_prompt_overlap_delta=0.5,
            max_type_token_ratio_delta=0.8,
            max_punct_rate_delta=0.5,
            require_all_controls=True,
            strata="dataset_subset",
            max_total_pairs=4,
            max_pairs_per_stratum=0,
            seed=7,
        )

        self.assertEqual(len(matched), 4)
        self.assertEqual(summary["n_matched_uncapped"], 8)
        self.assertEqual(summary["by_dataset"], {"hape": 2, "hc3": 2})
        self.assertEqual(summary["skipped"]["too_short"], 2)
        self.assertTrue(all(row["abs_token_delta"] <= 5 for row in matched))

    def test_matching_can_record_non_length_controls_without_filtering(self) -> None:
        rows = [
            {
                "pair_id": "p1",
                "source_dataset": "hc3",
                "subset": "qa",
                "question": "Explain recursion.",
                "human_text": "Recursion calls itself with smaller cases.",
                "llm_text": "Recursion calls itself with smaller cases!",
            }
        ]

        matched, summary = build_matched_pairs(
            rows,
            min_response_tokens=3,
            max_response_tokens=100,
            max_abs_token_delta=5,
            max_length_ratio=2.0,
            max_prompt_overlap_delta=0.0,
            max_type_token_ratio_delta=0.0,
            max_punct_rate_delta=0.0,
            require_all_controls=False,
            strata="dataset",
            max_total_pairs=0,
            max_pairs_per_stratum=0,
            seed=7,
        )

        self.assertEqual(len(matched), 1)
        self.assertIn("punct_rate_delta_llm_minus_human", matched[0])
        self.assertEqual(summary["n_matched_pairs"], 1)


if __name__ == "__main__":
    unittest.main()
