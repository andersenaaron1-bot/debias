import tempfile
import unittest
from pathlib import Path

import pandas as pd

from aisafety.scripts.build_d4_laurito_human_llm_pairs import (
    _filter_and_cap,
    _pair_row,
    _pairs_from_trials_csv,
)


class D4LauritoHumanLlmPairsTest(unittest.TestCase):
    def test_trials_csv_deduplicates_order_swaps(self) -> None:
        human_text = "A human review with enough words to behave like a complete comparison item."
        llm_text = "An LLM review with enough words to behave like a complete comparison item."
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "trials.csv"
            pd.DataFrame(
                [
                    {
                        "item_type": "movie",
                        "title": "Example movie",
                        "A_text": human_text,
                        "B_text": llm_text,
                        "A_source": "human",
                        "B_source": "llm",
                        "split": "test",
                        "llm_generator": "gpt",
                    },
                    {
                        "item_type": "movie",
                        "title": "Example movie",
                        "A_text": llm_text,
                        "B_text": human_text,
                        "A_source": "llm",
                        "B_source": "human",
                        "split": "test",
                        "llm_generator": "gpt",
                    },
                ]
            ).to_csv(path, index=False)

            rows, summary = _pairs_from_trials_csv(path, include_item_types={"movie", "paper"})

        self.assertEqual(len(rows), 1)
        self.assertEqual(summary["skipped"], {"duplicate_order_swap": 1})
        self.assertEqual(rows[0]["source_dataset"], "laurito_movie")
        self.assertEqual(rows[0]["bundle_creation_role"], "laurito_quality_validation")
        self.assertEqual(rows[0]["human_text"], human_text)
        self.assertEqual(rows[0]["llm_text"], llm_text)

    def test_filter_and_cap_balances_across_laurito_domains(self) -> None:
        rows = []
        for item_type in ("movie", "paper", "product"):
            for idx in range(4):
                row = _pair_row(
                    item_type=item_type,
                    title=f"{item_type}-{idx}",
                    human_text=f"human answer for {item_type} {idx} with enough words",
                    llm_text=f"llm answer for {item_type} {idx} with enough words",
                )
                assert row is not None
                rows.append(row)

        capped, summary = _filter_and_cap(
            rows,
            min_tokens=3,
            max_tokens=20,
            max_pairs_per_item_type=2,
            max_total_pairs=5,
            seed=13,
        )

        counts: dict[str, int] = {}
        for row in capped:
            counts[str(row["item_type"])] = counts.get(str(row["item_type"]), 0) + 1

        self.assertEqual(summary["n_before_length_filter"], 12)
        self.assertEqual(summary["n_after_length_filter"], 12)
        self.assertEqual(summary["n_pairs"], 5)
        self.assertEqual(set(counts), {"movie", "paper", "product"})
        self.assertLessEqual(max(counts.values()), 2)


if __name__ == "__main__":
    unittest.main()
