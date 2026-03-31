import unittest

import numpy as np
import pandas as pd

from aisafety.eval.debias import (
    add_pairwise_debias_columns,
    compute_swap_debiased_logit_diff,
    swap_ab_columns,
)


class TestSelectorDebias(unittest.TestCase):
    def _paired_trial_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "item_type": "movie",
                    "title": "T1",
                    "A_text": "LLM text",
                    "B_text": "Human text",
                    "A_source": "llm",
                    "B_source": "human",
                },
                {
                    "item_type": "movie",
                    "title": "T1",
                    "A_text": "Human text",
                    "B_text": "LLM text",
                    "A_source": "human",
                    "B_source": "llm",
                },
            ]
        )

    def test_swap_ab_columns(self):
        df = self._paired_trial_df()
        swapped = swap_ab_columns(df)
        self.assertEqual(swapped.loc[0, "A_text"], df.loc[0, "B_text"])
        self.assertEqual(swapped.loc[0, "B_text"], df.loc[0, "A_text"])
        self.assertEqual(swapped.loc[0, "A_source"], df.loc[0, "B_source"])
        self.assertEqual(swapped.loc[0, "B_source"], df.loc[0, "A_source"])

    def test_pairwise_debias_recovers_llm_preference(self):
        df = self._paired_trial_df()
        # Model has A-bias b=2 and true preference x=1 for the LLM option:
        # d_llm_on_A = b + x = 3, d_human_on_A = b - x = 1.
        df["logit_diff"] = [3.0, 1.0]
        out = add_pairwise_debias_columns(df, logit_diff_col="logit_diff", seed=0)
        self.assertTrue((out["preferred_source_debiased"] == "llm").all())
        self.assertEqual(out.loc[0, "choice_debiased"], "A")
        self.assertEqual(out.loc[1, "choice_debiased"], "B")

    def test_pairwise_debias_recovers_human_preference(self):
        df = self._paired_trial_df()
        # Same A-bias b=2, but true preference x=-1 (human preferred):
        # d_llm_on_A = b + x = 1, d_human_on_A = b - x = 3.
        df["logit_diff"] = [1.0, 3.0]
        out = add_pairwise_debias_columns(df, logit_diff_col="logit_diff", seed=0)
        self.assertTrue((out["preferred_source_debiased"] == "human").all())
        self.assertEqual(out.loc[0, "choice_debiased"], "B")
        self.assertEqual(out.loc[1, "choice_debiased"], "A")

    def test_swap_debiased_logit_diff(self):
        # If d_raw=b+x and d_swap=b-x, subtraction yields 2x.
        debiased = compute_swap_debiased_logit_diff(np.array([3.0]), np.array([1.0]))
        self.assertTrue(np.allclose(debiased, np.array([2.0])))
