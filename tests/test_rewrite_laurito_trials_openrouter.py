import unittest

import pandas as pd

from aisafety.scripts.rewrite_laurito_trials_openrouter import (
    apply_rewrites_to_trials,
    build_rewrite_requests,
)


class TestRewriteLauritoTrialsOpenRouter(unittest.TestCase):
    def _balanced_trials(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "item_type": "movie",
                    "title": "T1",
                    "A_text": "LLM version.",
                    "B_text": "Human version.",
                    "A_source": "llm",
                    "B_source": "human",
                },
                {
                    "item_type": "movie",
                    "title": "T1",
                    "A_text": "Human version.",
                    "B_text": "LLM version.",
                    "A_source": "human",
                    "B_source": "llm",
                },
            ]
        )

    def test_build_rewrite_requests_dedupes_balanced_order(self):
        df = self._balanced_trials()
        reqs = build_rewrite_requests(
            df,
            dimension="ai_tone",
            target_label="human_plain",
            model="openai/gpt-4o-mini",
            temperature=0.4,
            top_p=None,
            max_tokens=32,
            max_chars=2000,
        )
        # One title has two unique texts to rewrite (human + llm).
        self.assertEqual(len(reqs), 2)
        labels = sorted({r.target_label for r in reqs})
        self.assertEqual(labels, ["human_plain"])

    def test_apply_rewrites_replaces_both_sides(self):
        df = self._balanced_trials()
        reqs = build_rewrite_requests(
            df,
            dimension="ai_tone",
            target_label="human_plain",
            model="openai/gpt-4o-mini",
            temperature=0.4,
            top_p=None,
            max_tokens=32,
            max_chars=2000,
        )
        rewrite_map = {r.key: f"REWRITE({r.source})" for r in reqs}

        out = apply_rewrites_to_trials(
            df,
            rewrite_map=rewrite_map,
            dimension="ai_tone",
            target_label="human_plain",
            keep_original=True,
            model="openai/gpt-4o-mini",
            temperature=0.4,
            top_p=None,
            max_tokens=32,
            max_chars=2000,
        )
        # First row: A is llm, B is human
        self.assertEqual(out.loc[0, "A_text"], "REWRITE(llm)")
        self.assertEqual(out.loc[0, "B_text"], "REWRITE(human)")
        # Second row: A is human, B is llm
        self.assertEqual(out.loc[1, "A_text"], "REWRITE(human)")
        self.assertEqual(out.loc[1, "B_text"], "REWRITE(llm)")
        self.assertIn("A_text_original", out.columns)
        self.assertIn("B_text_original", out.columns)

    def test_per_source_target_labels(self):
        df = self._balanced_trials()
        reqs = build_rewrite_requests(
            df,
            dimension="ai_tone",
            target_label="human_plain",
            human_target_label="rlhf_ai_tone",
            llm_target_label="human_plain",
            model="openai/gpt-4o-mini",
            temperature=0.4,
            top_p=None,
            max_tokens=32,
            max_chars=2000,
        )
        self.assertEqual(len(reqs), 2)
        by_source = {r.source: r.target_label for r in reqs}
        self.assertEqual(by_source["human"], "rlhf_ai_tone")
        self.assertEqual(by_source["llm"], "human_plain")

