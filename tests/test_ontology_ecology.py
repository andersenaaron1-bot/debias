from __future__ import annotations

import unittest

import pandas as pd

from aisafety.ontology.ecology import (
    attach_atom_and_bundle_deltas,
    build_ecological_effect_tables,
    build_pair_text_atom_scores,
    collapse_scored_trials_to_pairs,
)


class EcologyTests(unittest.TestCase):
    def test_collapse_scored_trials_to_pairs_debiases_swaps(self):
        df = pd.DataFrame(
            [
                {
                    "item_type": "paper",
                    "title": "t1",
                    "A_text": "Human abstract with nuanced limitations and evidence.",
                    "A_source": "human",
                    "B_text": "LLM abstract with polished framing and summary style.",
                    "B_source": "llm",
                    "score_A": 0.1,
                    "score_B": 0.3,
                },
                {
                    "item_type": "paper",
                    "title": "t1",
                    "A_text": "LLM abstract with polished framing and summary style.",
                    "A_source": "llm",
                    "B_text": "Human abstract with nuanced limitations and evidence.",
                    "B_source": "human",
                    "score_A": 0.3,
                    "score_B": 0.1,
                },
            ]
        )
        pairs = collapse_scored_trials_to_pairs(df, seed=1234)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(int(pairs.iloc[0]["y_llm_chosen"]), 1)
        self.assertEqual(int(pairs.iloc[0]["n_order_rows"]), 2)

    def test_attach_and_effect_tables(self):
        df = pd.DataFrame(
            [
                {
                    "item_type": "paper",
                    "title": "p1",
                    "A_text": "Human abstract with modest claims and evidence from prior work.",
                    "A_source": "human",
                    "B_text": "LLM abstract that clearly demonstrates a powerful framework and results show strong gains.",
                    "B_source": "llm",
                    "score_A": 0.1,
                    "score_B": 0.4,
                },
                {
                    "item_type": "paper",
                    "title": "p1",
                    "A_text": "LLM abstract that clearly demonstrates a powerful framework and results show strong gains.",
                    "A_source": "llm",
                    "B_text": "Human abstract with modest claims and evidence from prior work.",
                    "B_source": "human",
                    "score_A": 0.4,
                    "score_B": 0.1,
                },
                {
                    "item_type": "movie",
                    "title": "m1",
                    "A_text": "Human synopsis follows a family facing danger and conflict before they survive.",
                    "A_source": "human",
                    "B_text": "LLM synopsis follows a mission, conflict, resolution, and an overall takeaway.",
                    "B_source": "llm",
                    "score_A": 0.2,
                    "score_B": 0.5,
                },
                {
                    "item_type": "movie",
                    "title": "m1",
                    "A_text": "LLM synopsis follows a mission, conflict, resolution, and an overall takeaway.",
                    "A_source": "llm",
                    "B_text": "Human synopsis follows a family facing danger and conflict before they survive.",
                    "B_source": "human",
                    "score_A": 0.5,
                    "score_B": 0.2,
                },
            ]
        )
        pairs = collapse_scored_trials_to_pairs(df, seed=1234)
        text_scores = build_pair_text_atom_scores(pairs)
        pairs = attach_atom_and_bundle_deltas(
            pairs,
            text_atom_scores=text_scores,
            bundle_members={"stance_calibration": ["booster_certainty_markers", "hedge_markers"]},
        )
        atom_effects, bundle_effects = build_ecological_effect_tables(pairs, n_bootstrap=32, seed=1234)
        self.assertIn("booster_certainty_markers", atom_effects)
        self.assertIn("stance_calibration", bundle_effects)
        self.assertIn("by_item_type", bundle_effects["stance_calibration"])


if __name__ == "__main__":
    unittest.main()
