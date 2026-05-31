import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from aisafety.mech.d4_io import sha1_hex
from aisafety.scripts.build_d4_assistant_style_atomic_counterfactual_pairs import build_atomic_rows
from aisafety.scripts.build_d4_style_rewrite_seeds import build_seed_rows
from aisafety.scripts.convert_d4_generated_style_counterfactual_pairs import (
    build_generated_counterfactual_rows,
)
from aisafety.scripts.run_d4_lm_style_activation_contrast import (
    analyze_activation_deltas,
    cross_validated_direction_scores,
)
from aisafety.scripts.summarize_d4_lm_style_activation_contrasts import summarize


PAIR = {
    "pair_id": "p1",
    "source_dataset": "hc3",
    "subset": "finance",
    "split": "test",
    "item_type": "qa",
    "question": "What is the tradeoff?",
    "human_text": "The human response explains the tradeoff with enough detail for a careful reader.",
    "llm_text": "The model response explains the tradeoff with enough detail for a careful reader.",
}


class D4StyleCausalityTests(unittest.TestCase):
    def test_builds_atomic_marker_edits_and_placebo(self) -> None:
        rows, summary = build_atomic_rows(
            [PAIR],
            max_pairs=0,
            min_tokens=5,
            max_length_ratio=1.5,
            seed=1234,
        )
        marker_ids = {row["marker_id"] for row in rows}
        self.assertIn("answer_label", marker_ids)
        self.assertIn("furthermore_preface", marker_ids)
        self.assertIn("response_label", marker_ids)
        self.assertIn("placebo_response_label", {row["axis"] for row in rows})
        self.assertEqual(summary["n_input_pairs"], 1)

    def test_generated_pair_converter_keeps_plain_vs_assistant_orientation(self) -> None:
        rows, summary = build_generated_counterfactual_rows(
            [
                {
                    "dimension": "ai_tone",
                    "label": "human_plain",
                    "seed_id": 0,
                    "seed_text": "Seed text.",
                    "generated_text": "The plain rewrite has enough words to satisfy the filter.",
                    "model": "generator",
                    "meta": {"rewrite_seed_id": "s1", "pair_id": "p1", "role": "human"},
                },
                {
                    "dimension": "ai_tone",
                    "label": "rlhf_ai_tone",
                    "seed_id": 0,
                    "seed_text": "Seed text.",
                    "generated_text": "Overall, the assistant rewrite has enough words to satisfy the filter.",
                    "model": "generator",
                    "meta": {"rewrite_seed_id": "s1", "pair_id": "p1", "role": "human"},
                },
            ],
            dimension="ai_tone",
            assistant_label="rlhf_ai_tone",
            plain_label="human_plain",
            min_tokens=2,
            min_length_ratio=0.5,
            max_length_ratio=2.0,
            min_content_token_jaccard=0.0,
        )
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["variant_text"].startswith("Overall"))
        self.assertEqual(rows[0]["direction"], "increase")
        self.assertEqual(summary["n_counterfactuals"], 1)

    def test_seed_builder_preserves_answer_metadata(self) -> None:
        rows = build_seed_rows([PAIR], max_seeds=10, min_tokens=5, max_tokens=100, seed=1234)
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["role"] for row in rows}, {"human", "llm"})
        self.assertTrue(all(row["prompt"] == "What is the tradeoff?" for row in rows))

    def test_cross_validated_direction_detects_consistent_shift(self) -> None:
        deltas = np.asarray([[1.0, 0.0], [0.9, 0.1], [1.1, -0.1], [1.0, 0.1], [0.8, -0.1], [1.2, 0.0]])
        ids = [f"cf{i}" for i in range(len(deltas))]
        projections, cosines, _ = cross_validated_direction_scores(deltas, ids, folds=3, seed=1234)
        self.assertGreater(float(np.nanmean(projections)), 0.8)
        self.assertGreater(float(np.nanmean(cosines)), 0.95)

    def test_activation_analysis_and_stage_summary(self) -> None:
        rows = [
            {
                "counterfactual_id": f"cf{i}",
                "pair_id": f"p{i}",
                "axis": "assistant_answer_label",
                "direction": "increase",
                "base_text": f"plain {i}",
                "variant_text": f"assistant {i}",
            }
            for i in range(8)
        ]
        texts = sorted({row[key] for row in rows for key in ("base_text", "variant_text")})
        text_index = {sha1_hex(text): index for index, text in enumerate(texts)}
        pooled = np.zeros((len(texts), 2), dtype=np.float32)
        for index, text in enumerate(texts):
            pooled[index] = [1.0, 0.0] if text.startswith("assistant") else [0.0, 0.0]
        details, summary, directions = analyze_activation_deltas(
            counterfactual_rows=rows,
            pooled_by_layer={1: pooled},
            text_index=text_index,
            binary_margin={f"cf{i}": float(i) for i in range(8)},
            run_label="base",
            model_id="model",
            folds=2,
            seed=1234,
        )
        self.assertEqual(len(details), 8)
        self.assertAlmostEqual(float(summary.iloc[0]["mean_cv_cosine_to_style_direction"]), 1.0)
        self.assertIn("hidden_1", directions)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = root / "base"
            instruct = root / "instruct"
            base.mkdir()
            instruct.mkdir()
            base_summary = summary.copy()
            instruct_summary = summary.copy()
            instruct_summary["direction_concentration"] += 0.2
            base_summary.to_csv(base / "layer_summary.csv", index=False)
            instruct_summary.to_csv(instruct / "layer_summary.csv", index=False)
            np.savez_compressed(base / "layer_mean_directions.npz", hidden_1=np.asarray([1.0, 0.0]))
            np.savez_compressed(instruct / "layer_mean_directions.npz", hidden_1=np.asarray([1.0, 0.0]))
            stage = summarize(base, instruct, contrast_label="it_minus_base")
            self.assertAlmostEqual(float(stage.iloc[0]["delta_direction_concentration"]), 0.2)
            self.assertAlmostEqual(float(stage.iloc[0]["base_instruct_mean_direction_cosine"]), 1.0)


if __name__ == "__main__":
    unittest.main()
