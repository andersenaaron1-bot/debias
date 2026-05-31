import argparse
from contextlib import redirect_stdout
import io
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from aisafety.scripts.audit_d4_lexical_judge_artifacts import (
    audit_hllm_artifacts,
    audit_surface_fragments,
)
from aisafety.scripts.read_d4_lexical_judge_artifacts import readout


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        word_ngram_max=2,
        char_ngram_min=3,
        char_ngram_max=5,
        word_max_features=100,
        char_max_features=0,
        word_min_df=1,
        char_min_df=1,
        elastic_alpha=0.001,
        elastic_l1_ratio=0.8,
        cv_folds=3,
        top_k=30,
        seed=1234,
    )


class D4LexicalJudgeArtifactTests(unittest.TestCase):
    def test_hllm_audit_recovers_answer_token_marker_after_controls(self) -> None:
        pair_rows = []
        target_rows = []
        for index in range(36):
            has_signal = index % 2 == 0
            pair_id = f"p{index:02d}"
            pair_rows.append(
                {
                    "pair_id": pair_id,
                    "source_dataset": "source",
                    "subset": "subset",
                    "item_type": "general",
                    "human_text": "baselineword common response",
                    "llm_text": (
                        "signalmarker common response"
                        if has_signal
                        else "neutralmarker common response"
                    ),
                    "human_tokens": 3,
                    "llm_tokens": 3,
                }
            )
            target_rows.append(
                {
                    "pair_id": pair_id,
                    "target_name": "judge::margin",
                    "target_value": 2.0 if has_signal else -2.0,
                }
            )

        coefficients, metrics, targets = audit_hllm_artifacts(
            pd.DataFrame(pair_rows),
            pd.DataFrame(target_rows),
            args=_args(),
        )

        signal = coefficients[coefficients["artifact_name"] == "word::signalmarker"].iloc[0]
        neutral = coefficients[coefficients["artifact_name"] == "word::neutralmarker"].iloc[0]
        self.assertGreater(float(signal["partial_corr"]), 0.9)
        self.assertLess(float(neutral["partial_corr"]), -0.9)
        self.assertEqual(len(metrics), 3)
        self.assertEqual(int(targets.iloc[0]["n_pairs"]), 36)

    def test_surface_audit_extracts_inserted_fragments(self) -> None:
        counterfactual_rows = []
        target_rows = []
        for index in range(12):
            counterfactual_id = f"cf{index:02d}"
            key_points = index % 2 == 0
            counterfactual_rows.append(
                {
                    "counterfactual_id": counterfactual_id,
                    "pair_id": f"p{index:02d}",
                    "source_dataset": "source",
                    "subset": "subset",
                    "item_type": "general",
                    "axis": "answer_likeness_packaging",
                    "direction": "cue_plus",
                    "role": "llm",
                    "transform_id": "prefix",
                    "base_text": "The answer is concise.",
                    "variant_text": (
                        "Key points: The answer is concise."
                        if key_points
                        else "Summary note: The answer is concise."
                    ),
                }
            )
            target_rows.append(
                {
                    "counterfactual_id": counterfactual_id,
                    "target_name": "judge::cue_margin",
                    "target_value": 1.0 if key_points else -1.0,
                }
            )

        effects, fragments = audit_surface_fragments(
            counterfactual_rows,
            pd.DataFrame(target_rows),
            max_fragment_tokens=3,
            min_df=2,
        )

        self.assertIn("insert::key points", set(fragments["artifact_name"]))
        signal = effects[effects["artifact_name"] == "insert::key points"].iloc[0]
        self.assertAlmostEqual(float(signal["artifact_minus_global_mean"]), 1.0)
        self.assertAlmostEqual(float(signal["mean_stratum_adjusted_target_value"]), 1.0)

    def test_compact_readout_prints_shortlists(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pd.DataFrame(
                [
                    {
                        "target_name": "judge::sft_minus_base",
                        "artifact_name": "word::signal",
                        "artifact_kind": "word",
                        "support_pairs": 12,
                        "mean_llm_minus_human_presence": 0.5,
                        "partial_corr": 0.4,
                        "abs_partial_corr": 0.4,
                        "partial_slope": 0.7,
                        "elastic_coef": 0.3,
                    }
                ]
            ).to_csv(root / "hllm_artifact_coefficients.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "target_name": "judge::sft_minus_base",
                        "fold": 0,
                        "r2": 0.2,
                        "pearson": 0.5,
                        "mae": 0.4,
                        "baseline_mae": 0.6,
                    }
                ]
            ).to_csv(root / "hllm_heldout_metrics.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "target_name": "judge::stage_delta::sft_minus_base",
                        "artifact_name": "insert::key points",
                        "support_counterfactuals": 12,
                        "abs_artifact_minus_global_mean": 0.3,
                        "artifact_minus_global_mean": 0.3,
                        "mean_stratum_adjusted_target_value": 0.1,
                    }
                ]
            ).to_csv(root / "surface_edit_fragment_effects.csv", index=False)

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                readout(
                    root,
                    top_k=2,
                    max_targets=2,
                    min_support=5,
                    target_regex=r"minus|stage_delta",
                )

            text = stdout.getvalue()
            self.assertIn("word::signal", text)
            self.assertIn("insert::key points", text)


if __name__ == "__main__":
    unittest.main()
