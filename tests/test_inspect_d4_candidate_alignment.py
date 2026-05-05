import csv
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from aisafety.scripts.inspect_d4_candidate_alignment import build_readout


class InspectD4CandidateAlignmentTest(unittest.TestCase):
    def test_build_readout_summarizes_alignment_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            run_dir.mkdir()
            (run_dir / "alignment_manifest.json").write_text(
                json.dumps(
                    {
                        "n_pairs": 2,
                        "n_unique_texts": 4,
                        "y_llm_chosen_rate": 0.5,
                        "mean_llm_margin_pair": 0.1,
                        "completed_hidden_layers": [30],
                        "skipped_layers": [],
                    }
                ),
                encoding="utf-8",
            )
            with (run_dir / "pair_scores.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["source_dataset", "item_type", "split"])
                writer.writeheader()
                writer.writerow({"source_dataset": "hc3", "item_type": "hc3", "split": "train"})
                writer.writerow({"source_dataset": "hc3_plus", "item_type": "general", "split": "test"})
            with (run_dir / "candidate_feature_human_llm_alignment.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                fields = [
                    "candidate_kind",
                    "hidden_layer",
                    "feature_idx",
                    "atoms",
                    "bundles",
                    "n_pairs",
                    "mean_llm_minus_human_activation",
                    "activation_auc_llm_vs_human",
                    "auc_j0_llm_choice_from_activation_delta",
                    "length_controlled_spearman_delta_with_j0_margin",
                    "length_controlled_spearman_q",
                    "source_sign_consistency",
                    "n_sources_with_min_pairs",
                    "candidate_reasons",
                ]
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "candidate_kind": "discovered",
                        "hidden_layer": "30",
                        "feature_idx": "5394",
                        "atoms": "epistemic_hedging",
                        "bundles": "certainty_and_stance_calibration",
                        "n_pairs": "2",
                        "mean_llm_minus_human_activation": "0.2",
                        "activation_auc_llm_vs_human": "0.7",
                        "auc_j0_llm_choice_from_activation_delta": "0.6",
                        "length_controlled_spearman_delta_with_j0_margin": "0.5",
                        "length_controlled_spearman_q": "0.04",
                        "source_sign_consistency": "1.0",
                        "n_sources_with_min_pairs": "1",
                        "candidate_reasons": "atom_val_auc",
                    }
                )
            with (run_dir / "candidate_feature_source_alignment.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                fields = [
                    "source_dataset",
                    "item_type",
                    "hidden_layer",
                    "feature_idx",
                    "atoms",
                    "bundles",
                    "n_pairs",
                    "mean_llm_minus_human_activation",
                    "activation_auc_llm_vs_human",
                    "auc_j0_llm_choice_from_activation_delta",
                    "spearman_delta_with_j0_margin",
                ]
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "source_dataset": "hc3",
                        "item_type": "hc3",
                        "hidden_layer": "30",
                        "feature_idx": "5394",
                        "atoms": "epistemic_hedging",
                        "bundles": "certainty_and_stance_calibration",
                        "n_pairs": "2",
                        "mean_llm_minus_human_activation": "0.2",
                        "activation_auc_llm_vs_human": "0.7",
                        "auc_j0_llm_choice_from_activation_delta": "0.6",
                        "spearman_delta_with_j0_margin": "0.5",
                    }
                )

            readout = build_readout(run_dir, top_k=5, source_top_k=5)
            self.assertIn("n_pairs: `2`", readout)
            self.assertIn("epistemic_hedging", readout)
            self.assertIn("Control Baseline", readout)

    def test_archive_fixture_is_valid_tar_gz(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_dir = root / "run"
            run_dir.mkdir()
            (run_dir / "alignment_manifest.json").write_text("{}", encoding="utf-8")
            archive = root / "run.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                tar.add(run_dir, arcname=run_dir.name)
            self.assertTrue(archive.is_file())


if __name__ == "__main__":
    unittest.main()
