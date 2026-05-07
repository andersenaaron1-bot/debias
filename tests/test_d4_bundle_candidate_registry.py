import csv
import json
from pathlib import Path
import tempfile
import unittest

from aisafety.scripts.build_d4_bundle_candidate_registry import build_registry


def _write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class D4BundleCandidateRegistryTest(unittest.TestCase):
    def test_registry_freezes_bundle_features_and_layer_matched_controls(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_dir = root / "alignment"
            source_dir = root / "merged"
            out_dir = root / "registry"
            run_dir.mkdir()
            source_dir.mkdir()
            (run_dir / "alignment_manifest.json").write_text(
                json.dumps({"n_pairs": 100, "completed_hidden_layers": [34, 40]}),
                encoding="utf-8",
            )
            config = {
                "version": "test",
                "sae_release": "rel",
                "aggregation": "max",
                "eligibility": {
                    "min_bundle_atoms_for_paper_claim": 3,
                    "min_sources_with_pairs": 4,
                    "preferred_source_sign_consistency": 0.8,
                    "allow_source_sensitive_consistency": 0.6,
                },
                "bundles": [
                    {
                        "bundle_id": "formal_packaging",
                        "paper_label": "Formal packaging",
                        "role": "primary",
                        "atoms": [
                            "formal_connectives",
                            "institutional_impersonality",
                            "professional_self_presentation",
                        ],
                        "support_atoms": [],
                        "primary_features": [
                            {"hidden_layer": 34, "feature_idx": 7691},
                            {"hidden_layer": 40, "feature_idx": 15478},
                            {"hidden_layer": 39, "feature_idx": 15970},
                        ],
                    }
                ],
            }
            cfg_path = root / "config.json"
            cfg_path.write_text(json.dumps(config), encoding="utf-8")

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
            _write_csv(
                run_dir / "candidate_feature_human_llm_alignment.csv",
                [
                    {
                        "candidate_kind": "discovered",
                        "hidden_layer": "34",
                        "feature_idx": "7691",
                        "atoms": "formal_connectives;institutional_impersonality",
                        "bundles": "formal_information_packaging",
                        "n_pairs": "100",
                        "mean_llm_minus_human_activation": "2.0",
                        "activation_auc_llm_vs_human": "0.7",
                        "auc_j0_llm_choice_from_activation_delta": "0.75",
                        "length_controlled_spearman_delta_with_j0_margin": "0.2",
                        "length_controlled_spearman_q": "0.001",
                        "source_sign_consistency": "1.0",
                        "n_sources_with_min_pairs": "4",
                        "candidate_reasons": "bundle_member",
                    },
                    {
                        "candidate_kind": "discovered",
                        "hidden_layer": "40",
                        "feature_idx": "15478",
                        "atoms": "professional_self_presentation",
                        "bundles": "formal_information_packaging",
                        "n_pairs": "100",
                        "mean_llm_minus_human_activation": "1.5",
                        "activation_auc_llm_vs_human": "0.65",
                        "auc_j0_llm_choice_from_activation_delta": "0.72",
                        "length_controlled_spearman_delta_with_j0_margin": "0.18",
                        "length_controlled_spearman_q": "0.002",
                        "source_sign_consistency": "0.7",
                        "n_sources_with_min_pairs": "4",
                        "candidate_reasons": "bundle_member",
                    },
                    {
                        "candidate_kind": "random_control",
                        "hidden_layer": "34",
                        "feature_idx": "100",
                        "atoms": "",
                        "bundles": "",
                        "n_pairs": "100",
                        "mean_llm_minus_human_activation": "0.0",
                        "activation_auc_llm_vs_human": "0.51",
                        "auc_j0_llm_choice_from_activation_delta": "0.53",
                        "length_controlled_spearman_delta_with_j0_margin": "0.03",
                        "length_controlled_spearman_q": "0.5",
                        "source_sign_consistency": "1.0",
                        "n_sources_with_min_pairs": "4",
                        "candidate_reasons": "",
                    },
                    {
                        "candidate_kind": "random_control",
                        "hidden_layer": "40",
                        "feature_idx": "200",
                        "atoms": "",
                        "bundles": "",
                        "n_pairs": "100",
                        "mean_llm_minus_human_activation": "0.0",
                        "activation_auc_llm_vs_human": "0.49",
                        "auc_j0_llm_choice_from_activation_delta": "0.54",
                        "length_controlled_spearman_delta_with_j0_margin": "0.04",
                        "length_controlled_spearman_q": "0.5",
                        "source_sign_consistency": "1.0",
                        "n_sources_with_min_pairs": "4",
                        "candidate_reasons": "",
                    },
                ],
                fields,
            )

            source_fields = [
                "source_dataset",
                "item_type",
                "hidden_layer",
                "feature_idx",
                "spearman_delta_with_j0_margin",
            ]
            _write_csv(
                run_dir / "candidate_feature_source_alignment.csv",
                [
                    {
                        "source_dataset": "hc3",
                        "item_type": "hc3",
                        "hidden_layer": "34",
                        "feature_idx": "7691",
                        "spearman_delta_with_j0_margin": "0.2",
                    },
                    {
                        "source_dataset": "hc3_plus",
                        "item_type": "general",
                        "hidden_layer": "34",
                        "feature_idx": "7691",
                        "spearman_delta_with_j0_margin": "0.3",
                    },
                ],
                source_fields,
            )

            _write_csv(
                source_dir / "merged_sae_atom_feature_scores.csv",
                [
                    {
                        "hidden_layer": "34",
                        "feature_idx": "7691",
                        "atom": "professional_self_presentation",
                        "source_run": "dense",
                    }
                ],
                ["hidden_layer", "feature_idx", "atom", "source_run"],
            )
            _write_csv(
                source_dir / "merged_sae_bundle_feature_scores.csv",
                [
                    {
                        "hidden_layer": "34",
                        "feature_idx": "7691",
                        "bundle_id": "formal_information_packaging",
                        "member_atoms_hit": "formal_connectives;institutional_impersonality",
                        "source_run": "dense",
                    }
                ],
                ["hidden_layer", "feature_idx", "bundle_id", "member_atoms_hit", "source_run"],
            )

            summary = build_registry(
                alignment_run_dir=run_dir,
                candidate_source_dir=source_dir,
                bundle_config_json=cfg_path,
                out_dir=out_dir,
                controls_per_feature=1,
                seed=7,
            )

            with (out_dir / "bundle_candidate_features.csv").open(newline="", encoding="utf-8") as handle:
                bundle_rows = list(csv.DictReader(handle))
            with (out_dir / "matched_random_feature_controls.csv").open(newline="", encoding="utf-8") as handle:
                controls = list(csv.DictReader(handle))
            self.assertEqual(summary["n_bundle_feature_rows"], 3)
            by_feature = {(row["hidden_layer"], row["feature_idx"]): row for row in bundle_rows}
            self.assertEqual(by_feature[("34", "7691")]["freeze_status"], "intervention_eligible")
            self.assertEqual(by_feature[("40", "15478")]["freeze_status"], "source_sensitive")
            self.assertEqual(by_feature[("39", "15970")]["freeze_status"], "missing_alignment")
            self.assertIn("professional_self_presentation", by_feature[("34", "7691")]["feature_atoms"])
            self.assertTrue(any(row["layer_matched"] == "true" for row in controls))
            self.assertTrue((out_dir / "feature_freeze_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
