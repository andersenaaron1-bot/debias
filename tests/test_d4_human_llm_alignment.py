import csv
import json
from argparse import Namespace
from pathlib import Path
import tempfile
import unittest

from aisafety.scripts.build_d4_human_llm_alignment_pairs import build_pairs_from_records
from aisafety.scripts.run_d4_candidate_feature_pair_alignment import build_candidate_registry


class D4HumanLlmAlignmentTest(unittest.TestCase):
    def test_pair_builder_groups_human_and_llm_records(self) -> None:
        rows = [
            {
                "example_id": "h1",
                "group_id": "g1",
                "split": "train",
                "item_type": "hc3",
                "dataset": "hc3",
                "subset": "finance",
                "source": "human",
                "title": "Q",
                "text": "human answer with enough tokens for the alignment pair builder",
                "meta": {"bundle_creation_dataset_id": "hc3", "bundle_creation_role": "discovery_core"},
            },
            {
                "example_id": "l1",
                "group_id": "g1",
                "split": "train",
                "item_type": "hc3",
                "dataset": "hc3",
                "subset": "finance",
                "source": "llm",
                "title": "Q",
                "text": "llm answer with enough tokens for the alignment pair builder",
                "generator": "chatgpt",
                "meta": {"bundle_creation_dataset_id": "hc3", "bundle_creation_role": "discovery_core"},
            },
        ]
        pairs, summary = build_pairs_from_records(
            rows,
            include_roles={"discovery_core"},
            include_datasets=set(),
            exclude_datasets=set(),
            min_tokens=3,
            max_tokens=100,
            max_human_per_group=1,
            max_llm_per_group=1,
            cap_strategy="dataset",
            max_pairs_per_dataset=0,
            max_total_pairs=0,
            seed=7,
        )
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["source_dataset"], "hc3")
        self.assertEqual(pairs[0]["llm_generator"], "chatgpt")
        self.assertEqual(summary["by_dataset"], {"hc3": 1})

    def test_pair_builder_balances_global_cap_by_dataset_subset(self) -> None:
        rows = []
        for subset, n_groups in (("finance", 5), ("medicine", 5)):
            for idx in range(n_groups):
                group_id = f"{subset}-{idx}"
                for source in ("human", "llm"):
                    rows.append(
                        {
                            "example_id": f"{source}-{group_id}",
                            "group_id": group_id,
                            "split": "train",
                            "item_type": "hc3",
                            "dataset": "hc3",
                            "subset": subset,
                            "source": source,
                            "title": group_id,
                            "text": f"{source} answer for {group_id} with enough tokens to pass filters",
                            "meta": {
                                "bundle_creation_dataset_id": "hc3",
                                "bundle_creation_role": "discovery_core",
                            },
                        }
                    )

        pairs, summary = build_pairs_from_records(
            rows,
            include_roles={"discovery_core"},
            include_datasets=set(),
            exclude_datasets=set(),
            min_tokens=3,
            max_tokens=100,
            max_human_per_group=1,
            max_llm_per_group=1,
            cap_strategy="dataset_subset",
            max_pairs_per_dataset=0,
            max_total_pairs=6,
            seed=7,
        )

        self.assertEqual(len(pairs), 6)
        self.assertEqual(summary["by_dataset_subset"], {"hc3::finance": 3, "hc3::medicine": 3})

    def test_candidate_registry_keeps_broad_weak_decision_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            atom_path = root / "merged_sae_atom_feature_scores.csv"
            fields = [
                "atom",
                "status",
                "hidden_layer",
                "sae_layer",
                "sae_release",
                "sae_id",
                "aggregation",
                "feature_idx",
                "val_auc",
                "test_auc",
                "abs_cohen_d",
                "laurito_spearman_with_atom_score",
                "auc_llm_choice",
                "spearman_with_llm_margin",
                "train_pos_activation_rate",
                "train_neg_activation_rate",
                "source_run",
            ]
            with atom_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "atom": "epistemic_hedging",
                        "status": "ok",
                        "hidden_layer": "30",
                        "sae_layer": "29",
                        "sae_release": "rel",
                        "sae_id": "layer_29/width_16k/canonical",
                        "aggregation": "max",
                        "feature_idx": "5394",
                        "val_auc": "0.87",
                        "test_auc": "0.82",
                        "abs_cohen_d": "1.69",
                        "laurito_spearman_with_atom_score": "0.07",
                        "auc_llm_choice": "0.52",
                        "spearman_with_llm_margin": "0.03",
                        "train_pos_activation_rate": "0.69",
                        "train_neg_activation_rate": "0.02",
                        "source_run": "dense_midlate",
                    }
                )
            (root / "merged_sae_bundle_feature_scores.csv").write_text(
                "bundle_id,hidden_layer,sae_layer,sae_release,sae_id,aggregation,feature_idx,"
                "n_member_atoms_hit,member_atoms_hit,mean_abs_cohen_d,max_val_auc,"
                "mean_laurito_abs_spearman,source_run\n",
                encoding="utf-8",
            )

            args = Namespace(
                atoms="",
                sae_release="rel",
                min_atom_val_auc=0.75,
                min_atom_test_auc=0.7,
                min_abs_cohen_d=0.8,
                min_laurito_abs_spearman=0.08,
                min_laurito_auc_delta=0.03,
                min_bundle_member_atoms=2,
                selected_layers="",
                max_features_per_layer=10,
                max_candidates=20,
            )
            registry = build_candidate_registry(source_dir=root, args=args)
            self.assertEqual(len(registry), 1)
            self.assertEqual(registry[0]["atoms"], "epistemic_hedging")
            self.assertIn("atom_val_auc", registry[0]["candidate_reasons"])


if __name__ == "__main__":
    unittest.main()
