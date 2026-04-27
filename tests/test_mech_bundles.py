import unittest

import pandas as pd

from aisafety.mech.bundles import build_bundle_feature_scores, normalize_trace_bundles


class MechBundlesTest(unittest.TestCase):
    def test_normalize_trace_bundles_accepts_ontology_list(self) -> None:
        bundles = normalize_trace_bundles(
            [
                {
                    "bundle_id": "formal_information_packaging",
                    "member_atoms": ["formal_connectives"],
                }
            ]
        )
        self.assertEqual(list(bundles), ["formal_information_packaging"])

    def test_build_bundle_feature_scores_groups_shared_features(self) -> None:
        feature_rows = pd.DataFrame(
            [
                {
                    "status": "ok",
                    "atom": "formal_connectives",
                    "hidden_layer": 42,
                    "sae_layer": 41,
                    "sae_release": "release",
                    "sae_id": "layer_41/width_16k/canonical",
                    "aggregation": "max",
                    "feature_idx": 10,
                    "abs_cohen_d": 1.0,
                    "val_auc": 0.9,
                    "laurito_spearman_with_atom_score": 0.4,
                },
                {
                    "status": "ok",
                    "atom": "nominalization_patterns",
                    "hidden_layer": 42,
                    "sae_layer": 41,
                    "sae_release": "release",
                    "sae_id": "layer_41/width_16k/canonical",
                    "aggregation": "max",
                    "feature_idx": 10,
                    "abs_cohen_d": 0.5,
                    "val_auc": 0.8,
                    "laurito_spearman_with_atom_score": 0.2,
                },
            ]
        )
        bundle_rows = build_bundle_feature_scores(
            feature_rows,
            {
                "formal_information_packaging": {
                    "status": "primary",
                    "member_atoms": ["formal_connectives", "nominalization_patterns"],
                }
            },
        )
        self.assertEqual(len(bundle_rows), 1)
        self.assertEqual(int(bundle_rows.iloc[0]["n_member_atoms_hit"]), 2)
        self.assertEqual(
            bundle_rows.iloc[0]["member_atoms_hit"],
            "formal_connectives;nominalization_patterns",
        )


if __name__ == "__main__":
    unittest.main()

