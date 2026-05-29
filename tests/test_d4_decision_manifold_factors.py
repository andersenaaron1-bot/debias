import unittest

import pandas as pd

from aisafety.scripts.analyze_d4_decision_manifold_factors import analyze


class D4DecisionManifoldFactorsTests(unittest.TestCase):
    def test_analyzes_small_matrix(self) -> None:
        df = pd.DataFrame(
            {
                "unit_id": [f"u{i}" for i in range(6)],
                "source_dataset": ["s"] * 6,
                "a": [0, 1, 2, 3, 4, 5],
                "b": [5, 4, 3, 2, 1, 0],
                "c": [0, 0, 1, 1, 2, 2],
            }
        )
        outputs, manifest = analyze(
            df,
            n_components=2,
            min_feature_coverage=0.5,
            min_unit_coverage=0.5,
            include_feature_regex=[],
            exclude_feature_regex=[],
            stratify_by=["source_dataset"],
            center_within_strata=False,
            top_k=2,
            sparse_alpha=0.1,
            seed=1234,
        )
        self.assertEqual(manifest["n_features_used"], 3)
        self.assertIn("component_summary", outputs)
        self.assertIn("component_loadings", outputs)
        self.assertIn("top_units", outputs)
        self.assertIn("stratum_feature_summary", outputs)
        self.assertIn("stratum_cancellation_summary", outputs)
        self.assertEqual(set(outputs["pca_scores"].columns), {"unit_id", "pc1", "pc2"})

    def test_center_within_strata_runs(self) -> None:
        df = pd.DataFrame(
            {
                "unit_id": [f"u{i}" for i in range(8)],
                "source_dataset": ["a", "a", "a", "a", "b", "b", "b", "b"],
                "x": [0, 1, 2, 3, 10, 11, 12, 13],
                "y": [3, 2, 1, 0, 13, 12, 11, 10],
            }
        )
        outputs, manifest = analyze(
            df,
            n_components=1,
            min_feature_coverage=0.5,
            min_unit_coverage=0.5,
            include_feature_regex=[],
            exclude_feature_regex=[],
            stratify_by=["source_dataset"],
            center_within_strata=True,
            top_k=2,
            sparse_alpha=0.1,
            seed=1234,
        )
        self.assertTrue(manifest["center_within_strata"])
        self.assertEqual(manifest["n_features_used"], 2)
        self.assertIn("pca_scores", outputs)


if __name__ == "__main__":
    unittest.main()
