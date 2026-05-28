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
            top_k=2,
            sparse_alpha=0.1,
            seed=1234,
        )
        self.assertEqual(manifest["n_features_used"], 3)
        self.assertIn("component_summary", outputs)
        self.assertIn("component_loadings", outputs)
        self.assertIn("top_units", outputs)
        self.assertEqual(set(outputs["pca_scores"].columns), {"unit_id", "pc1", "pc2"})


if __name__ == "__main__":
    unittest.main()
