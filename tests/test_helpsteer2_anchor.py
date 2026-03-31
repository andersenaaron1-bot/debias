import unittest

from aisafety.scripts.build_helpsteer2_anchor import _convert_row, _parse_weights


class TestHelpSteer2Anchor(unittest.TestCase):
    def test_parse_weights_normalizes(self):
        weights = _parse_weights(
            "helpfulness=3,correctness=3,coherence=2,complexity=1,verbosity=1"
        )
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)
        self.assertGreater(weights["helpfulness"], weights["complexity"])

    def test_convert_row_builds_targets(self):
        row = {
            "prompt": "Explain recursion.",
            "response": "Recursion is a function calling itself.",
            "helpfulness": 4,
            "correctness": 3,
            "coherence": 4,
            "complexity": 2,
            "verbosity": 1,
        }
        weights = _parse_weights(
            "helpfulness=0.30,correctness=0.30,coherence=0.20,complexity=0.10,verbosity=0.10"
        )
        out = _convert_row(
            row,
            dataset_id="nvidia/HelpSteer2",
            split="train",
            normalize_targets=True,
            weights=weights,
        )
        self.assertIsNotNone(out)
        self.assertAlmostEqual(out["attribute_targets"]["helpfulness"], 1.0, places=6)
        self.assertGreater(out["utility_target"], 0.0)
        self.assertLessEqual(out["utility_target"], 1.0)


if __name__ == "__main__":
    unittest.main()
