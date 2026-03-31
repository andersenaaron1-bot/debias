import unittest

from aisafety.features.surface_cues import SURFACE_FEATURE_NAMES, extract_surface_features


class TestSurfaceCues(unittest.TestCase):
    def test_extract_surface_features_contains_expected_keys(self):
        feats = extract_surface_features("This paper presents a method. However, it may improve results.")
        self.assertEqual(set(feats), set(SURFACE_FEATURE_NAMES))
        self.assertGreater(feats["word_count"], 0)
        self.assertGreaterEqual(feats["academic_phrase_rate"], 0.0)

    def test_extract_surface_features_detects_lexicons(self):
        text = (
            "This paper presents a model. We propose a method that may improve results. "
            "Overall, it is important to note the policy guidance."
        )
        feats = extract_surface_features(text)
        self.assertGreater(feats["academic_phrase_rate"], 0.0)
        self.assertGreater(feats["hedge_rate"], 0.0)
        self.assertGreater(feats["template_phrase_rate"], 0.0)
        self.assertGreater(feats["safety_phrase_rate"], 0.0)

    def test_extract_surface_features_detects_promotional_and_bullets(self):
        text = "- Premium design\n- Perfect for travel\nEnjoy versatile performance!"
        feats = extract_surface_features(text)
        self.assertGreater(feats["bullet_line_ratio"], 0.0)
        self.assertGreater(feats["promo_phrase_rate"], 0.0)
        self.assertGreater(feats["exclamation_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
