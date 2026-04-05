import unittest

from aisafety.ontology.atoms import extract_atom_scores, get_atom_specs


class TestOntologyAtoms(unittest.TestCase):
    def test_specs_cover_inventory(self):
        specs = get_atom_specs()
        self.assertIn("formal_connectives", specs)
        self.assertIn("academic_abstract_register", specs)
        self.assertTrue(specs["formal_connectives"].validation_subset)
        self.assertFalse(specs["academic_abstract_register"].validation_subset)

    def test_academic_atoms_fire_on_abstract_like_text(self):
        text = (
            "This paper presents a framework for evaluation. However, prior work suggests "
            "that the method may improve results. In conclusion, the findings indicate a robust model."
        )
        scores = extract_atom_scores(text)
        self.assertGreater(scores["formal_connectives"], 0.0)
        self.assertGreater(scores["technical_abstract_nouns"], 0.0)
        self.assertGreater(scores["background_method_result_script"], 0.0)
        self.assertGreater(scores["academic_abstract_register"], 0.0)

    def test_promotional_and_narrative_atoms_fire(self):
        promo = "Premium design offers powerful performance. Perfect for travel, it delivers exceptional value."
        narrative = "The story follows a family that confronts a mystery and struggles to survive."
        promo_scores = extract_atom_scores(promo)
        narrative_scores = extract_atom_scores(narrative)
        self.assertGreater(promo_scores["promotional_adjectives"], 0.0)
        self.assertGreater(promo_scores["feature_benefit_call_to_value_script"], 0.0)
        self.assertGreater(narrative_scores["narrative_engagement_stance"], 0.0)
        self.assertGreater(narrative_scores["movie_synopsis_register"], 0.0)


if __name__ == "__main__":
    unittest.main()
