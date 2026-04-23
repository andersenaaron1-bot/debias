import unittest

from aisafety.scripts.run_d4_sae_feature_analysis import (
    format_sae_id,
    hidden_layer_to_sae_layer,
    _parse_int_list,
)


class RunD4SaeFeatureAnalysisHelpersTest(unittest.TestCase):
    def test_hidden_layer_to_sae_layer_maps_hf_hidden_state_to_block(self) -> None:
        self.assertEqual(hidden_layer_to_sae_layer(1), 0)
        self.assertEqual(hidden_layer_to_sae_layer(42), 41)

    def test_hidden_layer_to_sae_layer_rejects_embedding_index(self) -> None:
        with self.assertRaises(ValueError):
            hidden_layer_to_sae_layer(0)

    def test_format_sae_id_uses_sae_layer(self) -> None:
        self.assertEqual(
            format_sae_id("layer_{sae_layer}/width_16k/canonical", hidden_layer=40),
            "layer_39/width_16k/canonical",
        )

    def test_parse_int_list_deduplicates_and_preserves_order(self) -> None:
        self.assertEqual(_parse_int_list("42, 1, 8, 8"), [42, 1, 8])


if __name__ == "__main__":
    unittest.main()
