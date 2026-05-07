import csv
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch
import torch.nn as nn

from aisafety.mech.interventions import (
    FeatureSpec,
    assign_quantile_bins,
    load_bundle_feature_specs,
    load_matched_random_control_specs,
    register_feature_damping_hooks,
    remove_hooks,
)


class FakeLayer(nn.Module):
    def forward(self, x):
        return x


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([FakeLayer()])


class FakeSae:
    def __init__(self):
        self.W_dec = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    def encode(self, x):
        return torch.relu(x)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class D4FeaturePerturbationTests(unittest.TestCase):
    def test_assign_quantile_bins_returns_expected_extremes(self) -> None:
        bins = assign_quantile_bins(np.asarray([0.0, 1.0, 2.0, 3.0]), high_low_frac=0.25)
        self.assertEqual(bins.count("low"), 1)
        self.assertEqual(bins.count("high"), 1)
        self.assertEqual(bins.count("middle"), 2)

    def test_registry_loaders_filter_bundle_status_and_target_controls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_csv(
                root / "bundle_candidate_features.csv",
                [
                    {
                        "bundle_id": "formal",
                        "hidden_layer": 40,
                        "feature_idx": 15478,
                        "sae_release": "rel",
                        "sae_id": "layer_39/width_16k/canonical",
                        "aggregation": "max",
                        "freeze_status": "intervention_eligible",
                        "length_controlled_spearman_delta_with_j0_margin": "0.4",
                    },
                    {
                        "bundle_id": "formal",
                        "hidden_layer": 34,
                        "feature_idx": 7691,
                        "sae_release": "rel",
                        "sae_id": "layer_33/width_16k/canonical",
                        "aggregation": "max",
                        "freeze_status": "source_sensitive",
                        "length_controlled_spearman_delta_with_j0_margin": "0.5",
                    },
                ],
            )
            _write_csv(
                root / "matched_random_feature_controls.csv",
                [
                    {
                        "bundle_id": "formal",
                        "target_hidden_layer": 40,
                        "target_feature_idx": 15478,
                        "control_rank": 1,
                        "control_hidden_layer": 40,
                        "control_feature_idx": 111,
                    },
                    {
                        "bundle_id": "formal",
                        "target_hidden_layer": 34,
                        "target_feature_idx": 7691,
                        "control_rank": 1,
                        "control_hidden_layer": 34,
                        "control_feature_idx": 222,
                    },
                ],
            )
            specs = load_bundle_feature_specs(root, bundle_id="formal")
            self.assertEqual([(s.hidden_layer, s.feature_idx) for s in specs], [(40, 15478)])
            controls = load_matched_random_control_specs(
                root,
                bundle_id="formal",
                allowed_target_keys={(40, 15478)},
            )
            self.assertEqual([(s.hidden_layer, s.feature_idx) for s in controls], [(40, 111)])

    def test_feature_damping_hook_subtracts_decoder_contribution(self) -> None:
        model = FakeModel()
        spec = FeatureSpec(
            bundle_id="formal",
            hidden_layer=1,
            feature_idx=0,
            sae_release="rel",
            sae_id="layer_0/width_16k/canonical",
        )
        handles = register_feature_damping_hooks(
            model,
            features_by_layer={1: [spec]},
            sae_by_layer={1: FakeSae()},
            strength=0.5,
        )
        try:
            x = torch.tensor([[[2.0, -1.0], [0.5, 3.0]]])
            y = model.model.layers[0](x)
            expected = torch.tensor([[[1.0, -1.0], [0.25, 3.0]]])
            self.assertTrue(torch.allclose(y, expected))
        finally:
            remove_hooks(handles)


if __name__ == "__main__":
    unittest.main()
