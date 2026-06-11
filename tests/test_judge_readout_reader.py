import tempfile
import unittest
from pathlib import Path

import pandas as pd

from aisafety.scripts.read_judge_readout_calibration import (
    _dataset_metrics,
    _diagonal_bootstrap,
    _endpoint_transfer,
)


class JudgeReadoutReaderTests(unittest.TestCase):
    def test_compact_tables(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pd.DataFrame(
                [
                    {
                        "probe_target": "criterion_target",
                        "point_name": "readout_0",
                        "n_pairs": 12,
                        "balanced_accuracy": 0.7,
                        "ci95_low": 0.6,
                        "ci95_high": 0.8,
                    }
                ]
            ).to_csv(root / "point_pair_bootstrap.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "probe_target": "criterion_target",
                        "hidden_layer": 4,
                        "train_point_index": 0,
                        "train_point_name": "readout_0",
                        "test_point_name": "readout_0",
                        "n_pairs": 12,
                        "balanced_accuracy": 0.6,
                        "macro_roc_auc": 0.7,
                    },
                    {
                        "probe_target": "criterion_target",
                        "hidden_layer": 8,
                        "train_point_index": 3,
                        "train_point_name": "readout_2048",
                        "test_point_name": "readout_0",
                        "n_pairs": 12,
                        "balanced_accuracy": 0.8,
                        "macro_roc_auc": 0.9,
                    },
                ]
            ).to_csv(root / "cross_time_metrics.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "group_type": "source_dataset",
                        "group_value": "arc_challenge",
                        "probe_target": "current_choice",
                        "point_name": "readout_128",
                        "n_pairs": 10,
                        "balanced_accuracy": 0.75,
                        "accuracy": 0.8,
                    },
                    {
                        "group_type": "condition_id",
                        "group_value": "switch",
                        "probe_target": "current_choice",
                        "point_name": "readout_128",
                        "n_pairs": 10,
                        "balanced_accuracy": 0.5,
                        "accuracy": 0.5,
                    },
                ]
            ).to_csv(root / "point_metrics_by_group.csv", index=False)

            self.assertEqual(len(_diagonal_bootstrap(root)), 1)
            transfer = _endpoint_transfer(root)
            self.assertEqual(len(transfer), 1)
            self.assertEqual(transfer.iloc[0]["hidden_layer"], 8)
            datasets = _dataset_metrics(root)
            self.assertEqual(len(datasets), 1)
            self.assertEqual(datasets.iloc[0]["group_value"], "arc_challenge")


if __name__ == "__main__":
    unittest.main()
