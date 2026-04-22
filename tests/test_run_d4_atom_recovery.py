import unittest

import pandas as pd

from aisafety.scripts.run_d4_atom_recovery import _build_atom_label_frame, _flatten_content_pairs, select_hidden_layers


class RunD4AtomRecoveryHelpersTest(unittest.TestCase):
    def test_select_hidden_layers_includes_stride_tail_and_final(self) -> None:
        self.assertEqual(select_hidden_layers(10, stride=4, tail_layers=3), [1, 4, 8, 9, 10])

    def test_build_atom_label_frame_assigns_binary_labels(self) -> None:
        df = pd.DataFrame(
            [
                {"split": "train", "item_type": "paper", "atom_scores": {"formal_connectives": 0.1}},
                {"split": "train", "item_type": "paper", "atom_scores": {"formal_connectives": 0.2}},
                {"split": "train", "item_type": "paper", "atom_scores": {"formal_connectives": 0.9}},
                {"split": "val", "item_type": "paper", "atom_scores": {"formal_connectives": 0.05}},
                {"split": "val", "item_type": "paper", "atom_scores": {"formal_connectives": 1.2}},
            ]
        )
        labeled = _build_atom_label_frame(df, atoms=["formal_connectives"], q=0.8)
        labels = labeled["formal_connectives__label"].tolist()
        self.assertIn(0, labels)
        self.assertIn(1, labels)

    def test_flatten_content_pairs_splits_missing_pair_ids(self) -> None:
        rows = [
            {
                "source_dataset": "stanfordnlp/SHP-2",
                "domain": f"domain-{i % 3}",
                "prompt": f"Prompt {i}",
                "chosen_text": f"chosen {i}",
                "rejected_text": f"rejected {i}",
            }
            for i in range(200)
        ]
        flattened = _flatten_content_pairs(rows, seed=1234, max_pairs=200)
        self.assertEqual(len(flattened), 400)
        self.assertGreater((flattened["split"] == "val").sum(), 0)
        self.assertGreater((flattened["split"] == "test").sum(), 0)


if __name__ == "__main__":
    unittest.main()
