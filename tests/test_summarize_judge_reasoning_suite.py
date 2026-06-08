from pathlib import Path
import tempfile
import unittest

import pandas as pd

from aisafety.scripts.summarize_judge_reasoning_suite import _read_with_label


class SummarizeJudgeReasoningSuiteTests(unittest.TestCase):
    def test_read_with_label_adds_missing_run_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame([{"metric": 1.0}]).to_csv(root / "metrics.csv", index=False)
            frame = _read_with_label(root, "metrics.csv", "base")
            self.assertEqual(frame.columns[0], "run_label")
            self.assertEqual(frame.loc[0, "run_label"], "base")

    def test_read_with_label_overwrites_existing_run_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame([{"run_label": "internal", "metric": 1.0}]).to_csv(
                root / "metrics.csv",
                index=False,
            )
            frame = _read_with_label(root, "metrics.csv", "it")
            self.assertEqual(frame.columns.tolist().count("run_label"), 1)
            self.assertEqual(frame.columns[0], "run_label")
            self.assertEqual(frame.loc[0, "run_label"], "it")


if __name__ == "__main__":
    unittest.main()
