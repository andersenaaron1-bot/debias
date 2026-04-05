import json
from pathlib import Path
import tempfile
import unittest

from aisafety.scripts.build_bundle_validation_d2 import main as build_d2_main


class TestBuildBundleValidationD2(unittest.TestCase):
    def test_script_writes_bundle_outputs(self):
        rows = [
            {
                "example_id": "1",
                "group_id": "g1",
                "split": "train",
                "item_type": "paper",
                "dataset": "toy",
                "subset": "paper",
                "source": "human",
                "title": "A",
                "text": "This paper presents a framework. However, prior work suggests improvements.",
            },
            {
                "example_id": "2",
                "group_id": "g2",
                "split": "train",
                "item_type": "paper",
                "dataset": "toy",
                "subset": "paper",
                "source": "llm",
                "title": "B",
                "text": "This study presents a methodology. In conclusion, results show a robust model.",
            },
            {
                "example_id": "3",
                "group_id": "g3",
                "split": "train",
                "item_type": "product",
                "dataset": "toy",
                "subset": "product",
                "source": "human",
                "title": "C",
                "text": "Premium design offers powerful performance. Perfect for travel and daily use.",
            },
            {
                "example_id": "4",
                "group_id": "g4",
                "split": "train",
                "item_type": "movie",
                "dataset": "toy",
                "subset": "movie",
                "source": "llm",
                "title": "D",
                "text": "The story follows a family that confronts danger and fights to survive.",
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            in_jsonl = tmp / "toy.jsonl"
            out_dir = tmp / "out"
            with in_jsonl.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

            import sys

            argv = sys.argv
            sys.argv = [
                "build_bundle_validation_d2",
                "--input-jsonl",
                str(in_jsonl),
                "--out-dir",
                str(out_dir),
                "--bootstrap-samples",
                "20",
                "--null-samples",
                "50",
            ]
            try:
                build_d2_main()
            finally:
                sys.argv = argv

            self.assertTrue((out_dir / "summary.json").exists())
            self.assertTrue((out_dir / "atom_summary.json").exists())
            self.assertTrue((out_dir / "pairwise_cooccurrence.csv").exists())
            self.assertTrue((out_dir / "bundle_validation.json").exists())
            payload = json.loads((out_dir / "bundle_validation.json").read_text(encoding="utf-8"))
            self.assertIn("bundle_validation", payload)
            self.assertIn("academic_formality", payload["bundle_validation"])


if __name__ == "__main__":
    unittest.main()
