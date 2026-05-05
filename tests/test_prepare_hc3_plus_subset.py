import json
from pathlib import Path
import tempfile
import unittest

from aisafety.scripts.prepare_hc3_plus_subset import main as prepare_main


class PrepareHc3PlusSubsetTest(unittest.TestCase):
    def test_prepare_from_paired_fields(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data_dir = root / "data" / "en"
            data_dir.mkdir(parents=True)
            raw = data_dir / "test_hc3_si.jsonl"
            raw.write_text(
                json.dumps(
                    {
                        "task": "summarization",
                        "source_text": "Long source article",
                        "human": "Human summary with enough words to pass the filter",
                        "chatgpt": "ChatGPT summary with enough words to pass the filter",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            out_jsonl = root / "out.jsonl"

            import sys

            old_argv = sys.argv
            sys.argv = [
                "prepare_hc3_plus_subset",
                "--input",
                str(root),
                "--out-jsonl",
                str(out_jsonl),
                "--min-tokens",
                "3",
                "--file-glob",
                "*.jsonl",
                "--max-pairs",
                "10",
            ]
            try:
                prepare_main()
            finally:
                sys.argv = old_argv

            rows = [json.loads(line) for line in out_jsonl.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["source"] for row in rows}, {"human", "llm"})
            self.assertEqual({row["subset"] for row in rows}, {"summarization"})

    def test_prepare_from_classifier_rows(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data_dir = root / "data" / "en"
            data_dir.mkdir(parents=True)
            raw = data_dir / "test_hc3_si.jsonl"
            raw.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "task": "paraphrasing",
                                "source_text": "Original sentence",
                                "label": 0,
                                "text": "Human paraphrase with enough words to pass the filter",
                            }
                        ),
                        json.dumps(
                            {
                                "task": "paraphrasing",
                                "source_text": "Original sentence",
                                "label": 1,
                                "text": "ChatGPT paraphrase with enough words to pass the filter",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            out_jsonl = root / "out.jsonl"

            import sys

            old_argv = sys.argv
            sys.argv = [
                "prepare_hc3_plus_subset",
                "--input",
                str(root),
                "--out-jsonl",
                str(out_jsonl),
                "--min-tokens",
                "3",
                "--file-glob",
                "*.jsonl",
                "--max-pairs",
                "10",
            ]
            try:
                prepare_main()
            finally:
                sys.argv = old_argv

            rows = [json.loads(line) for line in out_jsonl.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["source"] for row in rows}, {"human", "llm"})
            self.assertEqual({row["subset"] for row in rows}, {"paraphrasing"})

    def test_prepare_from_adjacent_detector_rows_without_group_field(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data_dir = root / "data" / "en"
            data_dir.mkdir(parents=True)
            raw = data_dir / "test_hc3_QA.jsonl"
            raw.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "label": 0,
                                "text": "First human answer with enough words to pass the filter",
                            }
                        ),
                        json.dumps(
                            {
                                "label": 0,
                                "text": "Second human answer with enough words to pass the filter",
                            }
                        ),
                        json.dumps(
                            {
                                "label": 1,
                                "text": "ChatGPT answer with enough words to pass the filter",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            out_jsonl = root / "out.jsonl"

            import sys

            old_argv = sys.argv
            sys.argv = [
                "prepare_hc3_plus_subset",
                "--input",
                str(root),
                "--out-jsonl",
                str(out_jsonl),
                "--min-tokens",
                "3",
                "--file-glob",
                "*.jsonl",
                "--max-pairs",
                "10",
            ]
            try:
                prepare_main()
            finally:
                sys.argv = old_argv

            rows = [json.loads(line) for line in out_jsonl.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["source"] for row in rows}, {"human", "llm"})
            self.assertEqual({row["subset"] for row in rows}, {"test_hc3_qa"})


if __name__ == "__main__":
    unittest.main()
