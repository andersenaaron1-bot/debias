import json
from pathlib import Path
import tempfile
import unittest

from aisafety.data.bundle_corpus import (
    iter_excerpt_jsonl_records,
    load_bundle_creation_spec,
    materialize_bundle_creation_records,
)
from aisafety.scripts.build_bundle_creation_corpus import main as build_bundle_corpus_main


class TestBundleCorpus(unittest.TestCase):
    def test_iter_excerpt_jsonl_records_reads_normalized_schema(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pubmed.jsonl"
            rows = [
                {"title": "A", "text": "This paper presents a method.", "source": "human", "item_type": "paper"},
                {"title": "B", "text": "This study proposes a framework.", "source": "llm", "item_type": "paper"},
            ]
            with path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            records = iter_excerpt_jsonl_records(
                path,
                dataset_id="pubmed_abstracts",
                item_type_default="paper",
                subset_default="paper",
                source_default="human",
                role="domain_bolster",
                stratum_id="B1",
                holdout_from_discovery=False,
                seed=7,
            )
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0].dataset, "pubmed_abstracts")
            self.assertIn("bundle_creation_role", records[0].meta)

    def test_materialize_bundle_creation_records(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            hc3_dir = root / "HC3"
            hc3_dir.mkdir()
            (hc3_dir / "finance.jsonl").write_text(
                json.dumps(
                    {
                        "question": "What is inflation?",
                        "human_answers": ["A rise in prices over time."],
                        "chatgpt_answers": ["Inflation is a general increase in prices over time."],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            pubmed_jsonl = root / "pubmed.jsonl"
            pubmed_jsonl.write_text(
                json.dumps({"title": "P1", "text": "This paper presents a method.", "source": "human"}) + "\n",
                encoding="utf-8",
            )

            spec = {
                "strata": [
                    {
                        "stratum_id": "A1",
                        "role": "discovery_core",
                        "datasets": [{"dataset_id": "hc3", "target_texts_preferred": 100}],
                    },
                    {
                        "stratum_id": "B1",
                        "role": "domain_bolster",
                        "datasets": [{"dataset_id": "pubmed_abstracts", "target_texts_preferred": 100}],
                    },
                    {
                        "stratum_id": "D1",
                        "role": "ecological_validation",
                        "datasets": [{"dataset_id": "laurito_paper_product_movie"}],
                    },
                ]
            }

            records, summary = materialize_bundle_creation_records(
                spec,
                hc3_dir=hc3_dir,
                pubmed_jsonl=pubmed_jsonl,
                include_laurito_ecology=False,
                seed=5,
            )
            self.assertEqual(len(records), 3)
            self.assertIn("discovery_core", summary["by_role"])
            self.assertIn("domain_bolster", summary["by_role"])

    def test_build_bundle_creation_corpus_script_writes_role_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            spec_path = root / "spec.json"
            out_dir = root / "out"
            hc3_dir = root / "HC3"
            hc3_dir.mkdir()
            (hc3_dir / "finance.jsonl").write_text(
                json.dumps(
                    {
                        "question": "What is inflation?",
                        "human_answers": ["A rise in prices over time."],
                        "chatgpt_answers": ["Inflation is a general increase in prices over time."],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            spec = {
                "strata": [
                    {
                        "stratum_id": "A1",
                        "role": "discovery_core",
                        "datasets": [{"dataset_id": "hc3", "target_texts_preferred": 100}],
                    }
                ]
            }
            spec_path.write_text(json.dumps(spec), encoding="utf-8")

            import sys

            argv = sys.argv
            sys.argv = [
                "build_bundle_creation_corpus",
                "--spec-json",
                str(spec_path),
                "--out-dir",
                str(out_dir),
                "--hc3-dir",
                str(hc3_dir),
                "--no-laurito-ecology",
            ]
            try:
                build_bundle_corpus_main()
            finally:
                sys.argv = argv

            self.assertTrue((out_dir / "all_records.jsonl").exists())
            self.assertTrue((out_dir / "discovery_core.jsonl").exists())
            self.assertTrue((out_dir / "summary.json").exists())
            payload = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertIn("role_outputs", payload)
            self.assertIn("discovery_core", payload["role_outputs"])


if __name__ == "__main__":
    unittest.main()
