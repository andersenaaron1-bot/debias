import csv
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from aisafety.scripts.build_d4_dataset_pack import main as build_d4_dataset_pack_main


class BuildD4DatasetPackTest(unittest.TestCase):
    def test_builds_minimal_pack(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            d4_json = root / "d4.json"
            d4_json.write_text(
                json.dumps(
                    {
                        "trace_bundles": [
                            {
                                "bundle_id": "bundle_a",
                                "status": "primary",
                                "member_atoms": ["formal_connectives", "self_mention_markers"],
                                "readout_bundles": ["academic_formality"],
                            }
                        ],
                        "priority_atoms": ["formal_connectives"],
                    }
                ),
                encoding="utf-8",
            )

            atom_probe_jsonl = root / "atom_probe.jsonl"
            atom_probe_jsonl.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "example_id": "a1",
                                "group_id": "g1",
                                "split": "train",
                                "item_type": "paper",
                                "dataset": "bundle_creation",
                                "subset": "paper",
                                "source": "human",
                                "title": "t1",
                                "text": "However, we present our method.",
                                "meta": {"bundle_creation_role": "discovery_core"},
                            }
                        ),
                        json.dumps(
                            {
                                "example_id": "a2",
                                "group_id": "g2",
                                "split": "val",
                                "item_type": "product",
                                "dataset": "bundle_creation",
                                "subset": "product",
                                "source": "llm",
                                "title": "t2",
                                "text": "We clearly provide a powerful solution.",
                                "meta": {"bundle_creation_role": "domain_bolster"},
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            content_anchor_jsonl = root / "content_anchor.jsonl"
            content_anchor_jsonl.write_text(
                json.dumps(
                    {
                        "pair_id": "p1",
                        "source_dataset": "stanfordnlp/SHP-2",
                        "domain": "legaladvice",
                        "prompt": "Prompt",
                        "chosen": "However, we recommend caution.",
                        "rejected": "Buy now! This is perfect.",
                        "meta": {"split": "validation"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            rewrite_dir = root / "rewrites"
            rewrite_dir.mkdir()
            (rewrite_dir / "academic_formality.jsonl").write_text(
                json.dumps(
                    {
                        "dimension": "academic_formality",
                        "label": "academic_formal",
                        "seed_source": "Laurito/human",
                        "seed_id": 1,
                        "seed_text": "However, we present the results.",
                        "generated_text": "We show the results in a formal way.",
                        "model": "gpt-test",
                        "meta": {"item_type": "paper", "source": "human", "title": "x"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            d3_root = root / "artifacts" / "style_groups"
            run_dir = d3_root / "d3_j0_anchor_v1_h100compact"
            run_dir.mkdir(parents=True)
            (run_dir / "pair_level_inputs.csv").write_text(
                "pair_key,item_type,title,human_text,llm_text,y_llm_chosen,llm_margin_pair\n"
                "k1,paper,t1,human text,llm text,1,0.5\n",
                encoding="utf-8",
            )
            with (run_dir / "atom_effects.tsv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["name", "signed_effect_z"], delimiter="\t")
                writer.writeheader()
                writer.writerow({"name": "formal_connectives", "signed_effect_z": "-0.1"})
            with (run_dir / "bundle_effects.tsv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["name", "signed_effect_z"], delimiter="\t")
                writer.writeheader()
                writer.writerow({"name": "academic_formality", "signed_effect_z": "-0.2"})
            (run_dir / "text_atom_scores.csv").write_text(
                "text_id,text,item_type,source,title,word_count,formal_connectives,self_mention_markers\n"
                "t1,However we present,paper,human,t1,3,1.0,0.0\n",
                encoding="utf-8",
            )
            (run_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "n_pair_rows": 1,
                        "by_item_type": {"paper": {"llm_choice_rate": 0.5}},
                    }
                ),
                encoding="utf-8",
            )

            config_json = root / "config.json"
            config_json.write_text(
                json.dumps(
                    {
                        "name": "d4_dataset_pack_test",
                        "d4_ontology_json": "d4.json",
                        "atom_probe_jsonl": "atom_probe.jsonl",
                        "content_anchor_jsonl": "content_anchor.jsonl",
                        "rewrite_glob": "rewrites/*.jsonl",
                        "d3_root": "artifacts/style_groups",
                        "run_ids": ["j0_anchor_v1_h100compact"],
                        "canonical_text_score_run": "j0_anchor_v1_h100compact",
                        "out_dir": "out",
                    }
                ),
                encoding="utf-8",
            )

            build_d4_dataset_pack_main_args = [
                "--config-json",
                str(config_json),
                "--workspace-root",
                str(root),
            ]
            with mock.patch("sys.argv", ["build_d4_dataset_pack.py", *build_d4_dataset_pack_main_args]):
                build_d4_dataset_pack_main()

            out_dir = root / "out"
            self.assertTrue((out_dir / "manifest.json").is_file())
            self.assertTrue((out_dir / "atom_probe_set.jsonl").is_file())
            self.assertTrue((out_dir / "content_anchor_set.jsonl").is_file())
            self.assertTrue((out_dir / "rewrite_control_set.jsonl").is_file())
            self.assertTrue((out_dir / "laurito_pair_runs.csv").is_file())

            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["name"], "d4_dataset_pack_test")
            self.assertIn("bundle_a", manifest["trace_bundles"])


if __name__ == "__main__":
    unittest.main()
