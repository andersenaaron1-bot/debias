import json
import tempfile
import unittest
from pathlib import Path

from aisafety.data.cue_corpus import (
    _hllmc2_answer_fields,
    assign_group_split,
    iter_hc3_records,
    iter_local_domain_records,
    limit_records_by_item_type,
    summarize_cue_corpus,
)
from aisafety.data.domains import DomainConfig


class TestCueCorpus(unittest.TestCase):
    def test_assign_group_split_is_deterministic(self):
        a = assign_group_split("g1", seed=7, train_frac=0.8, val_frac=0.1)
        b = assign_group_split("g1", seed=7, train_frac=0.8, val_frac=0.1)
        self.assertEqual(a, b)

    def test_iter_local_domain_records(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            human_dir = root / "paper" / "human"
            llm_dir = root / "paper" / "llm"
            human_dir.mkdir(parents=True)
            llm_dir.mkdir(parents=True)

            human_payload = {
                "title": "Paper A",
                "abstract": "This paper presents a method.",
            }
            llm_payload = {
                "title": "Paper A",
                "descriptions": ["This paper presents a method.", "We propose a novel approach."],
                "llm_engine": "test-llm",
                "generation_prompt_nickname": "write_xml_paper_abstract",
            }
            (human_dir / "paper_a.json").write_text(json.dumps(human_payload), encoding="utf-8")
            (llm_dir / "paper_a_llm.json").write_text(json.dumps(llm_payload), encoding="utf-8")

            cfg = DomainConfig(item_type="paper", human_dir=human_dir, llm_dir=llm_dir)
            records = iter_local_domain_records(cfg, seed=11)
            self.assertEqual(len(records), 3)
            self.assertEqual({r.source for r in records}, {"human", "llm"})
            self.assertTrue(all(r.group_id.startswith("local::paper::") for r in records))

    def test_iter_local_domain_records_respects_variant_cap(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            human_dir = root / "product" / "human"
            llm_dir = root / "product" / "llm"
            human_dir.mkdir(parents=True)
            llm_dir.mkdir(parents=True)

            human_payload = {"title": "Product A", "descriptions": ["Human description."]}
            llm_payload = {"title": "Product A", "descriptions": ["LLM one.", "LLM two.", "LLM three."]}
            (human_dir / "product_a.json").write_text(json.dumps(human_payload), encoding="utf-8")
            (llm_dir / "product_a_llm.json").write_text(json.dumps(llm_payload), encoding="utf-8")

            cfg = DomainConfig(item_type="product", human_dir=human_dir, llm_dir=llm_dir)
            records = iter_local_domain_records(cfg, seed=11, max_variants_per_group_source=1)
            llm_records = [r for r in records if r.source == "llm"]
            self.assertEqual(len(llm_records), 1)

    def test_iter_local_domain_records_caps_across_multiple_llm_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            human_dir = root / "paper" / "human"
            llm_dir = root / "paper" / "llm"
            human_dir.mkdir(parents=True)
            llm_dir.mkdir(parents=True)

            (human_dir / "paper_a.json").write_text(
                json.dumps({"title": "Paper A", "abstract": "Human abstract."}),
                encoding="utf-8",
            )
            for idx in range(3):
                payload = {
                    "title": "Paper A",
                    "descriptions": [f"LLM variant {idx}a.", f"LLM variant {idx}b."],
                    "llm_engine": "mock",
                }
                (llm_dir / f"paper_a_{idx}.json").write_text(json.dumps(payload), encoding="utf-8")

            cfg = DomainConfig(item_type="paper", human_dir=human_dir, llm_dir=llm_dir)
            records = iter_local_domain_records(cfg, seed=11, max_variants_per_group_source=2)
            llm_records = [r for r in records if r.source == "llm"]
            self.assertEqual(len(llm_records), 2)

    def test_iter_hc3_records(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "finance.jsonl"
            row = {
                "question": "What is inflation?",
                "human_answers": ["A rise in prices over time."],
                "chatgpt_answers": ["Inflation is the general rise in prices over time."],
            }
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            records = iter_hc3_records(root, seed=5)
            self.assertEqual(len(records), 2)
            self.assertEqual({r.source for r in records}, {"human", "llm"})
            self.assertTrue(all(r.item_type == "hc3" for r in records))

    def test_hllmc2_answer_fields_excludes_human_and_thoughts(self):
        cols = [
            "index",
            "source",
            "question",
            "human_answers",
            "chatgpt_answers",
            "DeepSeek-R1-8B_answers",
            "DeepSeek-R1-8B_thoughts",
        ]
        self.assertEqual(
            _hllmc2_answer_fields(cols),
            ["chatgpt_answers", "DeepSeek-R1-8B_answers"],
        )

    def test_summarize_cue_corpus_includes_dataset_breakout(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            human_dir = root / "paper" / "human"
            llm_dir = root / "paper" / "llm"
            human_dir.mkdir(parents=True)
            llm_dir.mkdir(parents=True)

            (human_dir / "paper_a.json").write_text(
                json.dumps({"title": "Paper A", "abstract": "Human abstract."}),
                encoding="utf-8",
            )
            (llm_dir / "paper_a.json").write_text(
                json.dumps({"title": "Paper A", "descriptions": ["LLM abstract."], "llm_engine": "mock"}),
                encoding="utf-8",
            )

            cfg = DomainConfig(item_type="paper", human_dir=human_dir, llm_dir=llm_dir)
            records = iter_local_domain_records(cfg, seed=3)
            summary = summarize_cue_corpus(records)
            self.assertIn("by_dataset_source", summary)
            self.assertEqual(summary["by_dataset_source"]["local_paper"], {"human": 1, "llm": 1})

    def test_limit_records_by_item_type_keeps_groups_together(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            human_dir = root / "movie" / "human"
            llm_dir = root / "movie" / "llm"
            human_dir.mkdir(parents=True)
            llm_dir.mkdir(parents=True)

            for idx in range(3):
                title = f"Movie {idx}"
                human = {"title": title, "summary": f"Human summary {idx}."}
                llm = {"title": title, "descriptions": [f"LLM summary {idx}."], "llm_engine": "mock"}
                (human_dir / f"movie_{idx}.json").write_text(json.dumps(human), encoding="utf-8")
                (llm_dir / f"movie_{idx}.json").write_text(json.dumps(llm), encoding="utf-8")

            cfg = DomainConfig(item_type="movie", human_dir=human_dir, llm_dir=llm_dir)
            records = iter_local_domain_records(cfg, seed=3)
            kept = limit_records_by_item_type(records, max_groups_by_item_type={"movie": 1}, seed=3)
            self.assertEqual(len({r.group_id for r in kept}), 1)


if __name__ == "__main__":
    unittest.main()
