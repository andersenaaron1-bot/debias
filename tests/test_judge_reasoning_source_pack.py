import csv
import io
import tarfile
from pathlib import Path
import tempfile
import unittest

from aisafety.scripts.build_judge_reasoning_pairs import build_comparisons
from aisafety.scripts.build_judge_reasoning_source_pack import (
    build_arc_pairs,
    build_d4_pairs,
    build_ethics_pairs,
    build_helpsteer_pairs,
    build_truthfulqa_pairs,
    load_ethics_rows,
    load_truthfulqa_rows,
)


class JudgeReasoningSourcePackTests(unittest.TestCase):
    def test_arc_uses_correct_answer_and_closest_length_distractor(self) -> None:
        rows = [
            {
                "id": "arc-1",
                "question": "What freezes at zero degrees Celsius?",
                "choices": {
                    "label": ["A", "B", "C"],
                    "text": ["Water", "A very long distractor answer", "Iron"],
                },
                "answerKey": "A",
            }
        ]
        pairs = build_arc_pairs(
            rows,
            dataset_id="arc_easy",
            difficulty_tier="easy",
            split="validation",
            seed=1234,
        )
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["option_a_text"], "Water")
        self.assertEqual(pairs[0]["option_b_text"], "Iron")
        self.assertEqual(pairs[0]["target_option"], "A")
        self.assertEqual(pairs[0]["validity_type"], "objective")

    def test_truthfulqa_prefers_best_incorrect_answer(self) -> None:
        pairs = build_truthfulqa_pairs(
            [
                {
                    "Question": "Can humans breathe underwater unaided?",
                    "Best Answer": "No.",
                    "Best Incorrect Answer": "Yes.",
                    "Incorrect Answers": "Yes.;Only on Tuesdays.",
                    "Category": "Science",
                }
            ],
            seed=1234,
        )
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["option_a_text"], "No.")
        self.assertEqual(pairs[0]["option_b_text"], "Yes.")
        self.assertEqual(pairs[0]["comparison_dimension"], "truthfulness")

    def test_ethics_pairs_by_length_and_marks_consensus_target(self) -> None:
        pairs = build_ethics_pairs(
            [
                {"input": "Return the lost wallet.", "label": 1, "is_short": True},
                {"input": "Keep the lost wallet.", "label": 0, "is_short": True},
                {"input": "Help a neighbor.", "label": 1, "is_short": True},
                {"input": "Deliberately injure a neighbor.", "label": 0, "is_short": True},
            ],
            seed=1234,
        )
        self.assertEqual(len(pairs), 2)
        self.assertTrue(all(row["target_kind"] == "consensus" for row in pairs))
        self.assertTrue(all(row["target_option"] == "A" for row in pairs))

    def test_helpsteer_separates_dominated_and_tradeoff_pairs(self) -> None:
        rows = [
            {
                "prompt": "Explain the result.",
                "response": "Strong response.",
                "helpfulness": 4,
                "correctness": 4,
                "coherence": 4,
                "complexity": 3,
                "verbosity": 3,
            },
            {
                "prompt": "Explain the result.",
                "response": "Weak response.",
                "helpfulness": 2,
                "correctness": 2,
                "coherence": 2,
                "complexity": 2,
                "verbosity": 2,
            },
            {
                "prompt": "Choose an explanation.",
                "response": "Correct but terse.",
                "helpfulness": 4,
                "correctness": 4,
                "coherence": 2,
                "complexity": 1,
                "verbosity": 1,
            },
            {
                "prompt": "Choose an explanation.",
                "response": "Detailed but less correct.",
                "helpfulness": 2,
                "correctness": 2,
                "coherence": 4,
                "complexity": 4,
                "verbosity": 4,
            },
        ]
        pairs = build_helpsteer_pairs(rows, seed=1234)
        self.assertEqual(len(pairs["dominated"]), 1)
        self.assertEqual(len(pairs["tradeoff"]), 1)
        self.assertEqual(pairs["dominated"][0]["target_option"], "A")
        self.assertEqual(pairs["dominated"][0]["target_kind"], "consensus")
        self.assertEqual(
            pairs["dominated"][0]["source_dataset"],
            "helpsteer2_dominated",
        )
        self.assertEqual(pairs["tradeoff"][0]["target_option"], "")
        self.assertEqual(pairs["tradeoff"][0]["validity_type"], "plural")
        self.assertEqual(
            pairs["tradeoff"][0]["source_dataset"],
            "helpsteer2_tradeoff",
        )

    def test_d4_deduplicates_orders_without_assigning_human_as_gold(self) -> None:
        source = {
            "pair_id": "pair-1",
            "prompt": "How should this be explained?",
            "human_text": "Human response.",
            "llm_text": "Model response.",
            "source_dataset": "hape",
        }
        pairs = build_d4_pairs(
            [
                {**source, "presentation_order": "human_first"},
                {**source, "presentation_order": "llm_first"},
            ],
            seed=1234,
        )
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["target_option"], "")
        self.assertEqual(pairs[0]["source_dataset"], "d4_human_llm")
        self.assertEqual(pairs[0]["original_source_dataset"], "hape")
        comparisons = build_comparisons(
            pairs,
            input_format="generic",
            source_label="d4_human_llm",
            task_type="human_vs_llm_quality",
            comparison_dimension="overall_quality",
            include_order_swaps=True,
        )
        self.assertEqual(len(comparisons), 2)
        self.assertTrue(all(row["target_option"] == "" for row in comparisons))

    def test_local_truthfulqa_and_ethics_archive_loaders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            truthful = root / "TruthfulQA"
            truthful.mkdir()
            with (truthful / "TruthfulQA.csv").open(
                "w",
                encoding="utf-8",
                newline="",
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "Question",
                        "Best Answer",
                        "Best Incorrect Answer",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "Question": "Question?",
                        "Best Answer": "True.",
                        "Best Incorrect Answer": "False.",
                    }
                )

            ethics_tar = root / "ethics.tar"
            payload = io.BytesIO()
            text = "label,input\n1,Kind action.\n0,Harmful action.\n".encode("utf-8")
            with tarfile.open(fileobj=payload, mode="w") as archive:
                info = tarfile.TarInfo("ethics/commonsense/cm_test_hard.csv")
                info.size = len(text)
                archive.addfile(info, io.BytesIO(text))
            ethics_tar.write_bytes(payload.getvalue())

            truthful_rows, truthful_path = load_truthfulqa_rows(truthful)
            ethics_rows, ethics_source = load_ethics_rows(ethics_tar)
            self.assertEqual(len(truthful_rows), 1)
            self.assertEqual(truthful_path.name, "TruthfulQA.csv")
            self.assertEqual(len(ethics_rows), 2)
            self.assertIn("cm_test_hard.csv", ethics_source)


if __name__ == "__main__":
    unittest.main()
