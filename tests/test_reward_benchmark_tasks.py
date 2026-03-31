import unittest

from aisafety.eval.benchmark_tasks import (
    compute_mcq_metrics,
    make_mcq_record,
    parse_run_spec,
)
from aisafety.eval.benchmark_tasks import (
    _normalize_arc,
    _normalize_boolq,
    _normalize_hellaswag,
    _normalize_mmlu,
    _normalize_piqa,
    _normalize_social_iqa,
    _normalize_winogrande,
)


class TestRewardBenchmarkTasks(unittest.TestCase):
    def test_parse_run_spec_with_adapter(self):
        run = parse_run_spec("pref_only=artifacts/reward/pref/lora::artifacts/reward/pref/value_head.pt")
        self.assertEqual(run.name, "pref_only")
        self.assertEqual(run.adapter_dir.as_posix(), "artifacts/reward/pref/lora")
        self.assertEqual(run.value_head.as_posix(), "artifacts/reward/pref/value_head.pt")

    def test_parse_run_spec_base_run(self):
        run = parse_run_spec("base=::artifacts/reward/base/value_head.pt")
        self.assertEqual(run.name, "base")
        self.assertIsNone(run.adapter_dir)
        self.assertEqual(run.value_head.as_posix(), "artifacts/reward/base/value_head.pt")

    def test_normalize_arc(self):
        row = {
            "id": "arc-1",
            "question": "What planet do humans live on?",
            "choices": {"label": ["A", "B", "C"], "text": ["Mars", "Earth", "Venus"]},
            "answerKey": "B",
            "subject": "science",
        }
        ex = _normalize_arc(row, 0, benchmark="arc_challenge")
        self.assertEqual(ex.example_id, "arc-1")
        self.assertEqual(ex.correct_idx, 1)
        self.assertIn("Options:", ex.prompt)
        self.assertEqual(ex.responses[1], "Earth")
        self.assertEqual(ex.group, "science")

    def test_normalize_boolq(self):
        row = {
            "question": "is the sky blue",
            "passage": "The sky can appear blue on a clear day.",
            "answer": True,
        }
        ex = _normalize_boolq(row, 0)
        self.assertEqual(ex.correct_idx, 0)
        self.assertEqual(ex.responses, ("yes", "no"))

    def test_normalize_piqa(self):
        row = {"goal": "Open a bottle", "sol1": "Use a bottle opener", "sol2": "Kick it", "label": 0}
        ex = _normalize_piqa(row, 0)
        self.assertEqual(ex.correct_idx, 0)
        self.assertEqual(ex.choice_labels, ("A", "B"))

    def test_normalize_winogrande(self):
        row = {
            "sentence": "The trophy doesn't fit in the suitcase because _ is too big.",
            "option1": "the trophy",
            "option2": "the suitcase",
            "answer": "1",
        }
        ex = _normalize_winogrande(row, 0)
        self.assertEqual(ex.correct_idx, 0)
        self.assertIn("the trophy", ex.responses[0])
        self.assertIn("the suitcase", ex.prompt)

    def test_normalize_hellaswag(self):
        row = {
            "ctx_a": "A person is slicing bread.",
            "ctx_b": "They continue preparing lunch.",
            "endings": ["They put the bread away.", "They make a sandwich."],
            "label": "1",
            "activity_label": "cooking",
        }
        ex = _normalize_hellaswag(row, 0)
        self.assertEqual(ex.correct_idx, 1)
        self.assertEqual(ex.group, "cooking")

    def test_normalize_social_iqa(self):
        row = {
            "context": "Jordan helped Casey carry groceries.",
            "question": "Why did Jordan do this?",
            "answerA": "To be kind",
            "answerB": "To make a mess",
            "answerC": "To leave early",
            "label": "1",
        }
        ex = _normalize_social_iqa(row, 0)
        self.assertEqual(ex.correct_idx, 0)
        self.assertEqual(len(ex.responses), 3)

    def test_normalize_mmlu_with_letter_answer(self):
        row = {
            "question": "2 + 2 = ?",
            "choices": ["1", "2", "3", "4"],
            "answer": "D",
            "subject": "math",
        }
        ex = _normalize_mmlu(row, 0)
        self.assertEqual(ex.correct_idx, 3)
        self.assertEqual(ex.group, "math")

    def test_make_mcq_record_and_metrics(self):
        row = {
            "question": "Capital of France?",
            "choices": {"label": ["A", "B"], "text": ["Paris", "Berlin"]},
            "answerKey": "A",
        }
        ex = _normalize_arc(row, 0, benchmark="arc_challenge")
        rec_good = make_mcq_record(ex, scores=[2.0, 1.0], run_name="test")
        rec_bad = make_mcq_record(ex, scores=[0.5, 1.5], run_name="test")
        self.assertTrue(rec_good["is_correct"])
        self.assertFalse(rec_bad["is_correct"])
        metrics = compute_mcq_metrics([rec_good, rec_bad])
        self.assertEqual(metrics["n_examples"], 2)
        self.assertAlmostEqual(metrics["accuracy"], 0.5, places=6)
        self.assertAlmostEqual(metrics["mean_gold_rank"], 1.5, places=6)


if __name__ == "__main__":
    unittest.main()
