import unittest

import numpy as np

from aisafety.scripts.analyze_judge_criterion_switch_decoders import (
    _fit,
    _metrics,
)
from aisafety.scripts.build_helpsteer2_criterion_switch_suite import (
    _pair_signature,
    _transition_candidates,
    build_episodes,
    build_switch_pairs,
)
from aisafety.scripts.build_judge_reasoning_source_pack import ATTRIBUTE_NAMES
from aisafety.scripts.run_judge_criterion_switch_activations import (
    point_labels,
    point_token_sequences,
)
from aisafety.scripts.run_judge_criterion_switch_behavior import (
    _semantic_verdict,
)


def _row(prompt: str, response: str, values: tuple[float, ...]) -> dict:
    return {
        "prompt": prompt,
        "response": response,
        **dict(zip(ATTRIBUTE_NAMES, values, strict=True)),
    }


def _source_rows() -> list[dict]:
    rows: list[dict] = []
    for index in range(5):
        rows.extend(
            [
                _row(f"choice-{index}", "helpful", (4, 2, 3, 2, 2)),
                _row(f"choice-{index}", "correct", (2, 4, 3, 2, 2)),
                _row(f"tie-{index}", "helpful", (4, 3, 3, 2, 2)),
                _row(f"tie-{index}", "tied", (2, 3, 3, 2, 2)),
                _row(f"same-{index}", "strong", (4, 4, 4, 2, 2)),
                _row(f"same-{index}", "weak", (2, 2, 2, 2, 2)),
            ]
        )
    return rows


class CriterionSwitchSuiteTests(unittest.TestCase):
    def test_transition_types(self) -> None:
        choice = _transition_candidates(
            {"helpfulness": "A", "correctness": "B", "coherence": "C"},
            {"helpfulness": 2.0, "correctness": -2.0, "coherence": 0.0},
            min_choice_gap=1.0,
        )
        self.assertTrue(choice["choice_to_choice"])
        self.assertTrue(choice["tie_to_choice"])
        same = _transition_candidates(
            {"helpfulness": "A", "correctness": "A", "coherence": "A"},
            {"helpfulness": 2.0, "correctness": 2.0, "coherence": 2.0},
            min_choice_gap=1.0,
        )
        self.assertTrue(same["same_target"])

    def test_content_signature_ignores_response_order(self) -> None:
        self.assertEqual(
            _pair_signature("prompt", "left", "right"),
            _pair_signature("prompt", "right", "left"),
        )
        excluded = _pair_signature("choice-0", "helpful", "correct")
        pairs = build_switch_pairs(
            _source_rows(),
            excluded_pair_signatures={excluded},
            max_pairs_per_transition=4,
            min_pairs_per_transition=4,
            min_choice_gap=1.0,
            seed=1234,
        )
        self.assertNotIn(excluded, {row["pair_signature"] for row in pairs})

    def test_pair_splits_and_episode_conditions(self) -> None:
        pairs = build_switch_pairs(
            _source_rows(),
            excluded_pair_signatures=set(),
            max_pairs_per_transition=5,
            min_pairs_per_transition=5,
            min_choice_gap=1.0,
            seed=1234,
        )
        self.assertEqual(len(pairs), 15)
        for transition in (
            "choice_to_choice",
            "tie_to_choice",
            "same_target",
        ):
            values = [
                row for row in pairs if row["transition_type"] == transition
            ]
            self.assertEqual(
                [row["analysis_split"] for row in values].count("fit"), 3
            )
            self.assertEqual(
                [row["analysis_split"] for row in values].count("selection"),
                1,
            )
            self.assertEqual(
                [row["analysis_split"] for row in values].count(
                    "intervention"
                ),
                1,
            )
        for pair in pairs:
            pair["source_split"] = "train"
        episodes = build_episodes(pairs[:1])
        self.assertEqual(len(episodes), 10)
        self.assertEqual({row["split"] for row in episodes}, {"train"})
        self.assertEqual(
            {row["condition_id"] for row in episodes},
            {"stable", "reminder", "switch", "placebo", "delayed"},
        )
        self.assertEqual(
            {row["presentation_order"] for row in episodes},
            {"original", "swapped"},
        )

    def test_semantic_verdict_swap(self) -> None:
        self.assertEqual(_semantic_verdict("A", "original"), "A")
        self.assertEqual(_semantic_verdict("A", "swapped"), "B")
        self.assertEqual(_semantic_verdict("C", "swapped"), "C")


class CriterionSwitchActivationTests(unittest.TestCase):
    def test_point_prefixes_and_labels(self) -> None:
        row = {
            "phase1_prompt_token_ids": [1, 2],
            "phase1_response_token_ids": list(range(200)),
            "phase2_prompt_token_ids": [3, 4, 5],
            "phase2_response_token_ids": list(range(500)),
            "phase1_criterion_id": "correctness",
            "phase2_criterion_id": "helpfulness",
            "phase1_target_semantic": "A",
            "phase2_target_semantic": "B",
        }
        sequences = point_token_sequences(row)
        self.assertEqual(len(sequences), 7)
        self.assertEqual(len(sequences[1]), 66)
        self.assertEqual(len(sequences[5]), 131)
        criteria, targets = point_labels(row)
        self.assertEqual(
            criteria,
            ["correctness"] * 3 + ["helpfulness"] * 4,
        )
        self.assertEqual(targets, ["A"] * 3 + ["B"] * 4)

    def test_multiclass_decoder_helpers(self) -> None:
        x = np.asarray(
            [
                [3.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
                [0.0, 0.0, 2.0],
            ],
            dtype=np.float32,
        )
        labels = np.asarray(["a", "a", "b", "b", "c", "c"])
        center, model = _fit(x, labels, c_value=10.0, seed=1234)
        metrics = _metrics(model, center, x, labels)
        self.assertGreaterEqual(metrics["balanced_accuracy"], 0.99)


if __name__ == "__main__":
    unittest.main()
