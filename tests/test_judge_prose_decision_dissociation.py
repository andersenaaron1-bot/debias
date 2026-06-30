from __future__ import annotations

import numpy as np
import pandas as pd

from aisafety.scripts.analyze_judge_prose_decision_dissociation import (
    score_prose_row,
)
from aisafety.scripts.run_judge_prose_decision_direction_patching import (
    _direction_for_probe,
)
from aisafety.scripts.analyze_judge_factual_mediator_dissociation import (
    labels_for_target,
)


def test_score_prose_row_detects_criterion_target_and_binding() -> None:
    row = pd.Series(
        {
            "phase2_criterion_id": "coherence",
            "phase2_target_semantic": "A",
            "decoder_final_choice_semantic": "A",
            "phase1_response_text": "",
            "phase2_response_text": (
                "For coherence, Option A is better organized. "
                "Option B has less coherent flow. FINAL: A"
            ),
        }
    )

    scored = score_prose_row(row)

    assert scored["criterion_semantics"] is True
    assert scored["option_grounding"] is True
    assert scored["target_grounding"] is True
    assert scored["verdict_binding"] is True
    assert scored["prose_score"] == 1.0


def test_direction_for_binary_semantic_probe_flips_to_requested_class() -> None:
    arrays = {
        "criterion_target_coef": np.asarray([[3.0, 4.0]], dtype=np.float32),
        "criterion_target_classes": np.asarray(["A", "B"], dtype=str),
    }

    toward_b, label_b = _direction_for_probe(
        arrays,
        "criterion_target",
        desired_label="B",
        normalize=True,
    )
    toward_a, label_a = _direction_for_probe(
        arrays,
        "criterion_target",
        desired_label="A",
        normalize=True,
    )

    assert label_b == "B"
    assert label_a == "A"
    np.testing.assert_allclose(toward_b, -toward_a)
    np.testing.assert_allclose(np.linalg.norm(toward_b), 1.0)


def test_factual_mediator_labels_use_point_specific_values() -> None:
    frame = pd.DataFrame(
        [
            {
                "point_target_semantics": ["A", "B"],
                "point_forced_choices_semantic": ["A", "A"],
                "decoder_final_choice_semantic": "B",
            },
            {
                "point_target_semantics": ["B", "B"],
                "point_forced_choices_semantic": ["A", "B"],
                "decoder_final_choice_semantic": "B",
            },
        ]
    )

    assert labels_for_target(frame, "criterion_target", 1).tolist() == ["B", "B"]
    assert labels_for_target(frame, "current_choice", 1).tolist() == ["A", "B"]
    assert labels_for_target(frame, "final_choice", 1).tolist() == ["B", "B"]
    assert labels_for_target(frame, "target_reached", 1).tolist() == [0, 1]
