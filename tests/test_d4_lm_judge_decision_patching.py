import unittest

import numpy as np
import pandas as pd
import torch

from aisafety.mech.decision_patching import (
    DecoderOutputPatchHook,
    deterministic_fit_mask,
    fit_low_rank_basis,
    normalized_recovery,
    replace_decision_positions,
    replace_span_positions,
    suppress_subspace,
)
from aisafety.scripts.run_d4_lm_judge_decision_patching import (
    _fit_suppression_basis,
    _heldout_component_recovery,
    _neutral_row,
    _prompt_record,
    _reuse_fit_probe_for_eval,
)
from aisafety.scripts.read_d4_lm_judge_suppression_controls import _safe_ratio


class D4LMJudgeDecisionPatchingTests(unittest.TestCase):
    def test_fit_split_keeps_order_swaps_together(self) -> None:
        ids = ["cf1", "cf1", "cf2", "cf2", "cf3", "cf3", "cf4", "cf4"]
        mask = deterministic_fit_mask(ids, fit_frac=0.5, seed=1234)
        for start in range(0, len(mask), 2):
            self.assertEqual(bool(mask[start]), bool(mask[start + 1]))
        self.assertGreater(int(mask.sum()), 0)
        self.assertLess(int(mask.sum()), len(mask))

    def test_low_rank_suppression_removes_fitted_direction(self) -> None:
        basis = fit_low_rank_basis(
            np.asarray([[2.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32),
            rank=1,
        )
        damped = suppress_subspace(
            torch.tensor([[4.0, 2.0, -1.0]]),
            basis_rows=torch.as_tensor(basis),
            center=torch.zeros(3),
            alpha=1.0,
        )
        self.assertTrue(torch.allclose(damped, torch.tensor([[0.0, 2.0, -1.0]]), atol=1e-5))

    def test_patch_helpers_edit_expected_positions(self) -> None:
        hidden = torch.zeros((2, 4, 3))
        replacements = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        decision = replace_decision_positions(
            hidden,
            positions=torch.tensor([1, 3]),
            replacements=replacements,
        )
        self.assertTrue(torch.equal(decision[0, 1], replacements[0]))
        self.assertTrue(torch.equal(decision[1, 3], replacements[1]))
        span = replace_span_positions(
            hidden,
            span_positions=[[0, 2], [1]],
            replacements=replacements,
        )
        self.assertTrue(torch.equal(span[0, 0], replacements[0]))
        self.assertTrue(torch.equal(span[0, 2], replacements[0]))
        self.assertTrue(torch.equal(span[1, 1], replacements[1]))

    def test_decoder_tuple_hook_preserves_auxiliary_outputs(self) -> None:
        hook = DecoderOutputPatchHook(
            positions=torch.tensor([1]),
            replacements=torch.tensor([[3.0, 4.0]]),
        )
        hidden = torch.zeros((1, 3, 2))
        edited = hook(None, (), (hidden, "cache"))
        self.assertEqual(edited[1], "cache")
        self.assertTrue(torch.equal(edited[0][0, 1], torch.tensor([3.0, 4.0])))

    def test_neutralized_prompt_replaces_only_cue_plus_slot(self) -> None:
        row = {
            "bt_pair_id": "bt1",
            "counterfactual_id": "cf1",
            "cue_plus_option": "A",
            "cue_plus_text": "Overall, styled answer.",
            "cue_minus_text": "Plain answer.",
            "option_a_text": "Overall, styled answer.",
            "option_b_text": "Plain answer.",
            "prompt": "Explain the phrase Plain answer.",
        }
        neutral = _neutral_row(row)
        self.assertEqual(neutral["option_a_text"], "Plain answer.")
        self.assertEqual(neutral["option_b_text"], "Plain answer.")
        record = _prompt_record(
            "generated",
            row,
            object(),
            prompt_style="plain",
            comparison_template="standard",
        )
        self.assertIn("Overall, styled answer.", record.observed_prompt)
        self.assertEqual(record.neutral_prompt.count("Plain answer."), 3)
        self.assertEqual(
            record.neutral_prompt[record.neutral_cue_span[0] : record.neutral_cue_span[1]],
            "Plain answer.",
        )
        self.assertLess(record.neutral_cue_span[0], record.neutral_cue_span[1])

    def test_normalized_recovery(self) -> None:
        values = normalized_recovery(
            np.asarray([2.0, 3.0]),
            np.asarray([4.0, 4.0]),
            np.asarray([0.0, 2.0]),
        )
        self.assertTrue(np.allclose(values, np.asarray([0.5, 0.5])))

    def test_component_recovery_uses_heldout_subset_only(self) -> None:
        values = _heldout_component_recovery(
            patched=[2.0, 3.0],
            observed=[100.0, 4.0, 4.0, 100.0],
            neutral=[100.0, 0.0, 2.0, 100.0],
            verify_indices=[1, 2],
        )
        self.assertTrue(np.allclose(values, np.asarray([0.5, 0.5])))

    def test_fit_probe_reuses_its_sample_for_eval_reporting(self) -> None:
        self.assertTrue(_reuse_fit_probe_for_eval("generated", "generated"))
        self.assertFalse(_reuse_fit_probe_for_eval("atomic", "generated"))

    def test_suppression_basis_controls_are_matched_rank(self) -> None:
        observed = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32)
        neutral = np.zeros_like(observed)
        fitted = _fit_suppression_basis(observed, neutral, rank=2, basis_control="fitted", seed=1234)
        random = _fit_suppression_basis(observed, neutral, rank=2, basis_control="random", seed=1234)
        shuffled = _fit_suppression_basis(observed, neutral, rank=2, basis_control="shuffled_pair", seed=1234)
        self.assertEqual(fitted.shape, (2, 3))
        self.assertEqual(random.shape, (2, 3))
        self.assertEqual(shuffled.shape, (2, 3))

    def test_suppression_readout_leaves_undefined_attenuation_blank(self) -> None:
        values = _safe_ratio(
            pd.Series([1.0, 1.0]),
            pd.Series([2.0, 0.0]),
        )
        self.assertEqual(float(values.iloc[0]), 0.5)
        self.assertTrue(np.isnan(values.iloc[1]))


if __name__ == "__main__":
    unittest.main()
