import unittest

import torch

from aisafety.reward.losses import (
    cue_bce_loss,
    group_robust_reduce,
    inv_loss,
    lambda_schedule,
    multi_head_bce_losses,
    multi_head_mse_losses,
    pointwise_mse_loss,
    pref_loss,
)


class TestRewardLosses(unittest.TestCase):
    def test_pref_loss_direction(self):
        chosen = torch.tensor([2.0, 1.0])
        rejected = torch.tensor([0.0, -1.0])
        loss_good = float(pref_loss(chosen, rejected).item())
        loss_bad = float(pref_loss(rejected, chosen).item())
        self.assertLess(loss_good, loss_bad)

    def test_inv_loss_zero_when_equal(self):
        a = torch.tensor([0.5, -1.2, 3.4])
        b = a.clone()
        self.assertAlmostEqual(float(inv_loss(a, b).item()), 0.0, places=6)

    def test_lambda_schedule_ramp(self):
        lam0 = lambda_schedule(0, total_steps=100, lambda_max=0.5, ramp_frac=0.1)
        lam_mid = lambda_schedule(5, total_steps=100, lambda_max=0.5, ramp_frac=0.1)
        lam_full = lambda_schedule(10, total_steps=100, lambda_max=0.5, ramp_frac=0.1)
        self.assertAlmostEqual(lam0, 0.0, places=6)
        self.assertGreater(lam_mid, 0.0)
        self.assertAlmostEqual(lam_full, 0.5, places=6)

    def test_cue_bce_loss_direction(self):
        logits_good = torch.tensor([[4.0, -4.0]])
        logits_bad = torch.tensor([[-4.0, 4.0]])
        targets = torch.tensor([[1.0, 0.0]])
        loss_good = float(cue_bce_loss(logits_good, targets).item())
        loss_bad = float(cue_bce_loss(logits_bad, targets).item())
        self.assertLess(loss_good, loss_bad)

    def test_pointwise_mse_loss_direction(self):
        preds_good = torch.tensor([0.9, 0.1])
        preds_bad = torch.tensor([0.1, 0.9])
        targets = torch.tensor([1.0, 0.0])
        self.assertLess(
            float(pointwise_mse_loss(preds_good, targets).item()),
            float(pointwise_mse_loss(preds_bad, targets).item()),
        )

    def test_multi_head_losses_shapes(self):
        pred = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        targ = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        mse = multi_head_mse_losses(pred, targ)
        bce = multi_head_bce_losses(pred, targ)
        self.assertEqual(tuple(mse.shape), (2,))
        self.assertEqual(tuple(bce.shape), (2,))

    def test_group_robust_reduce_interpolates_toward_max(self):
        vals = torch.tensor([1.0, 3.0])
        mean_only = float(group_robust_reduce(vals, strength=0.0).item())
        max_only = float(group_robust_reduce(vals, strength=1.0).item())
        self.assertAlmostEqual(mean_only, 2.0, places=6)
        self.assertAlmostEqual(max_only, 3.0, places=6)
