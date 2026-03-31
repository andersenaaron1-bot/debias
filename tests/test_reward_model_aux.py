import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from aisafety.reward.model import RewardScorer


class DummyBackbone(nn.Module):
    def __init__(self, hidden_size: int = 4):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, return_dict: bool = True):
        hidden_size = int(self.config.hidden_size)
        last_hidden = torch.nn.functional.one_hot(input_ids % hidden_size, num_classes=hidden_size).float()
        return SimpleNamespace(last_hidden_state=last_hidden)


class TestRewardModelAux(unittest.TestCase):
    def test_reward_scorer_with_cue_heads(self):
        backbone = DummyBackbone(hidden_size=4)
        value_head = nn.Linear(4, 1)
        cue_heads = nn.ModuleDict({"academic_formality": nn.Linear(4, 1), "template_boilerplate": nn.Linear(4, 1)})
        model = RewardScorer(backbone=backbone, value_head=value_head, cue_heads=cue_heads)

        input_ids = torch.tensor([[1, 2, 0], [3, 1, 0]])
        attention_mask = torch.tensor([[1, 1, 0], [1, 1, 0]])
        pooled = model.encode(input_ids, attention_mask)
        scores = model.score_from_pooled(pooled)
        logits = model.cue_logits_from_pooled(pooled, head_names=["academic_formality"], grl_scale=1.0)

        self.assertEqual(tuple(scores.shape), (2,))
        self.assertEqual(set(logits), {"academic_formality"})
        self.assertEqual(tuple(logits["academic_formality"].shape), (2,))

    def test_reward_scorer_with_attribute_heads(self):
        backbone = DummyBackbone(hidden_size=4)
        value_head = nn.Linear(4, 1)
        attribute_heads = nn.ModuleDict({"helpfulness": nn.Linear(4, 1), "correctness": nn.Linear(4, 1)})
        model = RewardScorer(backbone=backbone, value_head=value_head, attribute_heads=attribute_heads)

        input_ids = torch.tensor([[1, 2, 0], [3, 1, 0]])
        attention_mask = torch.tensor([[1, 1, 0], [1, 1, 0]])
        pooled = model.encode(input_ids, attention_mask)
        outputs = model.attribute_logits_from_pooled(pooled, head_names=["helpfulness", "correctness"])

        self.assertEqual(set(outputs), {"helpfulness", "correctness"})
        self.assertEqual(tuple(outputs["helpfulness"].shape), (2,))

    def test_save_and_load_cue_heads(self):
        backbone = DummyBackbone(hidden_size=4)
        value_head = nn.Linear(4, 1)
        cue_heads = nn.ModuleDict({"academic_formality": nn.Linear(4, 1)})
        model = RewardScorer(backbone=backbone, value_head=value_head, cue_heads=cue_heads)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "cue_heads.pt"
            model.save_cue_heads(path)
            loaded = RewardScorer.load_cue_heads(path)
            self.assertIn("academic_formality", loaded)
            self.assertEqual(tuple(loaded["academic_formality"].weight.shape), (1, 4))

    def test_save_and_load_attribute_heads(self):
        backbone = DummyBackbone(hidden_size=4)
        value_head = nn.Linear(4, 1)
        attribute_heads = nn.ModuleDict({"helpfulness": nn.Linear(4, 1)})
        model = RewardScorer(backbone=backbone, value_head=value_head, attribute_heads=attribute_heads)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "attribute_heads.pt"
            model.save_attribute_heads(path)
            loaded = RewardScorer.load_attribute_heads(path)
            self.assertIn("helpfulness", loaded)
            self.assertEqual(tuple(loaded["helpfulness"].weight.shape), (1, 4))
