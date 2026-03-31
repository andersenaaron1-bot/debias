import unittest

import torch

from aisafety.features.token_positions import last_non_pad_index, take_last_token


class TestTokenPositions(unittest.TestCase):
    def test_last_non_pad_index_right_padding(self):
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.long)
        idx = last_non_pad_index(mask, padding_side="right")
        self.assertTrue(torch.equal(idx, torch.tensor([2, 4], dtype=torch.long)))

    def test_last_non_pad_index_left_padding(self):
        mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=torch.long)
        idx = last_non_pad_index(mask, padding_side="left")
        self.assertTrue(torch.equal(idx, torch.tensor([4, 4], dtype=torch.long)))

    def test_take_last_token_right_padding(self):
        h = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.long)
        last = take_last_token(h, mask, padding_side="right")
        self.assertTrue(torch.equal(last[0], h[0, 2]))
        self.assertTrue(torch.equal(last[1], h[1, 4]))

    def test_take_last_token_left_padding(self):
        h = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
        mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=torch.long)
        last = take_last_token(h, mask, padding_side="left")
        self.assertTrue(torch.equal(last[0], h[0, 4]))
        self.assertTrue(torch.equal(last[1], h[1, 4]))

