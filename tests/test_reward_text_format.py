import unittest

from aisafety.reward.text_format import format_prompt_response


class TestRewardTextFormat(unittest.TestCase):
    def test_empty_prompt_returns_response(self):
        self.assertEqual(format_prompt_response("", "resp"), "resp")
        self.assertEqual(format_prompt_response(None, "resp"), "resp")

    def test_prompt_and_response_joined(self):
        out = format_prompt_response("prompt", "resp")
        self.assertIn("prompt", out)
        self.assertIn("resp", out)
        self.assertIn("\n\n", out)

