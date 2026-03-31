import unittest
from pathlib import Path

from aisafety.scripts.run_experiment_config import build_train_command


class TestRunExperimentConfig(unittest.TestCase):
    def test_build_train_command_includes_expected_flags(self):
        config = {
            "model_id": "google/gemma-2-9b-it",
            "pref_train_jsonl": "data/train.jsonl",
            "pref_val_jsonl": "data/val.jsonl",
            "output_dir": "artifacts/run",
            "train_args": {
                "use_4bit": True,
                "bf16": True,
                "max_steps": 100,
                "cue_families": ["a", "b"],
            },
        }
        cmd = build_train_command(config, workspace_root=Path("/workspace"), output_dir_override=None)
        joined = " ".join(cmd)
        self.assertIn("aisafety.scripts.train_reward_lora", joined)
        self.assertIn("--use-4bit", cmd)
        self.assertIn("--bf16", cmd)
        self.assertIn("--max-steps", cmd)
        self.assertIn("a,b", joined)
        self.assertTrue(any(str(part).endswith("workspace\\data\\train.jsonl") for part in cmd))


if __name__ == "__main__":
    unittest.main()
