import json
import tempfile
import unittest
from pathlib import Path

from aisafety.scripts.run_full_reward_eval import infer_experiment_config_path, resolve_run_context


class TestRunFullRewardEval(unittest.TestCase):
    def test_infers_matching_experiment_config_by_run_dir_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg_dir = root / "configs" / "experiments"
            cfg_dir.mkdir(parents=True)
            exp_path = cfg_dir / "m2_full_v1.json"
            exp_path.write_text("{}", encoding="utf-8")
            run_dir = root / "artifacts" / "reward" / "m2_full_v1"
            run_dir.mkdir(parents=True)

            got = infer_experiment_config_path(run_dir, root)
            self.assertEqual(got, exp_path)

    def test_resolve_run_context_prefers_experiment_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data" / "derived" / "pref").mkdir(parents=True)
            (root / "data" / "derived" / "style").mkdir(parents=True)
            pref_path = root / "data" / "derived" / "pref" / "val.jsonl"
            style_path = root / "data" / "derived" / "style" / "val.jsonl"
            pref_path.write_text("", encoding="utf-8")
            style_path.write_text("", encoding="utf-8")

            cfg_dir = root / "configs" / "experiments"
            cfg_dir.mkdir(parents=True)
            exp_path = cfg_dir / "m3_full_v1.json"
            exp_path.write_text(
                json.dumps(
                    {
                        "model_id": "google/gemma-2-9b-it",
                        "pref_val_jsonl": "data/derived/pref/val.jsonl",
                        "style_val_jsonl": "data/derived/style/val.jsonl",
                    }
                ),
                encoding="utf-8",
            )
            run_dir = root / "artifacts" / "reward" / "m3_full_v1"
            run_dir.mkdir(parents=True)

            ctx = resolve_run_context(
                run_dir=run_dir,
                workspace_root=root,
                experiment_config=None,
                model_id_override=None,
                pref_jsonl_override=None,
                style_jsonl_override=None,
            )

            self.assertEqual(ctx["model_id"], "google/gemma-2-9b-it")
            self.assertEqual(ctx["pref_jsonl"], pref_path.resolve())
            self.assertEqual(ctx["style_jsonl"], style_path.resolve())


if __name__ == "__main__":
    unittest.main()
