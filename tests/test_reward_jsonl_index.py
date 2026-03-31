import json
import tempfile
import unittest
from pathlib import Path

from aisafety.reward.jsonl_index import build_offsets, build_offsets_by_key


class TestRewardJsonlIndex(unittest.TestCase):
    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def test_build_offsets_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.jsonl"
            rows = [{"a": 1}, {"a": 2}, {"a": 3}]
            self._write_jsonl(p, rows)
            idx = build_offsets(p)
            got = [idx.read_at(off) for off in idx.offsets]
            self.assertEqual(got, rows)

    def test_build_offsets_by_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.jsonl"
            rows = [
                {"style_axis": "a", "x": 1},
                {"style_axis": "b", "x": 2},
                {"style_axis": "a", "x": 3},
            ]
            self._write_jsonl(p, rows)
            by = build_offsets_by_key(p, key="style_axis")
            self.assertEqual(set(by.keys()), {"a", "b"})
            a_rows = [by["a"].read_at(off) for off in by["a"].offsets]
            self.assertEqual([r["x"] for r in a_rows], [1, 3])

