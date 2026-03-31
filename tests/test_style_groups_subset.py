import unittest

from aisafety.scripts.materialize_style_groups_subset import _limit_rows, _parse_rename_map


class TestStyleGroupsSubset(unittest.TestCase):
    def test_parse_rename_map(self):
        self.assertEqual(
            _parse_rename_map("paws_surface=paraphrase_surface,foo=bar"),
            {"paws_surface": "paraphrase_surface", "foo": "bar"},
        )

    def test_limit_rows_keeps_whole_groups(self):
        rows = [
            {"style_axis": "formality", "group_id": "g1", "variants": ["a", "b"]},
            {"style_axis": "formality", "group_id": "g2", "variants": ["c", "d"]},
            {"style_axis": "fluency", "group_id": "g3", "variants": ["e", "f"]},
            {"style_axis": "fluency", "group_id": "g4", "variants": ["g", "h"]},
        ]
        kept = _limit_rows(rows, max_groups_per_axis=1, seed=7)
        formality = {row["group_id"] for row in kept if row["style_axis"] == "formality"}
        fluency = {row["group_id"] for row in kept if row["style_axis"] == "fluency"}
        self.assertEqual(len(formality), 1)
        self.assertEqual(len(fluency), 1)


if __name__ == "__main__":
    unittest.main()
