import unittest

from aisafety.scripts.build_style_groups_hf import (
    _assign_split,
    _compute_group_id,
    _dedup_variants,
    _paradetox_has_explicit_threat,
    _passes_number_filter,
)


class TestStyleGroupsHelpers(unittest.TestCase):
    def test_dedup_variants_strips_and_dedups(self):
        out = _dedup_variants(["  A  ", "A", "B", "", "  "])
        self.assertEqual(out, ["A", "B"])

    def test_number_filter_requires_exact_digit_multiset(self):
        anchor = "I have 2 dogs."
        self.assertTrue(_passes_number_filter(anchor, ["I have 2 dogs!", "Dogs: 2."]))
        self.assertFalse(_passes_number_filter(anchor, ["I have two dogs."]))

    def test_group_id_order_invariant(self):
        gid1 = _compute_group_id("axis", "ds", ["x", "y"])
        gid2 = _compute_group_id("axis", "ds", ["y", "x"])
        self.assertEqual(gid1, gid2)

    def test_assign_split_deterministic(self):
        a = _assign_split("gid", seed=123, train_frac=0.9, val_frac=0.05)
        b = _assign_split("gid", seed=123, train_frac=0.9, val_frac=0.05)
        self.assertEqual(a, b)

    def test_paradetox_threat_regex(self):
        self.assertTrue(_paradetox_has_explicit_threat("I will kill you."))
        self.assertFalse(_paradetox_has_explicit_threat("This is rude, but not a threat."))
