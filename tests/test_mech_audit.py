import json
import tempfile
import unittest
from pathlib import Path

from aisafety.mech.audit import build_manifest_audit


class D4ManifestAuditTest(unittest.TestCase):
    def test_build_manifest_audit_flags_missing_current_atoms(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_path = root / "atom_probe_set.jsonl"
            output_path.write_text('{"text": "example"}\n', encoding="utf-8")

            ontology_path = root / "ontology.json"
            manifest_path = root / "manifest.json"
            ontology_path.write_text(
                json.dumps(
                    {
                        "priority_atoms": ["formal_connectives"],
                        "trace_bundles": [
                            {
                                "bundle_id": "formal_information_packaging",
                                "status": "primary",
                                "member_atoms": ["formal_connectives", "nominalization_patterns"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        "d4_atoms": ["formal_connectives"],
                        "trace_bundles": {
                            "formal_information_packaging": {
                                "status": "primary",
                                "member_atoms": ["formal_connectives"],
                            }
                        },
                        "outputs": {"atom_probe_set": str(output_path)},
                    }
                ),
                encoding="utf-8",
            )

            audit = build_manifest_audit(
                ontology_json=ontology_path,
                manifest_json=manifest_path,
                workspace_root=root,
            )

        self.assertEqual(audit["manifest_atom_count"], 1)
        self.assertEqual(audit["missing_current_ontology_atoms"], ["nominalization_patterns"])
        self.assertEqual(
            audit["bundle_support"]["formal_information_packaging"]["support_status"],
            "partial",
        )
        self.assertTrue(audit["rebuild_recommended"])


if __name__ == "__main__":
    unittest.main()

