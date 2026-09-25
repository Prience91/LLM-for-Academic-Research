from __future__ import annotations

import unittest

from giis_prod.ingestion import _add_z_stroke_warning, normalize_text
from giis_prod.io import stable_digest


class TextAndIdentityTests(unittest.TestCase):
    def test_normalization_preserves_accents_and_removes_controls(self):
        self.assertEqual(normalize_text("  José\x00  Zürich  "), "José Zürich")

    def test_z_with_stroke_is_normalized(self):
        self.assertEqual(normalize_text("Ƶambia and ƶebra"), "Zambia and zebra")

    def test_z_with_stroke_audit_counts_are_aggregated(self):
        warnings = ["authors_missing"]
        _add_z_stroke_warning(warnings, 2)
        _add_z_stroke_warning(warnings, 3)
        self.assertEqual(
            warnings,
            ["authors_missing", "unicode_z_stroke_normalized=5"],
        )

    def test_stable_digest_is_deterministic_and_order_sensitive(self):
        self.assertEqual(stable_digest("a", "b"), stable_digest("a", "b"))
        self.assertNotEqual(stable_digest("a", "b"), stable_digest("b", "a"))


if __name__ == "__main__":
    unittest.main()
