from __future__ import annotations

import unittest

from giis_prod.ingestion import _enforce_ingestion_gate


class IngestionPolicyTests(unittest.TestCase):
    def test_replacements_allow_bounded_quarantine_without_underfilling(self):
        fraction = _enforce_ingestion_gate(50, 3, 50, 0.10)
        self.assertAlmostEqual(fraction, 3 / 53)

    def test_accepted_document_floor_cannot_be_satisfied_by_duplicates_or_quarantine(self):
        with self.assertRaisesRegex(RuntimeError, "49 accepted unique documents"):
            _enforce_ingestion_gate(49, 0, 50, 0.10)

    def test_quarantine_circuit_breaker_remains_bounded(self):
        with self.assertRaisesRegex(RuntimeError, "quarantine fraction"):
            _enforce_ingestion_gate(50, 6, 50, 0.10)


if __name__ == "__main__":
    unittest.main()
