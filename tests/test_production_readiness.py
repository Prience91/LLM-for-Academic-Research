from __future__ import annotations

import unittest

from giis_prod.production_readiness import (
    REQUIRED_PRODUCTION_GATES,
    ProductionReadinessError,
    assert_dataset_generation_ready,
    production_readiness_report,
)


class FakeSettings:
    def __init__(self, *, target: int = 2500, complete: bool = False):
        self.values = {
            "teacher.expected_training_documents": 630,
            "production_readiness.production_document_threshold": target,
        }
        for key, _ in REQUIRED_PRODUCTION_GATES:
            self.values[f"production_readiness.{key}"] = complete

    def get(self, dotted_key: str):
        return self.values[dotted_key]


class ProductionReadinessTests(unittest.TestCase):
    def test_five_document_smoke_run_is_allowed(self):
        report = assert_dataset_generation_ready(FakeSettings(), 5)
        self.assertFalse(report.production_lock_applies)
        self.assertTrue(report.complete)

    def test_default_production_target_is_locked(self):
        with self.assertRaisesRegex(ProductionReadinessError, "2,500"):
            assert_dataset_generation_ready(FakeSettings())

    def test_explicit_production_target_is_locked(self):
        with self.assertRaisesRegex(ProductionReadinessError, "before selection"):
            assert_dataset_generation_ready(FakeSettings(), 2500)

    def test_all_deliberately_completed_gates_unlock_production(self):
        report = assert_dataset_generation_ready(FakeSettings(complete=True), 2500)
        self.assertTrue(report.production_lock_applies)
        self.assertTrue(report.complete)
        self.assertEqual(report.incomplete_gates, ())

    def test_missing_gate_is_fail_closed(self):
        settings = FakeSettings(complete=True)
        key, _ = REQUIRED_PRODUCTION_GATES[0]
        del settings.values[f"production_readiness.{key}"]
        report = production_readiness_report(settings, 2500)
        self.assertFalse(report.complete)
        self.assertIn(key, report.incomplete_gates[0])

    def test_non_positive_document_override_is_rejected(self):
        for value in (0, -1, True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    assert_dataset_generation_ready(FakeSettings(), value)

if __name__ == "__main__":
    unittest.main()
