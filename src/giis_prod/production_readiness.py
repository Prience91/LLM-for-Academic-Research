from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# These checks intentionally represent work deferred until after the five-paper v15-RC
# smoke run. A production-size dataset cannot be started merely by forgetting the
# CLI override; every item must be deliberately completed and enabled.
REQUIRED_PRODUCTION_GATES: tuple[tuple[str, str], ...] = (
    (
        "multi_source_checkpoint_resume_verified",
        "multi-source checkpoint/resume is implemented and recovery-tested",
    ),
    (
        "minimum_multi_source_yield_gate_verified",
        "a minimum multi-source yield gate is implemented and calibrated",
    ),
    (
        "answer_mode_distribution_gate_verified",
        "answer-mode and long-form distribution gates are implemented and calibrated",
    ),
    (
        "representative_canary_verified",
        "a representative 50-100 paper canary, including multi-source cases, has passed",
    ),
    (
        "benchmark_frozen_and_passed",
        "the production benchmark is frozen and its quality gates have passed",
    ),
    (
        "production_scale_preflight_verified",
        "the 2,500-paper capacity, runtime, storage, and restart preflight has passed",
    ),
    (
        "cumulative_quarantine_verified",
        "cumulative rejection/quarantine review is implemented and accepted",
    ),
)


class ProductionReadinessError(RuntimeError):
    """Raised before a production-size dataset build when readiness is incomplete."""


@dataclass(frozen=True)
class ProductionReadinessReport:
    effective_documents: int
    production_documents: int
    production_lock_applies: bool
    complete: bool
    incomplete_gates: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "effective_documents": self.effective_documents,
            "production_documents": self.production_documents,
            "production_lock_applies": self.production_lock_applies,
            "complete": self.complete,
            "incomplete_gates": list(self.incomplete_gates),
        }


def _positive_document_count(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer, not a boolean")
    try:
        documents = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if documents <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return documents


def production_readiness_report(
    settings: Any,
    expected_documents: int | None = None,
) -> ProductionReadinessReport:
    try:
        production_value = settings.get("production_readiness.production_document_threshold")
        production_label = "production_readiness.production_document_threshold"
    except KeyError:
        production_value = settings.get("teacher.expected_training_documents")
        production_label = "teacher.expected_training_documents"
    production_documents = _positive_document_count(production_value, production_label)
    effective_documents = _positive_document_count(
        production_documents if expected_documents is None else expected_documents,
        "expected_documents",
    )
    lock_applies = effective_documents >= production_documents

    incomplete: list[str] = []
    if lock_applies:
        for key, description in REQUIRED_PRODUCTION_GATES:
            try:
                complete = settings.get(f"production_readiness.{key}") is True
            except KeyError:
                complete = False
            if not complete:
                incomplete.append(f"{key}: {description}")

    return ProductionReadinessReport(
        effective_documents=effective_documents,
        production_documents=production_documents,
        production_lock_applies=lock_applies,
        complete=not incomplete,
        incomplete_gates=tuple(incomplete),
    )


def assert_dataset_generation_ready(
    settings: Any,
    expected_documents: int | None = None,
) -> ProductionReadinessReport:
    """Fail before loading the teacher builder when a production run is not ready."""

    report = production_readiness_report(settings, expected_documents)
    if report.production_lock_applies and not report.complete:
        details = "\n  - ".join(report.incomplete_gates)
        raise ProductionReadinessError(
            "Production dataset generation is locked before selection, teacher API calls, "
            "or SFT artifact mutation. Requested "
            f"{report.effective_documents:,} documents; the production threshold is "
            f"{report.production_documents:,}. Complete every deferred v15-RC gate in "
            "PRODUCTION_READINESS_V15_RC.md and set its matching "
            "[production_readiness] flag to true. Incomplete gates:\n  - "
            f"{details}"
        )
    return report
