from __future__ import annotations

import pytest

from giis_prod.creative_audit import _CASES, _deterministic_review


EVIDENCE = """[E1]
Document ID: doc_a
Title: A
Pages: 1
Section: Results
Text: Alpha precedes beta.

[E2]
Document ID: doc_a
Title: A
Pages: 2
Section: Results
Text: Beta precedes gamma."""


@pytest.mark.parametrize(
    ("style", "answer"),
    (
        ("table", "| Stage | Finding |\n|---|---|\n| First | Alpha [E1] |\n| Second | Beta [E2] |"),
        ("comparison", "The first result reports alpha [E1]. The second reports beta [E2]."),
        ("causal_chain", "1. Alpha precedes beta [E1].\n2. Beta precedes gamma [E2]."),
        (
            "concept_diagram",
            "```mermaid\nflowchart LR\nA[Alpha precedes beta [E1]]\nB[Beta precedes gamma [E2]]\nA --> B\n```",
        ),
    ),
)
def test_creative_formats_pass_existing_python_gates(style: str, answer: str) -> None:
    question = "What sequence is reported?"
    review = _deterministic_review(
        {
            "question": question,
            "answer": answer,
            "response_mode": "standard",
            "answer_style": style,
        },
        question=question,
        answer_style=style,
        evidence=EVIDENCE,
        minimum_coverage=0.40,
    )
    assert review["passed"] is True
    assert review["issues"] == []


def test_creative_audit_has_exactly_five_unique_cases() -> None:
    assert len(_CASES) == 5
    assert len({case["case_id"] for case in _CASES}) == 5
    assert {case["answer_style"] for case in _CASES} == {
        "table",
        "comparison",
        "causal_chain",
        "concept_diagram",
    }


def test_question_change_fails_closed() -> None:
    review = _deterministic_review(
        {
            "question": "A different question",
            "answer": "The first result reports alpha [E1]. The second reports beta [E2].",
            "response_mode": "standard",
            "answer_style": "comparison",
        },
        question="What sequence is reported?",
        answer_style="comparison",
        evidence=EVIDENCE,
        minimum_coverage=0.40,
    )
    assert review["passed"] is False
    assert "question_changed" in review["issues"]
