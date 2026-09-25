"""Self-contained V17-Turbo evidence, obligation, and answer-unit contract."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from jsonschema import Draft202012Validator

_MARKER = re.compile(r"\[E[1-9]\d*\]")
_SENTENCE_END = re.compile(r"[.!?](?:[\"'’”)]*)")
_START = re.compile(r"^[\"'“‘(\[]*[A-Z0-9]")
_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


@dataclass(frozen=True, slots=True)
class SupportUnit:
    support_unit_id: str
    evidence_id: str
    text: str


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _array(items: dict[str, Any], *, minimum: int = 1) -> dict[str, Any]:
    return {"type": "array", "items": items, "minItems": minimum}


def build_support_units(evidence: list[dict[str, Any]]) -> list[SupportUnit]:
    """Return stable complete-sentence units; unmatched fragments are ineligible."""
    units: list[SupportUnit] = []
    for evidence_index, chunk in enumerate(evidence, start=1):
        evidence_id = f"[E{evidence_index}]"
        text = re.sub(r"\s+", " ", str(chunk.get("text") or "")).strip()
        cursor = 0
        sentence_index = 0
        for match in _SENTENCE_END.finditer(text):
            segment = text[cursor:match.end()].strip()
            cursor = match.end()
            if segment and _START.search(segment):
                sentence_index += 1
                units.append(SupportUnit(
                    support_unit_id=f"{evidence_id}:S{sentence_index:03d}",
                    evidence_id=evidence_id,
                    text=segment,
                ))
    return units


def render_support_context(
    evidence: list[dict[str, Any]], units: list[SupportUnit] | None = None
) -> str:
    catalog = units if units is not None else build_support_units(evidence)
    grouped: dict[str, list[SupportUnit]] = {}
    for unit in catalog:
        grouped.setdefault(unit.evidence_id, []).append(unit)
    blocks: list[str] = []
    for index, chunk in enumerate(evidence, start=1):
        evidence_id = f"[E{index}]"
        rows = grouped.get(evidence_id, [])
        if not rows:
            continue
        metadata = (
            f"{evidence_id}\nDocument ID: {chunk['document_id']}\n"
            f"Title: {chunk['title']}\nPages: {chunk['page_start']}-{chunk['page_end']}\n"
            f"Section: {chunk.get('section') or 'Unspecified'}"
        )
        body = "\n".join(f"{row.support_unit_id} {row.text}" for row in rows)
        blocks.append(metadata + "\n" + body)
    return "\n\n".join(blocks)


def candidate_schema(evidence_ids: list[str], support_unit_ids: list[str]) -> dict[str, Any]:
    text = {"type": "string", "minLength": 1}
    identifier = {"type": "string", "pattern": _ID.pattern}
    evidence_id = {"type": "string", "enum": evidence_ids}
    support_id = {"type": "string", "enum": support_unit_ids}
    return _object({
        "question": text,
        "coverage_obligations": _array(_object({
            "obligation_id": identifier,
            "requirement": text,
            "supports": _array(_object({
                "evidence_id": evidence_id,
                "support_unit_ids": _array(support_id),
            })),
        })),
        "answer_units": _array(_object({
            "answer_unit_id": identifier,
            "obligation_ids": _array(identifier),
            "text": text,
        })),
        "response_mode": {"type": "string", "enum": ["standard", "long_form"]},
        "answer_style": {"type": "string", "enum": ["prose", "table", "comparison"]},
    })


def materialize_candidate(raw: dict[str, Any]) -> dict[str, Any]:
    """Assemble the answer once from the compact answer-unit ledger."""
    answer = "\n\n".join(str(unit["text"]).strip() for unit in raw["answer_units"]).strip()
    return {
        "question": str(raw["question"]).strip(),
        "answer": answer,
        "response_mode": raw["response_mode"],
        "answer_style": raw["answer_style"],
        "coverage_obligations": raw["coverage_obligations"],
        "answer_units": raw["answer_units"],
    }


def schema_is_valid(raw: dict[str, Any], evidence_ids: list[str], support_unit_ids: list[str]) -> bool:
    schema = candidate_schema(evidence_ids, support_unit_ids)
    schema_input = {key: raw[key] for key in schema["properties"] if key in raw}
    return not list(Draft202012Validator(schema).iter_errors(schema_input))


def validate_structure(candidate: dict[str, Any], units: list[SupportUnit], *, lane: str = "core") -> dict[str, Any]:
    """Validate mappings and source ownership without making a semantic judgment."""
    reasons: list[str] = []
    def fail(reason: str) -> None:
        if reason not in reasons:
            reasons.append(reason)

    allowed_styles = {"prose"} if lane == "core" else {"table", "comparison"}
    if lane not in {"core", "qualification"}:
        fail("invalid_v17_turbo_lane")
    elif candidate["answer_style"] not in allowed_styles:
        fail("style_not_enabled_for_lane")

    unit_by_id = {unit.support_unit_id: unit for unit in units}
    obligations = candidate["coverage_obligations"]
    obligation_ids = [row["obligation_id"] for row in obligations]
    if len(obligation_ids) != len(set(obligation_ids)):
        fail("duplicate_obligation_id")

    evidence_by_obligation: dict[str, set[str]] = {}
    for obligation in obligations:
        supported: set[str] = set()
        for support in obligation["supports"]:
            for support_id in support["support_unit_ids"]:
                unit = unit_by_id.get(support_id)
                if unit is None:
                    fail("unknown_support_unit")
                elif unit.evidence_id != support["evidence_id"]:
                    fail("support_unit_evidence_mismatch")
                else:
                    supported.add(unit.evidence_id)
        evidence_by_obligation[obligation["obligation_id"]] = supported

    answer_units = candidate["answer_units"]
    answer_ids = [row["answer_unit_id"] for row in answer_units]
    if len(answer_ids) != len(set(answer_ids)):
        fail("duplicate_answer_unit_id")
    mapped = {obligation_id: 0 for obligation_id in obligation_ids}
    known = set(obligation_ids)
    for answer_unit in answer_units:
        referenced = answer_unit["obligation_ids"]
        if len(referenced) != len(set(referenced)):
            fail("duplicate_answer_unit_obligation")
        markers = set(_MARKER.findall(answer_unit["text"]))
        for obligation_id in referenced:
            if obligation_id not in known:
                fail("unknown_answer_unit_obligation")
                continue
            mapped[obligation_id] += 1
            if not (markers & evidence_by_obligation.get(obligation_id, set())):
                fail("answer_unit_missing_supporting_evidence")
    if any(count == 0 for count in mapped.values()):
        fail("unmapped_obligation")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "declared_obligations": len(obligation_ids),
        "answer_units": len(answer_units),
        "semantic_verdict": None,
    }


__all__ = [
    "SupportUnit", "build_support_units", "candidate_schema", "materialize_candidate",
    "render_support_context", "schema_is_valid", "validate_structure",
]
