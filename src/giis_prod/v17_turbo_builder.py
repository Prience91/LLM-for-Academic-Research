"""Standalone V17-Turbo generator with token-capacity lanes and compact coverage."""
from __future__ import annotations

from typing import Any
import json
import os
import re
from threading import Lock

from jsonschema import Draft202012Validator

from .candidate_validation import validate_answer_style
from .citations import citation_coverage, contains_non_corpus_citation
from .io import content_version
from .teacher import EvidenceBundle, TeacherDatasetBuilder
from .v17_turbo_retry import author_request, TerminalGenerationFailure
from .v17_turbo_layer import (
    build_support_units, candidate_schema, materialize_candidate,
    render_support_context, validate_structure,
)

_EVIDENCE_MARKER = re.compile(r"\[E\d+\]")
_EVIDENCE_LIKE_BRACKET = re.compile(r"\[[^\]\n]*E\d+[^\]\n]*\]")


def batch_response_format(bundle: EvidenceBundle, requested_examples: int) -> dict[str, Any]:
    units = build_support_units(bundle.chunks)
    if not units:
        raise ValueError("Evidence bundle has no boundary-complete V17-Turbo support units")
    schema = candidate_schema(
        [f"[E{i}]" for i in range(1, len(bundle.chunks) + 1)],
        [unit.support_unit_id for unit in units],
    )
    schema["properties"]["answer_style"]["enum"] = ["prose"]
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "giis_v17_turbo_candidates",
            "strict": True,
            "schema": {
                "type": "object", "additionalProperties": False,
                "required": ["examples"],
                "properties": {"examples": {
                    "type": "array", "minItems": requested_examples,
                    "maxItems": requested_examples, "items": schema,
                }},
            },
        },
    }


class V17TurboDatasetBuilder(TeacherDatasetBuilder):
    """Own the complete V17-Turbo generation and deterministic-validation contract."""

    def __init__(self, settings: Any):
        super().__init__(settings)
        self.client = self.client.with_options(max_retries=0)
        self._terminal_generation_bundles = set()

    def _audit(self, event: str, **fields: Any) -> None:
        super()._audit(event, **fields)
        if event == "generation_response_received" and fields.get("finish_reason") == "length":
            raise ValueError("Teacher completion was truncated; candidate is not fully generated")

    def _request_examples_with_retries(self, bundle, accepted_questions=None, *, allow_long_form=True,
                                      requested_examples=2, partition=None, request_number=None):
        directory=self.settings.path("paths.sft_dir") / "generation_fail_closed"
        identity={"partition":partition,"bundle_id":bundle.bundle_id,"request_number":request_number}
        try:
            return author_request(directory,identity,lambda:self._request_examples(
                bundle,accepted_questions=accepted_questions,allow_long_form=allow_long_form,
                requested_examples=requested_examples,partition=partition,request_number=request_number))
        except TerminalGenerationFailure as exc:
            # The exhausted request is already hash-bound in the terminal quarantine.
            # Preserve all fully generated candidates in other bundles and proceed.
            self._terminal_generation_bundles.add(bundle.bundle_id)
            self._audit("terminal_authoring_request_quarantined",partition=partition,
                        request_number=request_number,bundle_id=bundle.bundle_id,
                        document_ids=list(bundle.document_ids),reason=str(exc))
            return []

    def enable_raw_candidate_journal(self, path: Any) -> None:
        self._v17_turbo_raw_candidate_journal = path
        self._v17_turbo_raw_candidate_lock = Lock()

    def _request_examples(
        self, bundle: EvidenceBundle, accepted_questions: list[str] | None = None,
        *, allow_long_form: bool = True, requested_examples: int = 2,
        partition: str | None = None, request_number: int | None = None,
    ) -> list[dict[str, Any]]:
        examples = super()._request_examples(
            bundle, accepted_questions=accepted_questions, allow_long_form=allow_long_form,
            requested_examples=requested_examples, partition=partition,
            request_number=request_number,
        )
        path = getattr(self, "_v17_turbo_raw_candidate_journal", None)
        if path is not None:
            rows = [{
                "bundle_id": bundle.bundle_id,
                "document_ids": list(bundle.document_ids),
                "partition": partition,
                "request_number": request_number,
                "candidate": example,
            } for example in examples]
            with self._v17_turbo_raw_candidate_lock:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
        return examples

    def _answer_token_count(self, answer: str) -> int:
        return len(self.tokenizer.encode(answer, add_special_tokens=False))

    def _normalize_capacity_mode(self, example: dict[str, Any]) -> tuple[bool, str]:
        answer_tokens = self._answer_token_count(str(example.get("answer") or ""))
        standard_max = int(self.settings.get("response.standard_max_tokens"))
        long_max = int(self.settings.get("response.long_form_max_tokens"))
        if not 0 < standard_max < long_max:
            raise ValueError("V17-Turbo answer-capacity boundaries are invalid")
        example["_answer_token_count"] = answer_tokens
        if answer_tokens <= standard_max:
            example["response_mode"] = "standard"
            return True, "ok"
        if answer_tokens <= long_max:
            example["response_mode"] = "long_form"
            return True, "ok"
        return False, "answer_exceeds_long_form_capacity"

    def _generation_request_options(
        self, bundle: EvidenceBundle, accepted_questions: list[str] | None = None,
        *, allow_long_form: bool = True, requested_examples: int = 2,
    ) -> dict[str, Any]:
        units = build_support_units(bundle.chunks)
        if not units:
            raise ValueError("Evidence bundle has no boundary-complete V17-Turbo support units")
        accepted_questions = accepted_questions or []
        prior = ""
        if accepted_questions:
            prior = "\nAlready accepted questions; create materially different questions:\n" + "\n".join(
                f"- {question}" for question in accepted_questions
            )
        standard_max = int(self.settings.get("response.standard_max_tokens"))
        long_max = int(self.settings.get("response.long_form_max_tokens"))
        capacity = (
            f"Use the answer capacity justified by the obligations. Answers through {standard_max} "
            f"tokens occupy the standard lane; answers from {standard_max + 1} through {long_max} "
            "tokens occupy the long_form lane. The validator assigns the final lane from actual "
            "token count. There is no minimum or target word count. Stop as soon as every supported "
            "obligation is completely answered; never pad or compress merely to fit a label."
            if allow_long_form else
            f"Keep each answer within the standard capacity of {standard_max} tokens because this "
            "paper's long_form allocation is full."
        )
        multi = len(bundle.document_ids) > 1
        source_rule = (
            "Each candidate must assign at least one material obligation to each source paper, cite "
            "both papers substantively, and perform genuine comparison or synthesis. Preserve "
            "differences in scale, method, population, timeframe, and outcome."
            if multi else "Keep each candidate fully answerable from this paper alone."
        )
        prompt = f"""Create exactly {requested_examples} high-quality synthetic training candidates for a corpus-grounded scientific research assistant.

V17-Turbo contract:
- Use only the supplied numbered support units. Never add outside or foundational knowledge.
- Propose a coherent evidence-backed question rather than an isolated fact when the evidence supports breadth.
- Before writing the answer, identify every material requirement created by that question.
- Prefer three to six substantive obligations when justified; the count is diagnostic and never a reason to invent or split obligations.
- Record a compact ledger containing obligation_id, concise requirement, evidence_id, and support_unit_ids.
- Compose the answer exactly once as ordered answer_units and map every obligation to at least one answer unit.
- Every factual sentence must contain one or more exact [E#] markers, and every answer unit must cite evidence that supports every obligation mapped to it.
- Use markers separately, such as [E1][E3]. Never invent markers or reproduce nested bibliographic citations as corpus evidence.
- {source_rule}
- Never turn association, juxtaposition, or a framework into an unsupported causal claim.
- {capacity}
- Set answer_style=prose. Tables and comparisons use a separately qualified lane.
- If the evidence cannot completely support a proposed question, choose a narrower question.
- Do not expose chain-of-thought. The compact ledger is an auditable mapping, not hidden reasoning.
{prior}

EVIDENCE:
{render_support_context(bundle.chunks, units)}
"""
        request_options: dict[str, Any] = {}
        if "enable_thinking" in self.teacher:
            request_options["extra_body"] = {"chat_template_kwargs": {
                "enable_thinking": bool(self.teacher["enable_thinking"])
            }}
        return {
            "model": self.settings.get("models.teacher"),
            "messages": [
                {"role": "system", "content": "Generate strictly grounded synthetic training data in the requested schema."},
                {"role": "user", "content": prompt},
            ],
            "response_format": batch_response_format(bundle, requested_examples),
            "temperature": float(self.teacher["temperature"]),
            "max_tokens": int(self.teacher["max_output_tokens"]),
            **request_options,
        }

    def _validate_deterministic(self, bundle: EvidenceBundle, example: dict[str, Any]) -> tuple[bool, str]:
        structured = "candidate_id" in example or any(
            key in example for key in ("coverage_obligations", "answer_units")
        )
        if structured:
            units = build_support_units(bundle.chunks)
            schema = candidate_schema(
                [f"[E{i}]" for i in range(1, len(bundle.chunks) + 1)],
                [unit.support_unit_id for unit in units],
            )
            schema["properties"]["answer_style"]["enum"] = ["prose"]
            raw = {key: example[key] for key in schema["properties"] if key in example}
            if list(Draft202012Validator(schema).iter_errors(raw)):
                return False, "v17_turbo_schema_invalid"
            example.update(materialize_candidate(raw))
        else:
            units = []

        question = str(example.get("question") or "").strip()
        answer = str(example.get("answer") or "").strip()
        if not question or not answer:
            return False, "empty_question_or_answer"
        capacity_valid, capacity_reason = self._normalize_capacity_mode(example)
        if not capacity_valid:
            return False, capacity_reason
        style_valid, style_reason = validate_answer_style(example)
        if not style_valid:
            return False, style_reason
        evidence_like = set(_EVIDENCE_LIKE_BRACKET.findall(answer))
        malformed = {marker for marker in evidence_like if not _EVIDENCE_MARKER.fullmatch(marker)}
        if malformed:
            return False, "malformed_evidence_markers"
        allowed = {f"[E{index}]" for index in range(1, len(bundle.chunks) + 1)}
        used = set(_EVIDENCE_MARKER.findall(answer))
        if not used:
            return False, "missing_evidence_markers"
        if not used.issubset(allowed):
            return False, "out_of_range_evidence_markers"
        if contains_non_corpus_citation(answer):
            return False, "non_corpus_secondary_citation"
        if citation_coverage(answer) < float(self.teacher.get("minimum_training_citation_coverage", 0.8)):
            return False, "insufficient_citation_coverage"
        token_count = self._chat_token_count(self._training_messages(bundle, question, answer))
        if token_count > int(self.settings.get("training.max_length")):
            return False, "sample_exceeds_training_context"
        if structured:
            coverage = validate_structure(example, units, lane="core")
            if not coverage["passed"]:
                return False, str(coverage["reasons"][0])
        return True, "ok"

    def _checkpoint_policy_fingerprint(self, **_: Any) -> str:
        policy = {
            "architecture": "V17-Turbo",
            "contract_revision": int(self.settings.get("v17_turbo.contract_revision")),
            "teacher": self.teacher,
            "models": {
                "teacher": self.settings.get("models.teacher"),
                "base": self.settings.get("models.base"),
            },
            "capacity": self.settings.section("v17_turbo_capacity"),
            "coverage": self.settings.section("v17_turbo_coverage"),
            "training_max_length": int(self.settings.get("training.max_length")),
        }
        return content_version([policy])
