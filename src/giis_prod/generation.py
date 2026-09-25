from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass, field

from .citations import (
    citation_coverage,
    contains_non_corpus_citation,
    render_inline_citations,
    render_reference_panel,
    validate_markers,
)
from .config import Settings
from .models import Evidence, ResponseMode
from .policy import evidence_is_sufficient, select_response_mode
from .retrieval import HybridRetriever


@dataclass(slots=True)
class AnswerResult:
    answer: str
    references_markdown: str
    mode: ResponseMode
    abstained: bool
    evidence: list[Evidence]
    rewritten_query: str
    raw_marker_answer: str = ""
    abstention_reason: str | None = None
    rejection_reason_codes: list[str] = field(default_factory=list)
    repair_attempts: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    obligations: list[str] = field(default_factory=list)
    obligation_coverage: list[dict[str, object]] = field(default_factory=list)
    coverage_complete: bool | None = None
    retrieval_passes: int = 1
    atomic_planning_succeeded: bool | None = None


@dataclass(slots=True)
class _CompletionResult:
    text: str
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


def _completion_result(completion: object) -> _CompletionResult:
    choices = getattr(completion, "choices", [])
    text = ""
    finish_reason = None
    if choices:
        text = (getattr(choices[0].message, "content", None) or "").strip()
        finish_reason = getattr(choices[0], "finish_reason", None)
    usage = getattr(completion, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
    if not total_tokens:
        total_tokens = prompt_tokens + completion_tokens
    return _CompletionResult(
        text=text,
        finish_reason=finish_reason,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def _usage_totals(outputs: list[_CompletionResult]) -> dict[str, int]:
    return {
        "prompt_tokens": sum(output.prompt_tokens for output in outputs),
        "completion_tokens": sum(output.completion_tokens for output in outputs),
        "total_tokens": sum(output.total_tokens for output in outputs),
    }


def _evidence_prompt(evidence: list[Evidence]) -> str:
    blocks = []
    for item in evidence:
        blocks.append(
            f"[{item.evidence_id}]\n"
            f"Document ID: {item.document_id}\n"
            f"Title: {item.title}\n"
            f"Pages: {item.page_start}-{item.page_end}\n"
            f"Text: {item.text}"
        )
    return "\n\n".join(blocks)


def _coverage_prompt(
    obligations: list[str], obligation_coverage: list[dict[str, object]]
) -> str:
    if not obligations:
        return ""
    coverage_by_index = {
        int(row.get("obligation_index", 0)): row for row in obligation_coverage
    }
    lines = [
        "Evidence requirements derived from the question. Address every covered requirement; "
        "when a requirement is not covered, state the evidence limitation rather than inventing an answer:"
    ]
    for index, obligation in enumerate(obligations, 1):
        row = coverage_by_index.get(index, {})
        status = "covered" if bool(row.get("covered", False)) else "not fully covered"
        ids = ", ".join(str(value) for value in row.get("evidence_ids", []))
        suffix = f"; mapped passages: {ids}" if ids else ""
        lines.append(f"{index}. [{status}{suffix}] {obligation}")
    return "\n".join(lines)


class GroundedGenerator:
    def __init__(self, settings: Settings):
        from openai import OpenAI

        self.settings = settings
        self.response = settings.section("response")
        self.retrieval = settings.section("retrieval")
        serving = settings.section("serving")
        if "enable_thinking" not in serving or not isinstance(
            serving["enable_thinking"], bool
        ):
            raise ValueError("serving.enable_thinking must be explicitly configured as a boolean")
        self.enable_thinking = serving["enable_thinking"]
        self.retriever = HybridRetriever(settings)
        api_key = os.getenv("VLLM_API_KEY")
        if not api_key:
            raise RuntimeError("VLLM_API_KEY must be set")
        port = int(settings.get("serving.vllm_port"))
        self.client = OpenAI(
            base_url=f"http://127.0.0.1:{port}/v1",
            api_key=api_key,
            timeout=float(settings.get("serving.request_timeout_seconds")),
        )
        self.model = str(settings.get("models.served"))

    def _chat_template_options(self) -> dict[str, Any]:
        """Return the explicit Qwen chat-template mode for every inference request."""
        return {
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": bool(getattr(self, "enable_thinking", False))
                }
            }
        }

    def _system_prompt(self, mode: ResponseMode) -> str:
        base = (
            "You are the GIIS corpus research assistant. Use only the supplied evidence. "
            "Never use foundational, remembered, or outside knowledge. "
            "Never cite or directly attribute a claim to an author, study, report, or numbered "
            "reference merely mentioned inside a corpus paper. A source is directly citable only "
            "when it is supplied under its own Document ID; omit all second-hand citations. "
            "Synthesize by theme, represent limitations and disagreement, and do not expose hidden reasoning."
        )
        if mode is ResponseMode.SOURCE_SUMMARY:
            return (
                base
                + " Produce a concise IMRaD-oriented source summary using only sections actually "
                "supported by the evidence. Include limitations and implications when supported. "
                "Do not emit [E#] markers, in-text citations, footnotes, or a reference list because "
                "the selected source is already identified in the reference panel."
            )
        base += (
            " Every factual sentence must end with one or more exact evidence markers such as [E1]. "
            "Do not invent markers."
        )
        if mode is ResponseMode.LONG_FORM:
            return (
                base
                + f" Produce {self.response['target_long_form_words']}-{self.response['maximum_long_form_words']} "
                "words with clear thematic headings. Use tables only when they improve comparison."
            )
        return base + " Answer directly and proportionately; do not pad a narrow or closed question."

    def _generate(
        self,
        query: str,
        evidence: list[Evidence],
        mode: ResponseMode,
        obligations: list[str] | None = None,
        obligation_coverage: list[dict[str, object]] | None = None,
        expanded_capacity: bool = False,
    ) -> _CompletionResult:
        max_tokens = (
            int(self.response["long_form_max_tokens"])
            if mode is ResponseMode.LONG_FORM or expanded_capacity
            else int(self.response["summary_max_tokens"])
            if mode is ResponseMode.SOURCE_SUMMARY
            else int(self.response["standard_max_tokens"])
        )
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._system_prompt(mode)},
                {
                    "role": "user",
                    "content": (
                        f"Evidence:\n{_evidence_prompt(evidence)}\n\n"
                        f"{_coverage_prompt(obligations or [], obligation_coverage or [])}\n\n"
                        f"Question: {query}"
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=max_tokens,
            **self._chat_template_options(),
        )
        return _completion_result(completion)

    def _repair_once(
        self,
        query: str,
        answer: str,
        evidence: list[Evidence],
        mode: ResponseMode,
        obligations: list[str] | None = None,
        obligation_coverage: list[dict[str, object]] | None = None,
        expanded_capacity: bool = False,
    ) -> _CompletionResult:
        minimum_words = int(self.response["minimum_long_form_words"])
        length_instruction = (
            f"Reach at least {minimum_words} words only through deeper supported synthesis. "
            if mode in {ResponseMode.LONG_FORM, ResponseMode.SOURCE_SUMMARY}
            else "Keep the response proportional to the question. "
        )
        prompt = (
            "Revise the draft using only the evidence. Preserve supported content, remove unsupported "
            "claims, remove every author-year or numbered citation to a work that is not supplied under "
            "its own Document ID, ensure every factual sentence ends in valid [E#] markers, and do not "
            "invent facts. "
            f"{length_instruction}\n\n"
            f"Question: {query}\n\nEvidence:\n{_evidence_prompt(evidence)}\n\n"
            f"{_coverage_prompt(obligations or [], obligation_coverage or [])}\n\nDraft:\n{answer}"
        )
        max_tokens = (
            int(self.response["long_form_max_tokens"])
            if mode is ResponseMode.LONG_FORM or expanded_capacity
            else int(self.response["summary_max_tokens"])
            if mode is ResponseMode.SOURCE_SUMMARY
            else int(self.response["standard_max_tokens"])
        )
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._system_prompt(mode)},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
            **self._chat_template_options(),
        )
        return _completion_result(completion)

    def _response_rejection_reasons(
        self,
        answer: str,
        evidence: list[Evidence],
        mode: ResponseMode,
        finish_reason: str | None = None,
    ) -> list[str]:
        valid_markers, _ = validate_markers(answer, evidence)
        reasons: list[str] = []
        if not valid_markers:
            reasons.append("invalid_or_missing_evidence_markers")
        if citation_coverage(answer) < float(self.response["minimum_citation_coverage"]):
            reasons.append("insufficient_citation_coverage")
        if mode in {ResponseMode.LONG_FORM, ResponseMode.SOURCE_SUMMARY} and len(
            answer.split()
        ) < int(self.response["minimum_long_form_words"]):
            reasons.append("long_form_too_short")
        if contains_non_corpus_citation(answer):
            reasons.append("non_corpus_secondary_citation")
        if finish_reason == "length":
            reasons.append("truncated_at_completion_limit")
        return reasons

    def answer(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
        requested_mode: ResponseMode = ResponseMode.AUTO,
    ) -> AnswerResult:
        mode = select_response_mode(query, requested_mode)
        retrieval = self.retriever.retrieve(query, history)
        sufficient = evidence_is_sufficient(
            retrieval.evidence,
            mode,
            int(self.retrieval["minimum_distinct_sources_for_long_form"]),
        )
        if retrieval.abstained or not sufficient:
            message = str(self.response["no_evidence_message"])
            reason = (
                retrieval.reason
                if retrieval.abstained and retrieval.reason
                else "insufficient_evidence_for_requested_mode"
            )
            return AnswerResult(
                answer=message,
                references_markdown="## Reference Panel\n\nNo qualifying corpus evidence.",
                mode=ResponseMode.NO_EVIDENCE,
                abstained=True,
                evidence=retrieval.evidence,
                rewritten_query=retrieval.rewritten_query,
                abstention_reason=reason,
                rejection_reason_codes=[reason],
                obligations=list(getattr(retrieval, "obligations", [])),
                obligation_coverage=list(getattr(retrieval, "obligation_coverage", [])),
                coverage_complete=getattr(retrieval, "coverage_complete", None),
                retrieval_passes=int(getattr(retrieval, "retrieval_passes", 1)),
                atomic_planning_succeeded=getattr(
                    retrieval, "atomic_planning_succeeded", None
                ),
            )

        obligations = list(getattr(retrieval, "obligations", []))
        obligation_coverage = list(getattr(retrieval, "obligation_coverage", []))
        outputs = [
            self._generate(
                query,
                retrieval.evidence,
                mode,
                obligations,
                obligation_coverage,
            )
        ]
        marker_answer = outputs[-1].text
        rejection_reasons = self._response_rejection_reasons(
            marker_answer,
            retrieval.evidence,
            mode,
            outputs[-1].finish_reason,
        )
        repair_attempts = 0
        if rejection_reasons:
            outputs.append(
                self._repair_once(
                    query,
                    marker_answer,
                    retrieval.evidence,
                    mode,
                    obligations,
                    obligation_coverage,
                    expanded_capacity=outputs[-1].finish_reason == "length",
                )
            )
            marker_answer = outputs[-1].text
            repair_attempts = 1
            rejection_reasons = self._response_rejection_reasons(
                marker_answer,
                retrieval.evidence,
                mode,
                outputs[-1].finish_reason,
            )
        usage = _usage_totals(outputs)
        if rejection_reasons:
            message = str(
                self.response.get(
                    "validation_failure_message",
                    "The generated response failed grounding validation.",
                )
            )
            return AnswerResult(
                answer=message,
                references_markdown=render_reference_panel(retrieval.evidence),
                mode=ResponseMode.NO_EVIDENCE,
                abstained=True,
                evidence=retrieval.evidence,
                rewritten_query=retrieval.rewritten_query,
                raw_marker_answer=marker_answer,
                abstention_reason="response_policy_rejected_after_repair",
                rejection_reason_codes=rejection_reasons,
                repair_attempts=repair_attempts,
                obligations=obligations,
                obligation_coverage=obligation_coverage,
                coverage_complete=getattr(retrieval, "coverage_complete", None),
                retrieval_passes=int(getattr(retrieval, "retrieval_passes", 1)),
                atomic_planning_succeeded=getattr(
                    retrieval, "atomic_planning_succeeded", None
                ),
                **usage,
            )
        return AnswerResult(
            answer=render_inline_citations(marker_answer, retrieval.evidence),
            references_markdown=render_reference_panel(retrieval.evidence),
            mode=mode,
            abstained=False,
            evidence=retrieval.evidence,
            rewritten_query=retrieval.rewritten_query,
            raw_marker_answer=marker_answer,
            repair_attempts=repair_attempts,
            obligations=obligations,
            obligation_coverage=obligation_coverage,
            coverage_complete=getattr(retrieval, "coverage_complete", None),
            retrieval_passes=int(getattr(retrieval, "retrieval_passes", 1)),
            atomic_planning_succeeded=getattr(
                retrieval, "atomic_planning_succeeded", None
            ),
            **usage,
        )

    def summarize_source(self, document_id: str) -> AnswerResult:
        evidence = self.retriever.document_evidence(
            document_id,
            max_chunks=int(self.response["brief_summary_max_chunks"]),
        )
        if not evidence:
            message = str(self.response["no_evidence_message"])
            return AnswerResult(
                answer=message,
                references_markdown="## Reference Panel\n\nDocument not found in the active corpus.",
                mode=ResponseMode.NO_EVIDENCE,
                abstained=True,
                evidence=[],
                rewritten_query=document_id,
            )

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._system_prompt(ResponseMode.SOURCE_SUMMARY)},
                {
                    "role": "user",
                    "content": (
                        f"Create a concise {self.response['brief_summary_target_words']}-word overview "
                        "of this source in one pass. Put each supported section heading on its own line "
                        "in exactly this Markdown form: **Purpose**, **Approach**, **Findings**, and "
                        "**Limitations or Implications**. Begin each section's content on the following "
                        "line. Do not emit [E#] markers, "
                        "citations, footnotes, or a reference list. Do not pad, repeat, or introduce "
                        f"outside knowledge.\n\nEvidence:\n{_evidence_prompt(evidence)}"
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=int(self.response["brief_summary_max_tokens"]),
            **self._chat_template_options(),
        )
        output = _completion_result(completion)
        marker_answer = output.text
        # A brief summary is scoped to one selected source, so it does not need [E#]
        # attribution. Keep deterministic completeness and provenance checks without
        # a second model call.
        rejection_reasons: list[str] = []
        if not marker_answer:
            rejection_reasons.append("empty_source_summary")
        if output.finish_reason == "length":
            rejection_reasons.append("truncated_source_summary")
        if contains_non_corpus_citation(marker_answer):
            rejection_reasons.append("non_corpus_secondary_citation")
        usage = _usage_totals([output])
        if rejection_reasons:
            message = str(
                self.response.get(
                    "summary_validation_failure_message",
                    "The generated source summary was incomplete and could not be released.",
                )
            )
            return AnswerResult(
                answer=message,
                references_markdown=render_reference_panel(evidence),
                mode=ResponseMode.NO_EVIDENCE,
                abstained=True,
                evidence=evidence,
                rewritten_query=document_id,
                raw_marker_answer=marker_answer,
                abstention_reason="brief_source_summary_rejected",
                rejection_reason_codes=rejection_reasons,
                repair_attempts=0,
                **usage,
            )
        return AnswerResult(
            answer=marker_answer,
            references_markdown=render_reference_panel(evidence),
            mode=ResponseMode.SOURCE_SUMMARY,
            abstained=False,
            evidence=evidence,
            rewritten_query=document_id,
            raw_marker_answer=marker_answer,
            repair_attempts=0,
            **usage,
        )

    def summarize_source_stream(self, document_id: str) -> Iterator[str]:
        """Yield an accumulating source summary and replace it with validation failure if needed."""

        evidence = self.retriever.document_evidence(
            document_id,
            max_chunks=int(self.response["brief_summary_max_chunks"]),
        )
        if not evidence:
            yield str(self.response["no_evidence_message"])
            return

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._system_prompt(ResponseMode.SOURCE_SUMMARY)},
                {
                    "role": "user",
                    "content": (
                        f"Create a concise {self.response['brief_summary_target_words']}-word overview "
                        "of this source in one pass. Put each supported section heading on its own line "
                        "in exactly this Markdown form: **Purpose**, **Approach**, **Findings**, and "
                        "**Limitations or Implications**. Begin each section's content on the following "
                        "line. Do not emit "
                        "[E#] markers, citations, footnotes, or a reference list. Do not pad, repeat, or "
                        f"introduce outside knowledge.\n\nEvidence:\n{_evidence_prompt(evidence)}"
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=int(self.response["brief_summary_max_tokens"]),
            stream=True,
            **self._chat_template_options(),
        )
        parts: list[str] = []
        finish_reason: str | None = None
        last_emitted = ""
        for chunk in stream:
            choices = getattr(chunk, "choices", [])
            if not choices:
                continue
            choice = choices[0]
            content = getattr(getattr(choice, "delta", None), "content", None)
            if content:
                parts.append(str(content))
                accumulated = "".join(parts).strip()
                if accumulated and accumulated != last_emitted:
                    last_emitted = accumulated
                    yield accumulated
            chunk_finish_reason = getattr(choice, "finish_reason", None)
            if chunk_finish_reason is not None:
                finish_reason = str(chunk_finish_reason)

        final = "".join(parts).strip()
        rejection_reasons: list[str] = []
        if not final:
            rejection_reasons.append("empty_source_summary")
        if finish_reason == "length":
            rejection_reasons.append("truncated_source_summary")
        if contains_non_corpus_citation(final):
            rejection_reasons.append("non_corpus_secondary_citation")
        if rejection_reasons:
            yield str(
                self.response.get(
                    "summary_validation_failure_message",
                    "The generated source summary was incomplete and could not be released.",
                )
            )
        elif final and final != last_emitted:
            yield final
