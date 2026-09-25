from __future__ import annotations

import json
import math
import random
import re
import time
from collections import Counter
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from .citations import citation_coverage
from .config import Settings
from .io import content_version, read_jsonl, sha256_file, write_json_atomic
from .server import teacher_server_profile
from .teacher import (
    EvidenceBundle,
    TeacherDatasetBuilder,
    _cited_document_ids,
    build_bundles,
)

_ARMS = {"baseline": "ab-baseline", "candidate": "ab-candidate"}
_LEGACY_LANE = "legacy_v15rc"
_REFINED_LANE = "refined_multisource_v1"
_SINGLE_CONTROL_LANE = "single_source_control"
_RELATION_MODES = {
    "methodological_contrast",
    "complementary_findings",
    "explicit_causal_chain",
}
_MARKER = re.compile(r"\[E\d+\]")


def _output_dir(settings: Settings) -> Path:
    configured = Path(str(settings.get("teacher_ab.output_dir")))
    return configured if configured.is_absolute() else (settings.root / configured).resolve()


def _bundle_to_dict(bundle: EvidenceBundle) -> dict[str, Any]:
    return {
        "bundle_id": bundle.bundle_id,
        "document_ids": list(bundle.document_ids),
        "chunks": bundle.chunks,
    }


def _bundle_from_dict(row: dict[str, Any]) -> EvidenceBundle:
    return EvidenceBundle(
        bundle_id=str(row["bundle_id"]),
        document_ids=tuple(str(value) for value in row["document_ids"]),
        chunks=list(row["chunks"]),
    )


def _create_legacy_teacher_ab_plan(
    settings: Settings, request_count: int | None = None
) -> dict[str, Any]:
    release_path = settings.path("paths.sft_dir") / "dataset_manifest.json"
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if release.get("status") != "complete":
        raise RuntimeError("A completed immutable dataset release is required before A/B planning")

    count = request_count or int(settings.get("teacher_ab.request_count"))
    if count < 2:
        raise ValueError("The A/B plan requires at least two requests")
    multi_fraction = float(settings.get("teacher_ab.multi_source_fraction"))
    if not 0.0 <= multi_fraction <= 1.0:
        raise ValueError("teacher_ab.multi_source_fraction must be between zero and one")

    selection = json.loads(
        settings.path("paths.training_selection_manifest").read_text(encoding="utf-8")
    )
    document_ids = {
        str(row["document_id"]) for row in selection.get("documents", [])
    }
    chunks = read_jsonl(settings.path("paths.chunk_file"))
    desired_multi = min(count - 1, math.ceil(count * multi_fraction))
    bundle_pool = build_bundles(
        chunks,
        document_ids,
        multi_source_fraction=desired_multi / max(1, len(document_ids)),
        seed=int(settings.get("teacher.seed")),
        chunks_per_document=int(settings.get("teacher.evidence_chunks_per_document")),
        minimum_substantive_chunks=int(
            settings.get("teacher.minimum_substantive_chunks_per_document")
        ),
    )
    singles = sorted(
        (bundle for bundle in bundle_pool if len(bundle.document_ids) == 1),
        key=lambda bundle: bundle.bundle_id,
    )
    multis = sorted(
        (bundle for bundle in bundle_pool if len(bundle.document_ids) > 1),
        key=lambda bundle: bundle.bundle_id,
    )
    if len(singles) < count - desired_multi or len(multis) < desired_multi:
        raise RuntimeError("The corpus cannot provide the requested frozen A/B bundle mixture")

    rng = random.Random(int(settings.get("teacher.seed")))  # noqa: S311
    rng.shuffle(singles)
    rng.shuffle(multis)
    selected = singles[: count - desired_multi] + multis[:desired_multi]
    rng.shuffle(selected)
    requested_examples = int(settings.get("teacher_ab.requested_examples"))
    requests = [
        {
            "request_id": f"teacher_ab_{index:03d}",
            "requested_examples": requested_examples,
            "allow_long_form": True,
            "bundle": _bundle_to_dict(bundle),
        }
        for index, bundle in enumerate(selected, start=1)
    ]
    plan_digest = content_version(requests)
    plan = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "plan_digest": plan_digest,
        "policy_fingerprint": release["policy_fingerprint"],
        "dataset_manifest_sha256": sha256_file(release_path),
        "training_selection_digest": selection["selection_digest"],
        "request_count": len(requests),
        "single_source_requests": sum(
            len(row["bundle"]["document_ids"]) == 1 for row in requests
        ),
        "multi_source_requests": sum(
            len(row["bundle"]["document_ids"]) > 1 for row in requests
        ),
        "requests": requests,
    }
    destination = _output_dir(settings) / f"plan_{plan_digest[:12]}.json"
    write_json_atomic(destination, plan)
    return {
        "status": "planned",
        "path": str(destination),
        "plan_digest": plan_digest,
        "request_count": len(requests),
        "single_source_requests": plan["single_source_requests"],
        "multi_source_requests": plan["multi_source_requests"],
    }


def _configured_path(settings: Settings, dotted_key: str) -> Path:
    configured = Path(str(settings.get(dotted_key)))
    return configured if configured.is_absolute() else (settings.root / configured).resolve()


def _even_sample(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if len(rows) < count:
        raise RuntimeError(f"Need {count} evidence chunks but found only {len(rows)}")
    if len(rows) == count:
        return list(rows)
    positions = {
        round(index * (len(rows) - 1) / (count - 1)) for index in range(count)
    }
    selected = [rows[position] for position in sorted(positions)]
    if len(selected) != count:
        raise RuntimeError("Deterministic evidence sampling produced the wrong count")
    return selected


def _create_factorial_teacher_ab_plan(settings: Settings) -> dict[str, Any]:
    release_path = settings.path("paths.sft_dir") / "dataset_manifest.json"
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if release.get("status") != "complete":
        raise RuntimeError("A completed immutable dataset release is required before A/B planning")
    selection = json.loads(
        settings.path("paths.training_selection_manifest").read_text(encoding="utf-8")
    )
    document_ids = {str(row["document_id"]) for row in selection.get("documents", [])}
    chunks = read_jsonl(settings.path("paths.chunk_file"))
    single_bundles = build_bundles(
        chunks,
        document_ids,
        multi_source_fraction=0.0,
        seed=int(settings.get("teacher.seed")),
        chunks_per_document=int(settings.get("teacher.evidence_chunks_per_document")),
        minimum_substantive_chunks=int(
            settings.get("teacher.minimum_substantive_chunks_per_document")
        ),
    )
    by_document = {
        bundle.document_ids[0]: bundle
        for bundle in single_bundles
        if len(bundle.document_ids) == 1
    }
    pair_path = _configured_path(settings, "teacher_ab.factorial_pair_manifest")
    pair_manifest = json.loads(pair_path.read_text(encoding="utf-8"))
    if pair_manifest.get("schema_version") != 1:
        raise ValueError("Factorial pair manifest must use schema_version 1")
    pair_specs = pair_manifest.get("pairs")
    if not isinstance(pair_specs, list) or len(pair_specs) != 3:
        raise ValueError("Factorial v2 requires exactly three compatible source pairs")

    pair_document_ids: set[str] = set()
    pair_rows: list[dict[str, Any]] = []
    per_source = int(settings.get("teacher_ab.factorial_chunks_per_source"))
    for position, spec in enumerate(pair_specs, start=1):
        pair_id = str(spec.get("pair_id") or "").strip()
        sources = [str(value) for value in spec.get("document_ids", [])]
        relation_mode = str(spec.get("relation_mode") or "")
        if (
            not pair_id
            or len(sources) != 2
            or len(set(sources)) != 2
            or relation_mode not in _RELATION_MODES
        ):
            raise ValueError(f"Invalid factorial pair specification at position {position}")
        if any(source not in by_document for source in sources):
            raise ValueError(f"Factorial pair {pair_id} is outside the frozen SFT pool")
        pair_document_ids.update(sources)
        combined: list[dict[str, Any]] = []
        for source in sources:
            combined.extend(_even_sample(by_document[source].chunks, per_source))
        bundle = EvidenceBundle(
            bundle_id="bundle_factorial_"
            + content_version(
                [{
                    "pair_id": pair_id,
                    "document_ids": sources,
                    "chunk_ids": [row["chunk_id"] for row in combined],
                }]
            )[:20],
            document_ids=tuple(sorted(sources)),
            chunks=combined,
        )
        for lane in (_LEGACY_LANE, _REFINED_LANE):
            pair_rows.append(
                {
                    "request_id": f"teacher_factorial_{pair_id}_{lane}",
                    "policy_lane": lane,
                    "pair_id": pair_id,
                    "relation_mode": relation_mode,
                    "source_document_ids": sources,
                    "requested_examples": int(
                        settings.get("teacher_ab.requested_examples")
                    ),
                    "allow_long_form": True,
                    "bundle": _bundle_to_dict(bundle),
                }
            )

    control_count = int(settings.get("teacher_ab.single_source_control_requests"))
    eligible_controls = sorted(
        (
            bundle
            for document_id, bundle in by_document.items()
            if document_id not in pair_document_ids
        ),
        key=lambda bundle: bundle.bundle_id,
    )
    rng = random.Random(int(settings.get("teacher.seed")))  # noqa: S311
    rng.shuffle(eligible_controls)
    if len(eligible_controls) < control_count:
        raise RuntimeError("Not enough independent single-source controls for factorial v2")
    control_rows = [
        {
            "request_id": f"teacher_factorial_single_{index:03d}",
            "policy_lane": _SINGLE_CONTROL_LANE,
            "pair_id": None,
            "relation_mode": None,
            "source_document_ids": list(bundle.document_ids),
            "requested_examples": int(settings.get("teacher_ab.requested_examples")),
            "allow_long_form": True,
            "bundle": _bundle_to_dict(bundle),
        }
        for index, bundle in enumerate(eligible_controls[:control_count], start=1)
    ]
    requests = control_rows + pair_rows
    plan_digest = content_version(requests)
    policy_definition = {
        "legacy_lane": _LEGACY_LANE,
        "refined_lane": _REFINED_LANE,
        "relation_modes": sorted(_RELATION_MODES),
        "support_ledger_required": True,
    }
    plan = {
        "schema_version": 2,
        "design": "two_by_two_factorial_v2",
        "created_at": datetime.now(UTC).isoformat(),
        "plan_digest": plan_digest,
        "policy_definition_digest": content_version([policy_definition]),
        "policy_definition": policy_definition,
        "policy_fingerprint": release["policy_fingerprint"],
        "dataset_manifest_sha256": sha256_file(release_path),
        "training_selection_digest": selection["selection_digest"],
        "pair_manifest_path": str(pair_path),
        "pair_manifest_sha256": sha256_file(pair_path),
        "request_count": len(requests),
        "single_source_requests": len(control_rows),
        "multi_source_requests": len(pair_rows),
        "factorial_cells_per_infrastructure_arm": {
            _SINGLE_CONTROL_LANE: len(control_rows),
            _LEGACY_LANE: len(pair_specs),
            _REFINED_LANE: len(pair_specs),
        },
        "requests": requests,
    }
    destination = _output_dir(settings) / f"plan_factorial_v2_{plan_digest[:12]}.json"
    write_json_atomic(destination, plan)
    return {
        "status": "planned",
        "design": plan["design"],
        "path": str(destination),
        "plan_digest": plan_digest,
        "request_count": len(requests),
        "single_source_requests": plan["single_source_requests"],
        "multi_source_requests": plan["multi_source_requests"],
        "factorial_cells_per_infrastructure_arm": plan[
            "factorial_cells_per_infrastructure_arm"
        ],
    }


def create_teacher_ab_plan(
    settings: Settings, request_count: int | None = None
) -> dict[str, Any]:
    design = str(settings.get("teacher_ab.design"))
    if design == "two_by_two_factorial_v2":
        if request_count is not None:
            raise ValueError(
                "--requests cannot override the frozen factorial design; edit and reapprove "
                "the pair manifest instead"
            )
        return _create_factorial_teacher_ab_plan(settings)
    return _create_legacy_teacher_ab_plan(settings, request_count=request_count)


def _flag_value(arguments: list[str], flag: str) -> str | None:
    try:
        return arguments[arguments.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def running_teacher_profile(settings: Settings) -> dict[str, Any]:
    expected_port = str(settings.get("teacher_server.port"))
    teacher_source = str(settings.get("models.teacher_source"))
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            arguments = [
                value.decode("utf-8", errors="replace")
                for value in (process / "cmdline").read_bytes().split(b"\0")
                if value
            ]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if (
            "serve" not in arguments
            or teacher_source not in arguments
            or _flag_value(arguments, "--port") != expected_port
        ):
            continue
        model_len = _flag_value(arguments, "--max-model-len")
        max_sequences = _flag_value(arguments, "--max-num-seqs")
        if model_len is None or max_sequences is None:
            continue
        return {
            "pid": int(process.name),
            "max_model_len": int(model_len),
            "max_num_seqs": int(max_sequences),
        }
    raise RuntimeError("No matching teacher vLLM process was found")


def assert_context_fit(
    prompt_tokens: int, max_output_tokens: int, safety_margin: int, model_context: int
) -> int:
    required = prompt_tokens + max_output_tokens + safety_margin
    if required > model_context:
        raise RuntimeError(
            "A/B request cannot fit without truncation or a smaller output allowance: "
            f"prompt={prompt_tokens}, output={max_output_tokens}, margin={safety_margin}, "
            f"required={required}, context={model_context}"
        )
    return required


def _prompt_token_count(tokenizer: Any, request: dict[str, Any]) -> int:
    template_kwargs = (
        request.get("extra_body", {})
        .get("chat_template_kwargs", {})
    )
    encoded = tokenizer.apply_chat_template(
        request["messages"],
        tokenize=True,
        add_generation_prompt=True,
        **template_kwargs,
    )
    return TeacherDatasetBuilder._token_count(encoded)


def _health_check(settings: Settings) -> None:
    url = (
        f"http://{settings.get('teacher_server.host')}:"
        f"{settings.get('teacher_server.port')}/health"
    )
    with urlopen(url, timeout=5) as response:  # noqa: S310 - configured loopback endpoint
        if response.status != 200:
            raise RuntimeError(f"Teacher health endpoint returned HTTP {response.status}")


def _markers_by_document(bundle: EvidenceBundle) -> dict[str, list[str]]:
    markers: dict[str, list[str]] = {document_id: [] for document_id in bundle.document_ids}
    for index, chunk in enumerate(bundle.chunks, start=1):
        markers.setdefault(str(chunk["document_id"]), []).append(f"[E{index}]")
    return markers


def _refined_request_options(
    base_request: dict[str, Any], bundle: EvidenceBundle, row: dict[str, Any]
) -> dict[str, Any]:
    """Add the experimental support ledger without altering production generation code."""
    if row.get("policy_lane") != _REFINED_LANE:
        return deepcopy(base_request)
    sources = [str(value) for value in row.get("source_document_ids", [])]
    relation_mode = str(row.get("relation_mode") or "")
    if len(sources) != 2 or relation_mode not in _RELATION_MODES:
        raise ValueError("Refined multi-source requests require two sources and one relation mode")
    markers = _markers_by_document(bundle)
    if any(not markers.get(source) for source in sources):
        raise ValueError("Refined multi-source request lacks evidence for one declared source")

    request = deepcopy(base_request)
    prompt = str(request["messages"][-1]["content"])
    policy_block = f"""

FACTORIAL REFINED MULTI-SOURCE POLICY (benchmark metadata, not student-facing text):
- The allowed synthesis relation for this request is exactly: {relation_mode}.
- First establish one evidence-supported claim from each declared source.
- The relationship statement must be supported by passages from both sources; similarity of topic
  alone is not support.
- Do not infer cross-domain transfer, analogy, integrated recommendations, policy prescriptions, or
  a causal link absent from the excerpts.
- Return a support_ledger for every example. The ledger is an audit object and must not be mentioned
  in the question or answer.
- source_a_claim and source_b_claim must state only what their respective evidence markers support.
- relationship_evidence must include markers from both documents and those markers must also appear
  in the answer.
- unsupported_transfer_present must be false. If it would be true, choose another fully answerable
  question using the same permitted relation mode.
"""
    anchor = "\nEVIDENCE:\n"
    if anchor not in prompt:
        raise ValueError("Teacher prompt no longer exposes the expected evidence anchor")
    request["messages"][-1]["content"] = prompt.replace(
        anchor, policy_block + anchor, 1
    )

    schema = request["response_format"]["json_schema"]["schema"]
    example_schema = schema["properties"]["examples"]["items"]
    example_schema["required"].append("support_ledger")
    example_schema["properties"]["support_ledger"] = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "source_a_document_id",
            "source_a_claim",
            "source_a_evidence",
            "source_b_document_id",
            "source_b_claim",
            "source_b_evidence",
            "relation_type",
            "relationship_statement",
            "relationship_evidence",
            "unsupported_transfer_present",
        ],
        "properties": {
            "source_a_document_id": {"type": "string", "enum": [sources[0]]},
            "source_a_claim": {"type": "string", "minLength": 1},
            "source_a_evidence": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "enum": markers[sources[0]]},
            },
            "source_b_document_id": {"type": "string", "enum": [sources[1]]},
            "source_b_claim": {"type": "string", "minLength": 1},
            "source_b_evidence": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "enum": markers[sources[1]]},
            },
            "relation_type": {"type": "string", "enum": [relation_mode]},
            "relationship_statement": {"type": "string", "minLength": 1},
            "relationship_evidence": {
                "type": "array",
                "minItems": 2,
                "items": {
                    "type": "string",
                    "enum": sorted(marker for values in markers.values() for marker in values),
                },
            },
            "unsupported_transfer_present": {"type": "boolean", "enum": [False]},
        },
    }
    request["response_format"]["json_schema"]["name"] = (
        "giis_factorial_refined_multisource"
    )
    return request


def _request_options_for_row(
    builder: TeacherDatasetBuilder,
    bundle: EvidenceBundle,
    row: dict[str, Any],
) -> dict[str, Any]:
    base = builder._generation_request_options(
        bundle,
        allow_long_form=bool(row["allow_long_form"]),
        requested_examples=int(row["requested_examples"]),
    )
    return _refined_request_options(base, bundle, row)


def _request_examples_for_row(
    builder: TeacherDatasetBuilder,
    bundle: EvidenceBundle,
    row: dict[str, Any],
    request: dict[str, Any],
) -> list[dict[str, Any]]:
    response = builder.client.chat.completions.create(**request)
    content = response.choices[0].message.content
    if not content:
        raise ValueError("Teacher returned empty structured output")
    payload = json.loads(content)
    examples = list(payload["examples"])
    if len(examples) != int(row["requested_examples"]):
        raise ValueError("Teacher changed the requested factorial candidate count")
    builder._assign_candidate_ids(bundle, examples)
    with builder._prevalidation_lock:
        builder._prevalidate_for_verifier(bundle, examples, [])
    return examples


def _validate_support_ledger(
    bundle: EvidenceBundle, row: dict[str, Any], candidate: dict[str, Any]
) -> tuple[bool, str]:
    if row.get("policy_lane") != _REFINED_LANE:
        return True, "ok"
    ledger = candidate.get("support_ledger")
    if not isinstance(ledger, dict):
        return False, "missing_support_ledger"
    sources = [str(value) for value in row.get("source_document_ids", [])]
    if (
        ledger.get("source_a_document_id") != sources[0]
        or ledger.get("source_b_document_id") != sources[1]
        or ledger.get("relation_type") != row.get("relation_mode")
        or ledger.get("unsupported_transfer_present") is not False
    ):
        return False, "invalid_support_ledger_contract"
    markers = _markers_by_document(bundle)
    source_a = ledger.get("source_a_evidence")
    source_b = ledger.get("source_b_evidence")
    relationship = ledger.get("relationship_evidence")
    if (
        not isinstance(source_a, list)
        or not source_a
        or not set(source_a).issubset(markers[sources[0]])
        or not isinstance(source_b, list)
        or not source_b
        or not set(source_b).issubset(markers[sources[1]])
        or not isinstance(relationship, list)
        or not relationship
    ):
        return False, "invalid_support_ledger_provenance"
    relationship_documents = {
        document_id
        for document_id, allowed in markers.items()
        if set(relationship) & set(allowed)
    }
    if not set(sources).issubset(relationship_documents):
        return False, "relationship_ledger_does_not_use_both_sources"
    answer_markers = set(_MARKER.findall(str(candidate.get("answer") or "")))
    if not set(relationship).issubset(answer_markers):
        return False, "relationship_ledger_markers_missing_from_answer"
    if not str(ledger.get("source_a_claim") or "").strip() or not str(
        ledger.get("source_b_claim") or ""
    ).strip() or not str(ledger.get("relationship_statement") or "").strip():
        return False, "empty_support_ledger_claim"
    return True, "ok"


def _validate_candidate(
    builder: TeacherDatasetBuilder,
    bundle: EvidenceBundle,
    candidate: dict[str, Any],
    row: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    valid, reason = builder._validate_example(bundle, candidate)
    if valid and len(bundle.document_ids) > 1:
        cited_documents = _cited_document_ids(bundle, str(candidate.get("answer") or ""))
        if len(cited_documents) < 2:
            return False, "insufficient_distinct_cited_documents"
    if valid and row is not None:
        ledger_valid, ledger_reason = _validate_support_ledger(bundle, row, candidate)
        if not ledger_valid:
            return False, ledger_reason
    return valid, reason


def run_teacher_ab(
    settings: Settings,
    plan_path: Path,
    arm: str,
    *,
    preflight_only: bool = False,
) -> dict[str, Any]:
    if arm not in _ARMS:
        raise ValueError(f"Unknown A/B arm: {arm}")
    plan_path = plan_path.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if content_version(plan["requests"]) != plan.get("plan_digest"):
        raise RuntimeError("A/B plan digest does not match its frozen requests")

    release_path = settings.path("paths.sft_dir") / "dataset_manifest.json"
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if release.get("policy_fingerprint") != plan.get("policy_fingerprint"):
        raise RuntimeError("A/B plan policy fingerprint differs from the dataset release")
    if sha256_file(release_path) != plan.get("dataset_manifest_sha256"):
        raise RuntimeError("The dataset manifest changed after the A/B plan was frozen")

    profile_name = _ARMS[arm]
    expected_profile = teacher_server_profile(settings, profile_name)
    actual_profile = running_teacher_profile(settings)
    expected_runtime = {
        "max_model_len": int(expected_profile["max_model_len"]),
        "max_num_seqs": int(expected_profile["max_num_seqs"]),
    }
    if any(actual_profile[key] != value for key, value in expected_runtime.items()):
        raise RuntimeError(
            f"Running teacher profile {actual_profile} does not match {profile_name} "
            f"requirements {expected_runtime}"
        )
    _health_check(settings)

    # Heavy dependencies are constructed only after all immutable-plan and server-profile checks.
    from transformers import AutoTokenizer

    builder = TeacherDatasetBuilder(settings)
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        settings.get("models.teacher_source")
    )
    margin = int(settings.get("teacher_ab.token_safety_margin"))
    prepared: list[
        tuple[dict[str, Any], EvidenceBundle, dict[str, Any], int, int]
    ] = []
    for row in plan["requests"]:
        bundle = _bundle_from_dict(row["bundle"])
        request = _request_options_for_row(builder, bundle, row)
        prompt_tokens = _prompt_token_count(teacher_tokenizer, request)
        required_tokens = assert_context_fit(
            prompt_tokens,
            int(request["max_tokens"]),
            margin,
            int(expected_profile["max_model_len"]),
        )
        prepared.append((row, bundle, request, prompt_tokens, required_tokens))

    if preflight_only:
        return {
            "status": "preflight_passed",
            "arm": arm,
            "server_profile": profile_name,
            "plan_path": str(plan_path),
            "plan_digest": plan["plan_digest"],
            "design": plan.get("design", "matched_infrastructure_ab_v1"),
            "request_count": len(prepared),
            "maximum_prompt_tokens": max(row[3] for row in prepared),
            "maximum_required_context_tokens": max(row[4] for row in prepared),
            "model_context": int(expected_profile["max_model_len"]),
            "output_allowance_unchanged": int(settings.get("teacher.max_output_tokens")),
            "token_safety_margin": margin,
            "generation_performed": False,
        }

    # Warm the exact structured-output path, then discard its output from every measurement.
    warm_row, warm_bundle, warm_request, _, _ = prepared[0]
    _request_examples_for_row(builder, warm_bundle, warm_row, warm_request)

    def execute(
        item: tuple[dict[str, Any], EvidenceBundle, dict[str, Any], int, int]
    ) -> dict[str, Any]:
        row, bundle, request, prompt_tokens, required_tokens = item
        started = time.perf_counter()
        try:
            candidates = _request_examples_for_row(builder, bundle, row, request)
            elapsed = time.perf_counter() - started
            records = []
            # Granite token counting and every deterministic validator remain sequential; only
            # the teacher HTTP/decode requests overlap in the candidate arm.
            with builder._prevalidation_lock:
                for candidate in candidates:
                    valid, reason = _validate_candidate(
                        builder, bundle, candidate, row=row
                    )
                    records.append(
                        {
                            "candidate_id": candidate.get("candidate_id"),
                            "question": candidate.get("question"),
                            "answer": candidate.get("answer"),
                            "response_mode": candidate.get("response_mode"),
                            "answer_style": candidate.get("answer_style", "prose"),
                            "citation_coverage": round(
                                citation_coverage(str(candidate.get("answer") or "")), 6
                            ),
                            "deterministic_valid": valid,
                            "deterministic_reason": reason,
                            "support_ledger": candidate.get("support_ledger"),
                        }
                    )
            return {
                "request_id": row["request_id"],
                "policy_lane": row.get("policy_lane", _LEGACY_LANE),
                "pair_id": row.get("pair_id"),
                "relation_mode": row.get("relation_mode"),
                "bundle_id": bundle.bundle_id,
                "document_ids": list(bundle.document_ids),
                "prompt_tokens": prompt_tokens,
                "required_context_tokens": required_tokens,
                "elapsed_seconds": round(elapsed, 3),
                "status": "complete",
                "candidates": records,
            }
        except Exception as exc:
            return {
                "request_id": row["request_id"],
                "policy_lane": row.get("policy_lane", _LEGACY_LANE),
                "pair_id": row.get("pair_id"),
                "relation_mode": row.get("relation_mode"),
                "bundle_id": bundle.bundle_id,
                "document_ids": list(bundle.document_ids),
                "prompt_tokens": prompt_tokens,
                "required_context_tokens": required_tokens,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:500],
                "candidates": [],
            }

    concurrency = int(expected_profile["max_num_seqs"])
    wall_started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(execute, item) for item in prepared]
        for future in as_completed(futures):
            results.append(future.result())
    wall_seconds = time.perf_counter() - wall_started
    results.sort(key=lambda row: row["request_id"])
    candidates = [candidate for row in results for candidate in row["candidates"]]
    valid = sum(candidate["deterministic_valid"] for candidate in candidates)
    failures = sum(row["status"] != "complete" for row in results)
    rejection_counts = Counter(
        candidate["deterministic_reason"]
        for candidate in candidates
        if not candidate["deterministic_valid"]
    )
    lane_metrics: dict[str, dict[str, Any]] = {}
    for lane in sorted({str(row.get("policy_lane") or _LEGACY_LANE) for row in results}):
        lane_requests = [row for row in results if row.get("policy_lane") == lane]
        lane_candidates = [
            candidate for row in lane_requests for candidate in row["candidates"]
        ]
        lane_valid = sum(
            candidate["deterministic_valid"] for candidate in lane_candidates
        )
        lane_metrics[lane] = {
            "request_count": len(lane_requests),
            "failed_requests": sum(
                row["status"] != "complete" for row in lane_requests
            ),
            "generated_candidates": len(lane_candidates),
            "deterministic_valid_candidates": lane_valid,
            "deterministic_acceptance_rate": round(
                lane_valid / len(lane_candidates), 6
            )
            if lane_candidates
            else None,
            "mean_request_elapsed_seconds": round(
                sum(float(row["elapsed_seconds"]) for row in lane_requests)
                / len(lane_requests),
                3,
            )
            if lane_requests
            else None,
        }
    result = {
        "schema_version": 1,
        "completed_at": datetime.now(UTC).isoformat(),
        "arm": arm,
        "server_profile": profile_name,
        "plan_path": str(plan_path),
        "plan_digest": plan["plan_digest"],
        "design": plan.get("design", "matched_infrastructure_ab_v1"),
        "policy_fingerprint": plan["policy_fingerprint"],
        "generation_protocol_unchanged": plan.get("schema_version") == 1,
        "factorial_policy_lanes_isolated": plan.get("schema_version") == 2,
        "actual_server": actual_profile,
        "concurrency": concurrency,
        "wall_seconds": round(wall_seconds, 3),
        "request_count": len(results),
        "failed_requests": failures,
        "generated_candidates": len(candidates),
        "deterministic_valid_candidates": valid,
        "deterministic_acceptance_rate": round(valid / len(candidates), 6)
        if candidates
        else None,
        "candidates_per_hour": round(len(candidates) * 3600 / wall_seconds, 3)
        if wall_seconds
        else None,
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "policy_lane_metrics": lane_metrics,
        "requests": results,
        "semantic_review_status": "pending_blinded_matched_audit",
    }
    destination = _output_dir(settings) / (
        f"result_{arm}_{plan['plan_digest'][:12]}_"
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    write_json_atomic(destination, result)
    summary = {key: value for key, value in result.items() if key != "requests"}
    return {**summary, "path": str(destination)}


def compare_teacher_ab(
    settings: Settings, baseline_path: Path, candidate_path: Path
) -> dict[str, Any]:
    baseline = json.loads(baseline_path.resolve().read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.resolve().read_text(encoding="utf-8"))
    if baseline.get("arm") != "baseline" or candidate.get("arm") != "candidate":
        raise ValueError("A/B comparison requires baseline and candidate result files")
    if baseline.get("plan_digest") != candidate.get("plan_digest"):
        raise ValueError("A/B result files were not generated from the same frozen plan")
    base_rate = float(baseline.get("deterministic_acceptance_rate") or 0.0)
    candidate_rate = float(candidate.get("deterministic_acceptance_rate") or 0.0)
    tolerance = float(settings.get("quality_gates.maximum_material_metric_regression"))
    baseline_lanes = baseline.get("policy_lane_metrics") or {}
    candidate_lanes = candidate.get("policy_lane_metrics") or {}
    lane_comparison: dict[str, dict[str, Any]] = {}
    for lane in sorted(set(baseline_lanes) | set(candidate_lanes)):
        base_lane = baseline_lanes.get(lane, {})
        candidate_lane = candidate_lanes.get(lane, {})
        base_lane_rate = base_lane.get("deterministic_acceptance_rate")
        candidate_lane_rate = candidate_lane.get("deterministic_acceptance_rate")
        lane_comparison[lane] = {
            "baseline": base_lane,
            "candidate": candidate_lane,
            "acceptance_rate_change": round(
                float(candidate_lane_rate) - float(base_lane_rate), 6
            )
            if base_lane_rate is not None and candidate_lane_rate is not None
            else None,
            "deterministic_noninferiority": (
                float(candidate_lane_rate) + tolerance >= float(base_lane_rate)
            )
            if base_lane_rate is not None and candidate_lane_rate is not None
            else None,
        }
    comparison = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "plan_digest": baseline["plan_digest"],
        "design": baseline.get("design", "matched_infrastructure_ab_v1"),
        "baseline_path": str(baseline_path.resolve()),
        "candidate_path": str(candidate_path.resolve()),
        "wall_time_speedup": round(
            float(baseline["wall_seconds"]) / float(candidate["wall_seconds"]), 4
        ),
        "baseline_candidates_per_hour": baseline.get("candidates_per_hour"),
        "candidate_candidates_per_hour": candidate.get("candidates_per_hour"),
        "baseline_acceptance_rate": base_rate,
        "candidate_acceptance_rate": candidate_rate,
        "acceptance_rate_change": round(candidate_rate - base_rate, 6),
        "maximum_material_metric_regression": tolerance,
        "deterministic_noninferiority": candidate_rate + tolerance >= base_rate,
        "failed_requests": {
            "baseline": baseline.get("failed_requests"),
            "candidate": candidate.get("failed_requests"),
        },
        "policy_lane_comparison": lane_comparison,
        "factorial_interpretation": {
            "infrastructure_effect": "candidate versus baseline within each policy lane",
            "policy_effect": "refined versus legacy within each infrastructure arm",
            "semantic_status": "pending_blinded_review_of_all_multisource_outputs",
        },
        "decision": "pending_blinded_semantic_audit",
    }
    destination = _output_dir(settings) / (
        f"comparison_{baseline['plan_digest'][:12]}_"
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    write_json_atomic(destination, comparison)
    return {**comparison, "path": str(destination)}
