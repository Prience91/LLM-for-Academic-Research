"""Frozen arithmetic and review-sampling rules for the V17-Turbo canary."""
from __future__ import annotations

from collections import defaultdict, deque
import hashlib
import json
import math
import re
from typing import Any, Iterable

CANARY_DOCUMENTS = 80
HELDOUT_DOCUMENTS = 8
GENERATION_DOCUMENTS = 72
TRAIN_DOCUMENTS = 65
VALIDATION_DOCUMENTS = 7
INITIAL_CORE_CANDIDATES = 4
PREFERRED_CORE_MINIMUM = 3
MAXIMUM_CORE_CANDIDATES = 4
UNDERFILLED_RELEASE_FLOOR = 2
REFILL_CANDIDATES = 2
MAXIMUM_CORE_REQUESTS = 3
MULTI_SOURCE_TO_CORE_DRAFT_FRACTION = 0.35
MULTI_SOURCE_CANDIDATES_PER_BUNDLE = 2
QUALIFICATION_BUNDLES = 8
QUALIFICATION_STYLES = ("table", "comparison")
QUALIFICATION_CANDIDATES = QUALIFICATION_BUNDLES * len(QUALIFICATION_STYLES)
SEMANTIC_REVIEW_CAP = 400
SINGLE_SOURCE_MACHINE_REVIEW_FRACTION = 1.0
MULTI_SOURCE_GOLD_REVIEW_FRACTION = 1.0



RETRIEVAL_INDEX_POLICY = {
    "policy": "v17-turbo",
    "maximum_corpus_documents": 20000,
    "embedding_model": "ibm-granite/granite-embedding-30m-english",
    "dense_index": "chroma",
    "lexical_index": "sqlite_fts5",
    "require_content_addressed_binding": True,
    "require_registry_hashes": True,
    "require_immutable_vector_payload_digest": True,
    "require_runtime_collection_audit": True,
    "runtime_dependency_on_earlier_policy": False,
}


def validate_retrieval_index_policy(
    configuration: dict[str, Any], binding: dict[str, Any]
) -> dict[str, Any]:
    """Validate scalable policy invariants and the exact content-addressed active binding."""
    mismatched = {
        key: (configuration.get(key), expected)
        for key, expected in RETRIEVAL_INDEX_POLICY.items()
        if configuration.get(key) != expected
    }
    if mismatched:
        raise ValueError(f"Invalid V17-Turbo retrieval-index policy: {mismatched}")
    if binding.get("policy") != "v17-turbo":
        raise ValueError("V17-Turbo index binding has the wrong policy identity")
    if binding.get("status") != "active" or binding.get("schema_version") != 1:
        raise ValueError("V17-Turbo index binding must be schema 1 and active")
    document_count = int(binding.get("document_count") or 0)
    chunk_count = int(binding.get("chunk_count") or 0)
    if not 1 <= document_count <= RETRIEVAL_INDEX_POLICY["maximum_corpus_documents"]:
        raise ValueError("V17-Turbo index binding document count is outside the supported scale")
    if chunk_count < document_count:
        raise ValueError("V17-Turbo index binding chunk count cannot be smaller than document count")
    if binding.get("embedding_model") != RETRIEVAL_INDEX_POLICY["embedding_model"]:
        raise ValueError("V17-Turbo index binding uses the wrong embedding model")
    if binding.get("runtime_dependency_on_v17_index") is not False:
        raise ValueError("V17-Turbo index binding retains an earlier-policy runtime dependency")
    for key in (
        "corpus_version", "document_registry_sha256", "chunk_registry_sha256",
        "chroma_vector_payload_sha256", "lexical_db_sha256", "source_active_index_sha256",
    ):
        value = str(binding.get(key) or "")
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"V17-Turbo index binding lacks a valid digest: {key}")
    if int(binding.get("chroma_vector_file_count") or 0) < 1:
        raise ValueError("V17-Turbo Chroma binding contains no immutable vector files")
    if binding.get("chroma_sqlite_integrity") != "runtime_collection_identity_and_count_audit":
        raise ValueError("V17-Turbo Chroma SQLite must use runtime semantic integrity checks")
    return {**RETRIEVAL_INDEX_POLICY, "document_count": document_count, "chunk_count": chunk_count,
            "corpus_version": binding["corpus_version"]}


HELDOUT_REFERENCE_GENERATOR = "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4"
HELDOUT_AUTHORING_METHOD = "qwen_122b_obligation_mapped_from_frozen_heldout_evidence"
HELDOUT_REVIEWER_ROUTING = {
    "single_source": "GPT-OSS-120B",
    "multi_source": "Codex gold",
    "table": "Codex gold",
    "comparison": "Codex gold",
}


def validate_heldout_benchmark_policy(configuration: dict[str, Any]) -> dict[str, Any]:
    """Validate the locked final-benchmark author and lane-specific reviewer ownership."""
    expected = {
        "authoring_model": HELDOUT_REFERENCE_GENERATOR,
        "authoring_method": HELDOUT_AUTHORING_METHOD,
        "single_source_reviewer": HELDOUT_REVIEWER_ROUTING["single_source"],
        "multi_source_reviewer": HELDOUT_REVIEWER_ROUTING["multi_source"],
        "table_reviewer": HELDOUT_REVIEWER_ROUTING["table"],
        "comparison_reviewer": HELDOUT_REVIEWER_ROUTING["comparison"],
        "reference_answers_hidden_from_student": True,
        "obligation_scoring_required": True,
        "training_feedback_prohibited": True,
        "one_shot_final_evaluation": True,
    }
    mismatched = {key: (configuration.get(key), value) for key, value in expected.items()
                  if configuration.get(key) != value}
    if mismatched:
        raise ValueError(f"Invalid V17-Turbo held-out benchmark policy: {mismatched}")
    return {
        "authoring_model": HELDOUT_REFERENCE_GENERATOR,
        "authoring_method": HELDOUT_AUTHORING_METHOD,
        "reviewer_routing": dict(HELDOUT_REVIEWER_ROUTING),
        "reference_answers_hidden_from_student": True,
        "obligation_scoring_required": True,
        "training_feedback_prohibited": True,
        "one_shot_final_evaluation": True,
    }


def heldout_reviewer_for(candidate_type: str, configuration: dict[str, Any]) -> str:
    """Return the only reviewer authorized for a held-out candidate type."""
    validate_heldout_benchmark_policy(configuration)
    try:
        return HELDOUT_REVIEWER_ROUTING[candidate_type]
    except KeyError as exc:
        raise ValueError(f"Unknown held-out benchmark candidate type: {candidate_type}") from exc

def semantic_sample_size(population: int, fraction: float, cap: int = 400) -> int:
    """Return min(cap, ceil(fraction * Python-passing population))."""
    if population < 0 or cap < 1 or not 0.0 <= fraction <= 1.0:
        raise ValueError("Invalid semantic sample parameters")
    return min(cap, math.ceil(population * fraction))


def multi_source_bundle_allocation(
    partition_documents: dict[str, int],
    *,
    core_candidates_per_document: int = INITIAL_CORE_CANDIDATES,
    multi_source_fraction: float = MULTI_SOURCE_TO_CORE_DRAFT_FRACTION,
    candidates_per_bundle: int = MULTI_SOURCE_CANDIDATES_PER_BUNDLE,
) -> dict[str, int]:
    """Allocate an exact candidate-level multi-source target across partitions.

    Bundle counts use largest remainders, so the configured fraction is applied to
    planned core drafts rather than to document-bundle counts.
    """
    if not partition_documents or any(value < 0 for value in partition_documents.values()):
        raise ValueError("Partition document counts must be non-negative")
    total_documents = sum(partition_documents.values())
    if total_documents < 2 or core_candidates_per_document < 1 or candidates_per_bundle < 1:
        raise ValueError("Insufficient documents or invalid candidate allocation")
    target_candidates = round(
        total_documents * core_candidates_per_document * multi_source_fraction
    )
    target_bundles = round(target_candidates / candidates_per_bundle)
    raw = {
        name: target_bundles * count / total_documents
        for name, count in partition_documents.items()
    }
    allocation = {name: math.floor(value) for name, value in raw.items()}
    remaining = target_bundles - sum(allocation.values())
    order = sorted(raw, key=lambda name: (-(raw[name] - allocation[name]), name))
    for name in order[:remaining]:
        allocation[name] += 1
    return allocation


def _word_count(answer: str) -> int:
    return len(re.findall(r"\b\w+[\w'-]*\b", answer))


def _band(value: int, boundaries: tuple[int, int]) -> str:
    return "low" if value <= boundaries[0] else "mid" if value <= boundaries[1] else "high"


def _stable_order(row: dict[str, Any], seed: int) -> str:
    identity = str(row.get("example_id") or row.get("candidate_id") or "")
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{seed}:{identity}:{payload}".encode()).hexdigest()


def qualification_source_selection(
    candidates: Iterable[dict[str, Any]],
    *,
    seed: int,
    bundle_count: int = QUALIFICATION_BUNDLES,
) -> list[dict[str, Any]]:
    """Choose one Python-passing prose source from each of distinct multi-source bundles."""
    if bundle_count < 1:
        raise ValueError("Qualification bundle count must be positive")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        item = row.get("candidate", row)
        if (
            str(row.get("dataset_kind")) == "multi_source_extra"
            and bool(row.get("v17_turbo_python_passed"))
            and str(item.get("answer_style")) == "prose"
            and len({str(value) for value in row.get("document_ids", [])}) == 2
        ):
            grouped[str(row.get("bundle_id") or "")].append(row)
    grouped.pop("", None)
    if len(grouped) < bundle_count:
        raise ValueError(
            f"Qualification requires {bundle_count} distinct passing multi-source bundles; "
            f"found {len(grouped)}"
        )
    bundle_ids = sorted(
        grouped,
        key=lambda value: hashlib.sha256(f"{seed}:qualification:{value}".encode()).hexdigest(),
    )[:bundle_count]
    return [
        sorted(grouped[bundle_id], key=lambda row: _stable_order(row, seed))[0]
        for bundle_id in bundle_ids
    ]


def qualification_cases(
    candidates: Iterable[dict[str, Any]], *, seed: int,
    bundle_count: int = QUALIFICATION_BUNDLES,
) -> list[dict[str, Any]]:
    """Expand the configured source-bundle sample into paired rendering cases."""
    cases: list[dict[str, Any]] = []
    for source in qualification_source_selection(
        candidates, seed=seed, bundle_count=bundle_count
    ):
        for style in QUALIFICATION_STYLES:
            cases.append(
                {
                    "case_id": f"qualification_{source['bundle_id']}_{style}",
                    "answer_style": style,
                    "source_example_id": str(source["example_id"]),
                    "bundle_id": str(source["bundle_id"]),
                    "document_ids": sorted(str(value) for value in source["document_ids"]),
                    "source": source,
                }
            )
    return sorted(cases, key=lambda row: str(row["case_id"]))


def stratified_semantic_sample(
    candidates: Iterable[dict[str, Any]],
    *,
    fraction: float,
    seed: int,
    cap: int = SEMANTIC_REVIEW_CAP,
) -> list[dict[str, Any]]:
    """Select a reproducible, diversity-first semantic sample.

    The first pass uses at most one candidate per document scope/bundle. Within
    that constraint, round-robin strata cover partition, response capacity,
    obligation count, and answer length before a second candidate from a scope.
    """
    rows = list(candidates)
    target = semantic_sample_size(len(rows), fraction, cap)
    if target == 0:
        return []

    def candidate(row: dict[str, Any]) -> dict[str, Any]:
        return row.get("candidate", row)

    lengths = sorted(_word_count(str(candidate(row).get("answer") or "")) for row in rows)
    lower = lengths[(len(lengths) - 1) // 3]
    upper = lengths[(2 * (len(lengths) - 1)) // 3]

    strata: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        item = candidate(row)
        obligation_count = len(item.get("coverage_obligations") or [])
        key = (
            str(row.get("partition") or "unknown"),
            str(item.get("response_mode") or row.get("response_mode") or "standard"),
            _band(obligation_count, (2, 4)),
            _band(_word_count(str(item.get("answer") or "")), (lower, upper)),
        )
        strata[key].append(row)
    queues = {
        key: deque(sorted(group, key=lambda row: _stable_order(row, seed)))
        for key, group in strata.items()
    }
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    used_scopes: set[tuple[str, ...]] = set()

    def identity(row: dict[str, Any]) -> str:
        return str(row.get("example_id") or row.get("candidate_id") or _stable_order(row, seed))

    def scope(row: dict[str, Any]) -> tuple[str, ...]:
        documents = row.get("document_ids") or []
        return tuple(sorted(str(value) for value in documents)) or (str(row.get("bundle_id") or identity(row)),)

    for unique_only in (True, False):
        while len(selected) < target and any(queues.values()):
            progressed = False
            for key in sorted(queues):
                queue = queues[key]
                for _ in range(len(queue)):
                    row = queue.popleft()
                    row_id = identity(row)
                    row_scope = scope(row)
                    if row_id in selected_ids:
                        continue
                    if unique_only and row_scope in used_scopes:
                        queue.append(row)
                        continue
                    selected.append(row)
                    selected_ids.add(row_id)
                    used_scopes.add(row_scope)
                    progressed = True
                    break
                if len(selected) == target:
                    break
            if not progressed:
                break
    return selected


def semantic_review_plan(
    single_source: list[dict[str, Any]],
    multi_source: list[dict[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    """Return the frozen V17-Turbo reviewer census and reviewer ownership."""
    all_rows = single_source + multi_source
    structured = [
        row for row in all_rows
        if row.get("candidate", row).get("answer_style") in {"table", "comparison"}
    ]
    structured_ids = {str(row.get("example_id")) for row in structured}
    single_prose = [
        row for row in single_source if str(row.get("example_id")) not in structured_ids
    ]
    multi_prose = [
        row for row in multi_source if str(row.get("example_id")) not in structured_ids
    ]
    return {
        "single_source_machine_census": sorted(
            single_prose, key=lambda row: _stable_order(row, seed)
        ),
        "multi_source_gold_census": sorted(
            multi_prose, key=lambda row: _stable_order(row, seed + 1)
        ),
        "structured_gold_census": sorted(
            structured, key=lambda row: _stable_order(row, seed + 2)
        ),
        "reviewer_ownership": {
            "single_source_machine_census": "GPT-OSS-120B",
            "multi_source_gold_census": "gold",
            "structured_gold_census": "gold",
        },
        "single_source_gold_review_fraction": 0.0,
    }
