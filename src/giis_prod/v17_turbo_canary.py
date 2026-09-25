"""Plan and run a configuration-sized, independently bound V17-Turbo canary."""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
import fcntl
import hashlib
import json
import os
import random
from pathlib import Path
import tomllib
from typing import Any
from urllib.parse import urlparse

from .config import Settings
from .io import sha256_file
from .teacher import DatasetAuditJournal
from .v17_turbo_builder import V17TurboDatasetBuilder
from .v17_turbo_contract import (
    multi_source_bundle_allocation, validate_heldout_benchmark_policy,
    validate_retrieval_index_policy,
)
from .v17_turbo_qualification import run_v17_turbo_qualification

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "config/settings_v17_turbo.toml"
PARTITION_MANIFEST = ROOT / "config/v17_turbo_canary_partitions.json"
OUTPUT_ROOT = ROOT / "artifacts/v17_turbo"


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_state(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _protected_output(relative: str | Path) -> Path:
    target = (ROOT / relative).resolve()
    if target == OUTPUT_ROOT or not target.is_relative_to(OUTPUT_ROOT):
        raise ValueError("V17-Turbo output must be strictly below artifacts/v17_turbo")
    return target


def _validate_teacher_endpoint(configuration: dict[str, Any]) -> None:
    endpoint = urlparse(str(configuration["teacher"]["base_url"]))
    server = configuration["teacher_server"]
    if endpoint.scheme not in {"http", "https"} or endpoint.hostname is None or endpoint.port is None:
        raise ValueError("teacher.base_url must contain an explicit HTTP(S) host and port")
    if endpoint.hostname != str(server["host"]) or endpoint.port != int(server["port"]):
        raise ValueError(
            "Teacher endpoint/profile mismatch: teacher.base_url and teacher_server host/port "
            "must identify the same server"
        )


def _vector_payload_digest(root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for path in sorted(
        value for value in root.rglob("*")
        if value.is_file() and value.name != "chroma.sqlite3"
    ):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(sha256_file(path).encode())
        digest.update(b"\0")
        count += 1
    return digest.hexdigest(), count


def _validate_retrieval_index(configuration: dict[str, Any]) -> None:
    settings = Settings(root=ROOT, data=configuration)
    active_path = settings.path("paths.active_index")
    binding_path = settings.path("paths.index_binding")
    if not active_path.is_file() or not binding_path.is_file():
        raise ValueError("V17-Turbo active-index and binding manifests are required")
    active = _read(active_path)
    binding = _read(binding_path)
    validate_retrieval_index_policy(configuration["v17_turbo_retrieval_index"], binding)
    index_root = (ROOT / "artifacts/v17_turbo/index").resolve()
    chroma = Path(str(active.get("chroma_dir") or "")).resolve()
    lexical = Path(str(active.get("lexical_db") or "")).resolve()
    bound_manifest = Path(str(active.get("binding_manifest") or "")).resolve()
    if not chroma.is_relative_to(index_root) or not lexical.is_relative_to(index_root):
        raise ValueError("V17-Turbo retrieval artifacts must remain inside its private index tree")
    if bound_manifest != binding_path.resolve() or active.get("runtime_dependency_on_v17_index") is not False:
        raise ValueError("V17-Turbo active index is not bound to its private lineage manifest")
    if not chroma.is_dir() or not lexical.is_file():
        raise ValueError("V17-Turbo dense or lexical index artifact is missing")
    documents = _jsonl(settings.path("paths.document_registry"))
    chunks = _jsonl(settings.path("paths.chunk_file"))
    if len(documents) != int(binding["document_count"]):
        raise ValueError("V17-Turbo index binding document count drifted")
    if len(chunks) != int(binding["chunk_count"]):
        raise ValueError("V17-Turbo index binding chunk count drifted")
    if sha256_file(settings.path("paths.document_registry")) != binding["document_registry_sha256"]:
        raise ValueError("V17-Turbo index binding document registry drifted")
    if sha256_file(settings.path("paths.chunk_file")) != binding["chunk_registry_sha256"]:
        raise ValueError("V17-Turbo index binding chunk registry drifted")
    if sha256_file(lexical) != binding["lexical_db_sha256"]:
        raise ValueError("V17-Turbo lexical index digest drifted")
    payload_digest, payload_count = _vector_payload_digest(chroma)
    if (payload_digest != binding["chroma_vector_payload_sha256"]
            or payload_count != binding["chroma_vector_file_count"]):
        raise ValueError("V17-Turbo immutable Chroma vector payload drifted")
    for key in ("corpus_version", "chunk_count", "embedding_model", "collection_name"):
        if active.get(key) != binding.get(key):
            raise ValueError(f"V17-Turbo active index differs from binding: {key}")


def _manifest() -> dict[str, Any]:
    with MANIFEST.open("rb") as handle:
        value = tomllib.load(handle)
    if value["v17_turbo"]["policy_name"] != "V17-Turbo" or value["v17_turbo"]["version"] != "17-Turbo":
        raise ValueError("Canonical V17-Turbo policy configuration required")
    if value["v17_turbo"]["training_release_enabled"]:
        raise ValueError("V17-Turbo canary cannot create a training release")
    _validate_teacher_endpoint(value)
    validate_heldout_benchmark_policy(value["v17_turbo_heldout_benchmark"])
    _validate_retrieval_index(value)
    return value


def _base_settings(out: Path, configuration: dict[str, Any]) -> Settings:
    data = copy.deepcopy(configuration)
    data["paths"]["sft_dir"] = str((out / "work").resolve())
    return Settings(root=ROOT, data=data)


def _source_paths() -> list[str]:
    # Dataset generation binds the frozen paper split and corpus/index lineage. Final
    # held-out references are a later training/evaluation gate and may still be under review.
    return [
        "artifacts/v17_turbo/splits/heldout_manifest.json",
        "artifacts/v17_turbo/corpus/documents.jsonl",
        "artifacts/v17_turbo/corpus/chunks.jsonl",
        "artifacts/v17_turbo/index/active_index.json",
        "artifacts/v17_turbo/index/index_binding.json",
        "config/v17_turbo_canary_partitions.json",
        "config/settings_v17_turbo.toml",
        "V17_TURBO_POLICY.md",
        "V17_TURBO_PROTOCOL.md",
        "src/giis_prod/teacher.py",
        "src/giis_prod/config.py",
        "src/giis_prod/io.py",
        "src/giis_prod/benchmark_freeze.py",
        "src/giis_prod/heldout_authoring.py",
        "src/giis_prod/heldout_oss_review.py",
        "scripts/launch_oss120b_reviewer.sh",
        "scripts/run_v17_turbo_heldout_review.sh",
        "scripts/run_v17_turbo_heldout_pipeline.sh",
        "src/giis_prod/candidate_validation.py",
        "src/giis_prod/citations.py",
        "src/giis_prod/creative_audit.py",
        "src/giis_prod/teacher_ab.py",
        "src/giis_prod/v17_turbo_contract.py",
        "src/giis_prod/v17_turbo_layer.py",
        "src/giis_prod/v17_turbo_builder.py",
        "src/giis_prod/v17_turbo_canary.py",
        "src/giis_prod/v17_turbo_qualification.py",
    ]

def _source_bindings() -> dict[str, str]:
    return {path: sha256_file(ROOT / path) for path in _source_paths()}


def _partitions(configuration: dict[str, Any]) -> dict[str, list[str]]:
    """Create or validate the exact train/validation boundary from the frozen held-out split."""
    policy = configuration["v17_turbo"]
    corpus_count = int(policy["corpus_documents"])
    heldout_count = int(policy["frozen_heldout_documents"])
    generation_count = int(policy["generation_documents"])
    train_count = int(policy["train_documents"])
    validation_count = int(policy["validation_documents"])
    if corpus_count != heldout_count + generation_count:
        raise ValueError("V17-Turbo corpus, held-out, and generation counts are inconsistent")
    if generation_count != train_count + validation_count:
        raise ValueError("V17-Turbo generation, train, and validation counts are inconsistent")

    settings = Settings(root=ROOT, data=configuration)
    heldout_path = settings.path("paths.heldout_manifest")
    registry_path = settings.path("paths.document_registry")
    if not heldout_path.is_file() or not registry_path.is_file():
        raise RuntimeError(
            "Freeze the held-out split and corpus registry before creating a V17-Turbo plan"
        )
    heldout = _read(heldout_path)
    excluded = {str(value) for value in heldout["heldout_document_ids"]}
    eligible = sorted({str(value) for value in heldout["sft_document_ids"]})
    selected = excluded | set(eligible)
    registry = {str(row["document_id"]) for row in _jsonl(registry_path)}
    if (
        len(selected) != corpus_count
        or len(excluded) != heldout_count
        or len(eligible) != generation_count
        or excluded & set(eligible)
        or not selected <= registry
    ):
        raise ValueError("Frozen V17-Turbo selection or held-out count differs from configuration")

    if not PARTITION_MANIFEST.exists():
        shuffled = list(eligible)
        random.Random(int(configuration["teacher"]["seed"])).shuffle(shuffled)
        manifest = {
            "schema_version": 2,
            "policy": "V17-Turbo",
            "contract_revision": int(policy["contract_revision"]),
            "seed": int(configuration["teacher"]["seed"]),
            "corpus_documents": corpus_count,
            "heldout_documents": heldout_count,
            "generation_documents": generation_count,
            "train_documents": train_count,
            "validation_documents": validation_count,
            "heldout_selection_digest": str(heldout["selection_digest"]),
            "train_document_ids": sorted(shuffled[:train_count]),
            "validation_document_ids": sorted(shuffled[train_count:]),
        }
        manifest["partition_digest"] = _digest(manifest)
        _write_new(PARTITION_MANIFEST, manifest)
    else:
        manifest = _read(PARTITION_MANIFEST)

    result = {
        "train": sorted({str(value) for value in manifest.get("train_document_ids", [])}),
        "validation": sorted(
            {str(value) for value in manifest.get("validation_document_ids", [])}
        ),
    }
    observed = set(result["train"]) | set(result["validation"])
    if (
        int(manifest.get("corpus_documents", -1)) != corpus_count
        or int(manifest.get("heldout_documents", -1)) != heldout_count
        or int(manifest.get("generation_documents", -1)) != generation_count
        or len(result["train"]) != train_count
        or len(result["validation"]) != validation_count
        or set(result["train"]) & set(result["validation"])
        or observed != set(eligible)
        or observed & excluded
    ):
        raise ValueError("Frozen V17-Turbo partition boundary changed or is inconsistent")
    return result

def create_plan(relative: str | Path) -> dict[str, Any]:
    out = _protected_output(relative)
    out.mkdir(parents=True, exist_ok=False)
    configuration = _manifest()
    partitions = _partitions(configuration)
    policy = configuration["v17_turbo"]
    plan = {
        "version": "17-Turbo",
        "contract_revision": int(policy["contract_revision"]),
        "kind": (
            f"v17_turbo_{int(policy['corpus_documents'])}_corpus_"
            f"{int(policy['generation_documents'])}_generation_canary"
        ),
        "created_at": datetime.now(UTC).isoformat(),
        "training_release": False,
        "partitions": partitions,
        "configuration": configuration,
        "bindings": _source_bindings(),
    }
    plan["plan_digest"] = _digest(plan)
    _write_new(out / "plan.json", plan)
    print(
        json.dumps(
            {
                "plan": str(out / "plan.json"),
                "documents": sum(len(rows) for rows in plan["partitions"].values()),
                "plan_digest": plan["plan_digest"],
            }
        ),
        flush=True,
    )
    return plan


def load_plan(relative: str | Path) -> tuple[Path, dict[str, Any]]:
    out = _protected_output(relative)
    plan = _read(out / "plan.json")
    unsigned = {key: value for key, value in plan.items() if key != "plan_digest"}
    if _digest(unsigned) != plan["plan_digest"]:
        raise ValueError("Full-canary plan integrity failure")
    if plan["partitions"] != _partitions(plan["configuration"]):
        raise ValueError("Full-canary partitions changed")
    if plan["configuration"] != _manifest():
        raise ValueError("Full-canary configuration changed")
    if plan["bindings"] != _source_bindings():
        raise ValueError("Full-canary source binding changed")
    return out, plan


def _candidate_lookup(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    if not path.exists():
        return lookup
    for row in _jsonl(path):
        candidate = row["candidate"]
        answer = str(candidate.get("answer") or "").strip()
        key = (
            str(row["bundle_id"]),
            str(candidate.get("question") or "").strip(),
            answer,
        )
        lookup[key] = row
    return lookup


def _materialize_results(
    out: Path,
    plan: dict[str, Any],
    accepted_by_partition: dict[str, list[dict[str, Any]]],
    rejected_by_partition: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    journal = out / "raw_candidates.jsonl"
    lookup = _candidate_lookup(journal)
    result_dir = out / "passing_candidates"
    result_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    kinds: dict[str, int] = {}
    expected_result_files: set[str] = set()
    for partition in ("train", "validation"):
        for record in accepted_by_partition[partition]:
            messages = record["messages"]
            question = str(messages[-2]["content"]).split("\n\nQuestion: ", 1)[-1].strip()
            answer = str(messages[-1]["content"]).strip()
            key = (str(record["bundle_id"]), question, answer)
            raw = lookup.get(key)
            if raw is None:
                raise RuntimeError(
                    "Accepted candidate cannot be rebound to the raw V17-Turbo journal: "
                    + record["example_id"]
                )
            payload = {
                "example_id": record["example_id"],
                "plan_digest": plan["plan_digest"],
                "partition": partition,
                "dataset_kind": record["dataset_kind"],
                "document_ids": record["document_ids"],
                "bundle_id": record["bundle_id"],
                "candidate": raw["candidate"],
                "student_messages": record["messages"],
                "token_count": record["token_count"],
                "v17_turbo_python_passed": True,
                "semantic_review": None,
                "training_eligible": False,
            }
            target = result_dir / f"{record['example_id']}.json"
            expected_result_files.add(target.name)
            if target.exists():
                if _read(target) != payload:
                    raise RuntimeError(f"Existing candidate result changed: {target}")
            else:
                _write_new(target, payload)
            for document_id in record["document_ids"]:
                counts[document_id] = counts.get(document_id, 0) + (
                    1 if record["dataset_kind"] == "core_single_source" else 0
                )
            kinds[record["dataset_kind"]] = kinds.get(record["dataset_kind"], 0) + 1

    # passing_candidates is a materialized view of the current accepted set. A resumed
    # multi-source wave can change that set under the same plan, so remove derived residue.
    for existing in sorted(result_dir.glob("*.json")):
        if existing.name not in expected_result_files:
            existing.unlink()
    observed_result_files = {path.name for path in result_dir.glob("*.json")}
    if observed_result_files != expected_result_files:
        raise RuntimeError("Materialized passing-candidate view does not match the accepted set")

    summary = {
        "plan_digest": plan["plan_digest"],
        "completed_at": datetime.now(UTC).isoformat(),
        "documents": sum(len(rows) for rows in plan["partitions"].values()),
        "core_counts_by_document": dict(sorted(counts.items())),
        "accepted_by_kind": dict(sorted(kinds.items())),
        "rejected_by_partition": {
            name: len(rejected_by_partition[name]) for name in ("train", "validation")
        },
        "python_passing_candidates": sum(kinds.values()),
        "semantic_review_status": "pending_stratified_review_sampling",
        "training_release": False,
        "source_integrity_matches": _source_bindings() == plan["bindings"],
    }
    _write_state(out / "progress.json", summary)
    return summary


def _write_execution_progress(
    out: Path,
    plan: dict[str, Any],
    partition: str,
    *,
    completed_documents: list[str],
    total_documents: int,
    status: str,
    failed_documents: list[str] | None = None,
) -> None:
    path = out / "execution_progress.json"
    progress = (
        _read(path)
        if path.exists()
        else {
            "plan_digest": plan["plan_digest"],
            "partitions": {},
        }
    )
    if progress.get("plan_digest") != plan["plan_digest"]:
        raise RuntimeError("Execution progress belongs to a different plan")
    progress["updated_at"] = datetime.now(UTC).isoformat()
    progress["partitions"][partition] = {
        "status": status,
        "completed_documents": sorted(completed_documents),
        "completed_count": len(completed_documents),
        "total_documents": total_documents,
        "failed_documents": sorted(failed_documents or []),
    }
    _write_state(path, progress)


def _checkpoint_core_partition(
    out: Path,
    plan: dict[str, Any],
    builder: V17TurboDatasetBuilder,
    partition: str,
    document_ids: list[str],
    chunks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Continuously refill worker slots while each paper keeps its own checkpoint."""
    ordered = sorted(document_ids)
    concurrency = int(builder.teacher.get("generation_concurrency", 1))
    completed: list[str] = []
    failures: list[tuple[str, BaseException]] = []
    _write_execution_progress(
        out,
        plan,
        partition,
        completed_documents=completed,
        total_documents=len(ordered),
        status="running_core",
    )
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                builder._records_for_partition,
                partition,
                {document_id},
                chunks,
                checkpoint_digest=plan["plan_digest"],
                multi_source_fraction=0.0,
            ): document_id
            for document_id in ordered
        }
        for future in as_completed(futures):
            document_id = futures[future]
            try:
                future.result()
            except BaseException as exc:
                failures.append((document_id, exc))
            else:
                completed.append(document_id)
            _write_execution_progress(
                out,
                plan,
                partition,
                completed_documents=completed,
                total_documents=len(ordered),
                status="failed_core" if failures else "running_core",
                failed_documents=[value for value, _ in failures],
            )
    if failures:
        document_id, failure = sorted(failures, key=lambda row: row[0])[0]
        raise RuntimeError(
            f"Core generation completed all schedulable papers, but document {document_id} "
            "exhausted its request attempts. Successful documents were checkpointed and the "
            "plan can resume only the failed paper or papers."
        ) from failure

    # Reload all per-paper checkpoints in stable partition order. This performs the
    # inherited cross-paper duplicate and checkpoint-integrity checks without HTTP work.
    records = builder._records_for_partition(
        partition,
        set(ordered),
        chunks,
        checkpoint_digest=plan["plan_digest"],
        multi_source_fraction=0.0,
    )
    _write_execution_progress(
        out,
        plan,
        partition,
        completed_documents=ordered,
        total_documents=len(ordered),
        status="core_complete",
    )
    return records


def run(relative: str | Path) -> dict[str, Any]:
    out, plan = load_plan(relative)
    lock_handle = (out / "run.lock").open("a", encoding="utf-8")
    fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    settings = _base_settings(out, plan["configuration"])
    builder = V17TurboDatasetBuilder(settings)
    builder.enable_raw_candidate_journal(out / "raw_candidates.jsonl")
    builder.audit_journal = DatasetAuditJournal(
        out / "audit" / "dataset_audit.jsonl",
        "v17_turbo_" + plan["plan_digest"][:12],
    )
    chunks = _jsonl(settings.path("paths.chunk_file"))
    accepted: dict[str, list[dict[str, Any]]] = {}
    rejected: dict[str, list[dict[str, Any]]] = {}

    for partition in ("train", "validation"):
        accepted[partition], rejected[partition] = _checkpoint_core_partition(
            out,
            plan,
            builder,
            partition,
            plan["partitions"][partition],
            chunks,
        )

    # Persist the complete core result before optional multi-source generation begins.
    _materialize_results(out, plan, accepted, rejected)

    bundle_allocation = multi_source_bundle_allocation(
        {name: len(plan["partitions"][name]) for name in ("train", "validation")},
        core_candidates_per_document=int(
            plan["configuration"]["teacher"]["initial_candidate_batch_size"]
        ),
        multi_source_fraction=float(
            plan["configuration"]["v17_turbo_multi_source"][
                "fraction_of_planned_core_drafts"
            ]
        ),
        candidates_per_bundle=int(
            plan["configuration"]["v17_turbo_multi_source"]["candidates_per_bundle"]
        ),
    )
    for partition in ("train", "validation"):
        document_count = len(plan["partitions"][partition])
        # TeacherDatasetBuilder's compatibility interface accepts a bundle/document
        # fraction. V17-Turbo converts the frozen candidate-level allocation to that value.
        multi_source_fraction = bundle_allocation[partition] / document_count
        accepted[partition], rejected[partition] = builder._records_for_partition(
            partition,
            set(plan["partitions"][partition]),
            chunks,
            checkpoint_digest=plan["plan_digest"],
            multi_source_fraction=multi_source_fraction,
        )
        _write_execution_progress(
            out,
            plan,
            partition,
            completed_documents=plan["partitions"][partition],
            total_documents=len(plan["partitions"][partition]),
            status="complete_with_multi_source",
        )

    summary = _materialize_results(out, plan, accepted, rejected)
    if not summary["source_integrity_matches"]:
        raise ValueError("Full-canary source integrity changed during generation")
    qualification = run_v17_turbo_qualification(
        settings,
        passing_candidates_dir=out / "passing_candidates",
        output_dir=out / "structured_qualification",
        plan_digest=plan["plan_digest"],
    )
    summary["structured_qualification"] = {
        "source_bundle_count": qualification["source_bundle_count"],
        "case_count": qualification["case_count"],
        "cases_by_style": qualification["cases_by_style"],
        "deterministic_passed": qualification["deterministic_passed"],
        "semantic_review_required": qualification["semantic_review_required"],
    }
    _write_state(out / "progress.json", summary)
    print(json.dumps(summary), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run"):
        command = commands.add_parser(name)
        command.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "plan":
        create_plan(args.out)
    else:
        run(args.out)


if __name__ == "__main__":
    main()
