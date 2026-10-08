from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# Make the consolidated runner executable from a source checkout after dependencies are installed.
PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from giis_prod.benchmark_freeze import BenchmarkFreezeError, validate_benchmark_freeze
from giis_prod.config import Settings
from giis_prod.dataset_release import (
    DatasetReleaseError,
    dataset_release_path,
    validate_dataset_release,
)
from giis_prod.production_readiness import (
    assert_dataset_generation_ready,
    production_readiness_report,
)


def _load_project_openalex_secret() -> None:
    """Load the project-scoped OpenAlex key without persisting it in project artifacts."""
    name = "GIIS__METADATA__OPENALEX_API_KEY"
    if os.getenv(name):
        return
    path = Path(os.getenv("GIIS_OPENALEX_ENV", str(Path.home() / ".config/giis/openalex.env")))
    if not path.is_file():
        return
    if path.stat().st_mode & 0o077:
        raise PermissionError(f"OpenAlex credential file must be mode 600: {path}")
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("export "):
            line = line[7:].strip()
        if not line.startswith(name + "="):
            continue
        value = line.split("=", 1)[1].strip().strip('"').strip("'")
        if value:
            os.environ[name] = value
        return


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def _resume_value(raw: str | None) -> str | bool | None:
    if raw is None:
        return None
    if raw == "auto":
        return True
    checkpoint = Path(raw).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    return str(checkpoint)


def _redacted_command(command: list[str]) -> list[str]:
    redacted = list(command)
    for index, value in enumerate(redacted[:-1]):
        if value == "--api-key":
            redacted[index + 1] = "***"
    return redacted


def _path_state(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
    }


def _optional_path_state(settings: Settings, key: str) -> dict[str, Any]:
    try:
        return _path_state(settings.path(key))
    except KeyError:
        return {"path": None, "exists": False, "size_bytes": None}


def pipeline_status(settings: Settings) -> dict[str, Any]:
    pdf_dir = settings.path("paths.pdf_dir")
    pdf_count = (
        sum(
            path.is_file() and path.suffix.casefold() == ".pdf"
            for path in pdf_dir.iterdir()
        )
        if pdf_dir.exists()
        else 0
    )
    training_pdf_dir = settings.path("paths.training_pdf_dir")
    training_pdf_count = (
        sum(
            path.is_file() and path.suffix.casefold() == ".pdf"
            for path in training_pdf_dir.iterdir()
        )
        if training_pdf_dir.exists()
        else 0
    )
    metadata = settings.path("paths.metadata_overrides")
    try:
        metadata_review = settings.path("paths.metadata_review_queue")
    except KeyError:
        metadata_review = settings.root / "artifacts/metadata/review_required.jsonl"
    documents = settings.path("paths.document_registry")
    chunks = settings.path("paths.chunk_file")
    active_index = settings.path("paths.active_index")
    sft_dir = settings.path("paths.sft_dir")
    training_selection = settings.path("paths.training_selection_manifest")
    heldout_required = "heldout_manifest" in settings.data.get("paths", {})
    try:
        heldout_manifest = settings.path("paths.heldout_manifest")
    except KeyError:
        heldout_manifest = settings.root / "artifacts/splits/heldout_manifest.json"
    release_path = dataset_release_path(settings)
    try:
        release = validate_dataset_release(settings)
        release_valid = True
        release_error = None
    except DatasetReleaseError as exc:
        release = None
        release_valid = False
        release_error = str(exc)
    adapter = settings.path("models.adapter_output") / "final"
    merged = settings.path("models.merged_output")
    benchmark = settings.path("paths.benchmark_file")
    try:
        benchmark_freeze = validate_benchmark_freeze(settings)
        benchmark_valid = True
        benchmark_error = None
    except (BenchmarkFreezeError, KeyError, OSError, json.JSONDecodeError) as exc:
        benchmark_freeze = None
        benchmark_valid = False
        benchmark_error = str(exc)
    readiness = production_readiness_report(settings)

    if pdf_count == 0:
        next_step = "Add 700 PDFs, then run: python3 -u giis_pipeline.py corpus"
    elif not documents.exists() or not chunks.exists() or not active_index.exists():
        next_step = "python giis_pipeline.py corpus"
    elif training_pdf_count == 0:
        next_step = "python3 -u giis_pipeline.py materialize-training"
    elif heldout_required and not heldout_manifest.exists():
        next_step = "python3 -u giis_pipeline.py plan-heldout"
    elif not release_valid:
        smoke_limit = int(settings.get("teacher.timed_smoke_max_documents"))
        if training_pdf_count <= smoke_limit:
            next_step = (
                "Ensure `python3 -u giis_pipeline.py teacher-serve` is healthy in its "
                "tmux session, then run: python3 -u giis_pipeline.py dataset "
                f"--expected-documents {training_pdf_count}"
            )
        elif not readiness.complete:
            next_step = (
                "Complete and verify every gate in PRODUCTION_READINESS_V15_RC.md before "
                "the default 2,500-paper dataset run"
            )
        else:
            next_step = "python3 -u giis_pipeline.py dataset"
    elif not benchmark_valid:
        next_step = (
            "Freeze and validate benchmark/benchmark.jsonl plus its evidence rubric and manifest "
            "for the held-out papers before training; "
            "then run: python3 -u giis_pipeline.py train"
        )
    elif not adapter.exists():
        next_step = "python3 -u giis_pipeline.py train"
    elif not merged.exists():
        next_step = "python3 -u giis_pipeline.py merge"
    else:
        next_step = "Run evaluation, then start serving with: python giis_pipeline.py serve"

    return {
        "project_root": str(settings.root),
        "pdf_count": pdf_count,
        "training_pdf_count": training_pdf_count,
        "artifacts": {
            "metadata_overrides": _path_state(metadata),
            "metadata_review_queue": _path_state(metadata_review),
            "document_registry": _path_state(documents),
            "chunks": _path_state(chunks),
            "active_index": _path_state(active_index),
            "training_selection": _path_state(training_selection),
            "heldout_manifest": _path_state(heldout_manifest),
            "sft_release": _path_state(release_path),
            "sft_train": _path_state(sft_dir / "train.jsonl"),
            "sft_validation": _path_state(sft_dir / "validation.jsonl"),
            "adapter": _path_state(adapter),
            "merged_model": _path_state(merged),
            "benchmark": _path_state(benchmark),
            "benchmark_reference": _optional_path_state(
                settings, "paths.benchmark_reference"
            ),
            "benchmark_manifest": _optional_path_state(
                settings, "paths.benchmark_manifest"
            ),
        },
        "benchmark_freeze": {
            "valid": benchmark_valid,
            "error": benchmark_error,
            "details": benchmark_freeze,
        },
        "dataset_release": {
            "valid": release_valid,
            "error": release_error,
            "generation_policy_version": (
                release["generation_policy_version"] if release is not None else None
            ),
            "policy_fingerprint": (
                release["policy_fingerprint"] if release is not None else None
            ),
        },
        "production_readiness": readiness.as_dict(),
        "credentials_present": {
            "teacher_api_key": bool(os.getenv("TEACHER_API_KEY")),
            "vllm_api_key": bool(os.getenv("VLLM_API_KEY")),
            "wandb_api_key": bool(os.getenv("WANDB_API_KEY")),
            "hugging_face_cached_login": "not inspected; use `hf auth whoami`",
        },
        "next_step": next_step,
    }


def build_corpus(
    settings: Settings,
    corpus_only: bool = False,
    expected_documents: int | None = None,
) -> dict[str, Any]:
    from giis_prod.corpus_audit import CorpusAuditor
    from giis_prod.indexing import build_indexes
    from giis_prod.ingestion import CorpusBuilder

    output: dict[str, Any] = {
        "ingestion": CorpusBuilder(settings).build(expected_documents=expected_documents)
    }
    auditor = CorpusAuditor(settings)
    output["pre_index_audit"] = auditor.run(check_indexes=False)
    if not corpus_only:
        output["index"] = build_indexes(settings)
        output["audit"] = auditor.run(check_indexes=True)
    return output


def build_dataset(
    settings: Settings, expected_documents: int | None = None
) -> dict[str, Any]:
    # Keep the production lock ahead of builder construction so a default 2,500-paper
    # command cannot load teacher dependencies or mutate selection/SFT artifacts.
    assert_dataset_generation_ready(settings, expected_documents)
    from giis_prod.teacher import TeacherDatasetBuilder

    return TeacherDatasetBuilder(settings).build(expected_documents=expected_documents)


def train(settings: Settings, resume: str | None = None) -> dict[str, Any]:
    from giis_prod.training import train_sft

    return train_sft(settings, _resume_value(resume))


def merge(settings: Settings) -> dict[str, Any]:
    from giis_prod.packaging import merge_adapter

    return merge_adapter(settings)


def _benchmark_case(settings: Settings, case_id: str):
    from giis_prod.evaluation import BenchmarkCase
    from giis_prod.io import read_jsonl

    for row in read_jsonl(settings.path("paths.benchmark_file")):
        if row["case_id"] == case_id:
            return BenchmarkCase.from_dict(row)
    raise KeyError(f"Unknown case_id: {case_id}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Single operator interface for the GIIS production pipeline."
    )
    parser.add_argument("--config", help="Path to settings.toml")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("status", help="Show completed artifacts and the recommended next command")

    commands.add_parser(
        "audit-corpus",
        help="Audit metadata, chunks, split freshness, and active index integrity",
    )

    corpus = commands.add_parser("corpus", help="Parse PDFs and build versioned retrieval indexes")
    corpus.add_argument("--corpus-only", action="store_true", help="Skip index construction")
    corpus.add_argument(
        "--expected-documents",
        type=int,
        help="Require this many accepted unique documents; defaults to the active configuration",
    )

    heldout = commands.add_parser(
        "plan-heldout",
        help="Freeze the configuration-sized paper-level SFT and held-out split",
    )
    heldout.add_argument("--expected-corpus-documents", type=int)
    heldout.add_argument("--heldout-documents", type=int)

    heldout_author = commands.add_parser(
        "author-heldout",
        help="Author resumable obligation-mapped held-out references with Qwen-122B",
    )
    heldout_author.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    heldout_pipeline = commands.add_parser(
        "heldout-pipeline",
        help="Run canonical V17-Turbo held-out authoring and full gold-review queue preparation",
    )
    heldout_pipeline.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Fresh held-out namespace below artifacts/v17_turbo",
    )

    heldout_pipeline.add_argument("--main-run", type=Path, required=True, help="OSS-complete main run bound to split_review_v2")

    commands.add_parser(
        "materialize-training",
        help="Copy the frozen SFT paper partition into Papers_Training with hash checks",
    )

    dataset = commands.add_parser("dataset", help="Generate validated Qwen synthetic SFT data")
    dataset.add_argument(
        "--expected-documents",
        type=int,
        help="Override the configured generation-document target only for an intentional canary",
    )

    teacher_serve = commands.add_parser(
        "teacher-serve",
        help="Start the authenticated local configured teacher server",
    )
    teacher_serve.add_argument(
        "--dry-run",
        action="store_true",
        help="Print a redacted teacher vLLM command",
    )
    teacher_serve.add_argument(
        "--profile",
        choices=("default", "ab-baseline", "ab-candidate"),
        default="default",
        help="Use the default server or one isolated A/B serving profile",
    )

    teacher_ab_plan = commands.add_parser(
        "teacher-ab-plan",
        help="Freeze representative teacher requests outside the SFT release",
    )
    teacher_ab_plan.add_argument(
        "--requests",
        type=int,
        help="Override the configured request count for this frozen A/B plan",
    )

    teacher_ab_run = commands.add_parser(
        "teacher-ab-run",
        help="Replay one frozen teacher A/B arm against its matching server profile",
    )
    teacher_ab_run.add_argument("--arm", choices=("baseline", "candidate"), required=True)
    teacher_ab_run.add_argument("--plan", type=Path, required=True)
    teacher_ab_run.add_argument(
        "--preflight-only",
        action="store_true",
        help="Verify the frozen plan, running profile, and exact token fit without generation",
    )

    teacher_ab_compare = commands.add_parser(
        "teacher-ab-compare",
        help="Compare matched baseline/candidate teacher-serving results",
    )
    teacher_ab_compare.add_argument("--baseline", type=Path, required=True)
    teacher_ab_compare.add_argument("--candidate", type=Path, required=True)

    creative_audit = commands.add_parser(
        "creative-audit",
        help="Run five isolated creative-format cases against the selected 16K/2 teacher",
    )
    creative_audit.add_argument(
        "--release",
        type=Path,
        default=Path("artifacts/sft_challenger_16k2"),
        help="Immutable source release whose evidence bundles and questions are reused",
    )
    creative_audit.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/creative_audit_16k2"),
        help="Fresh output directory; frozen dataset releases are never modified",
    )

    training = commands.add_parser("train", help="Run LoRA supervised fine-tuning")
    training.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        help="Resume the latest checkpoint or provide a checkpoint path",
    )

    commands.add_parser("merge", help="Merge the LoRA adapter into the BF16 baseline")

    commands.add_parser(
        "plan-semantic-audit",
        help="Create the agreed min(400, 10%%) stratified semantic-audit queue",
    )

    serve = commands.add_parser("serve", help="Start the authenticated local vLLM server")
    serve.add_argument("--dry-run", action="store_true", help="Print a redacted vLLM command")

    commands.add_parser("ui", help="Start the local grounded Gradio interface")

    evaluate = commands.add_parser("evaluate", help="Evaluate the actual production path")
    evaluate.add_argument("--output", type=Path, default=Path("benchmark/results/current"))

    stream = commands.add_parser("stream-probe", help="Measure TTFT and decode throughput")
    stream.add_argument("--case-id", required=True)

    load = commands.add_parser("load-probe", help="Run the concurrent-user probe")
    load.add_argument("--case-id", required=True)
    load.add_argument("--users", type=int)

    compare = commands.add_parser("compare", help="Compare candidate and BF16 evaluation runs")
    compare.add_argument("--baseline", required=True, type=Path)
    compare.add_argument("--candidate", required=True, type=Path)

    all_steps = commands.add_parser(
        "build-all",
        help="Run corpus, dataset, training, and BF16 merge sequentially",
    )
    all_steps.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        help="Resume the training stage from latest or specified checkpoint",
    )
    all_steps.add_argument(
        "--expected-documents",
        type=int,
        help="Override the configured training-document target only for an intentional smoke test",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _load_project_openalex_secret()
    settings = Settings.load(args.config)

    # build-all would otherwise mutate corpus artifacts before it reaches build_dataset.
    # The defensive checks in build_dataset and TeacherDatasetBuilder.build remain too.
    if args.command in {"dataset", "build-all"}:
        assert_dataset_generation_ready(settings, args.expected_documents)

    if args.command == "status":
        result = pipeline_status(settings)
    elif args.command == "audit-corpus":
        from giis_prod.corpus_audit import CorpusAuditor

        result = CorpusAuditor(settings).run(check_indexes=True)
    elif args.command == "corpus":
        result = build_corpus(
            settings,
            corpus_only=args.corpus_only,
            expected_documents=args.expected_documents,
        )
    elif args.command == "plan-heldout":
        from giis_prod.splits import plan_heldout_split

        result = plan_heldout_split(
            settings,
            expected_corpus_documents=args.expected_corpus_documents,
            heldout_documents=args.heldout_documents,
        )
    elif args.command == "author-heldout":
        from giis_prod.heldout_authoring import author_heldout

        result = author_heldout(settings, (settings.root / args.output).resolve())
    elif args.command == "heldout-pipeline":
        output = (settings.root / args.output).resolve()
        v17_root = (settings.root / "artifacts/v17_turbo").resolve()
        if not output.is_relative_to(v17_root):
            raise ValueError("Held-out output must remain below artifacts/v17_turbo")
        command = [
            str(settings.root / "scripts/run_v17_turbo_heldout_pipeline_v2.sh"),
            str(output.relative_to(settings.root)),
            str((settings.root / args.main_run).resolve()),
        ]
        completed = subprocess.run(command, cwd=settings.root, check=False)
        if completed.returncode:
            raise RuntimeError(f"Held-out pipeline exited with status {completed.returncode}")
        result = json.loads((output / "status.json").read_text())
    elif args.command == "materialize-training":
        from giis_prod.training_materialization import materialize_training

        result = materialize_training(settings)
    elif args.command == "dataset":
        result = build_dataset(settings, args.expected_documents)
    elif args.command == "teacher-serve":
        from giis_prod.server import serve_teacher_vllm, teacher_vllm_command

        if args.dry_run:
            result = {
                "profile": args.profile,
                "command": _redacted_command(
                    teacher_vllm_command(settings, profile=args.profile)
                ),
            }
        else:
            serve_teacher_vllm(settings, profile=args.profile)
            return
    elif args.command == "teacher-ab-plan":
        from giis_prod.teacher_ab import create_teacher_ab_plan

        result = create_teacher_ab_plan(settings, request_count=args.requests)
    elif args.command == "teacher-ab-run":
        from giis_prod.teacher_ab import run_teacher_ab

        result = run_teacher_ab(
            settings,
            args.plan,
            args.arm,
            preflight_only=args.preflight_only,
        )
    elif args.command == "teacher-ab-compare":
        from giis_prod.teacher_ab import compare_teacher_ab

        result = compare_teacher_ab(settings, args.baseline, args.candidate)
    elif args.command == "creative-audit":
        from giis_prod.creative_audit import run_creative_audit

        result = run_creative_audit(
            settings,
            release_dir=args.release.resolve(),
            output_dir=args.output.resolve(),
        )
    elif args.command == "train":
        result = train(settings, args.resume)
    elif args.command == "merge":
        result = merge(settings)
    elif args.command == "plan-semantic-audit":
        from giis_prod.semantic_audit import plan_semantic_audit

        result = plan_semantic_audit(settings)
    elif args.command == "serve":
        from giis_prod.server import serve_vllm, vllm_command

        if args.dry_run:
            result = {"command": _redacted_command(vllm_command(settings))}
        else:
            serve_vllm(settings)
            return
    elif args.command == "ui":
        from giis_prod.ui import launch_app

        launch_app(settings)
        return
    elif args.command == "evaluate":
        from giis_prod.evaluation import SystemEvaluator

        result = SystemEvaluator(settings).run(
            settings.path("paths.benchmark_file"), args.output.resolve()
        )
    elif args.command == "stream-probe":
        from giis_prod.evaluation import SystemEvaluator

        result = SystemEvaluator(settings).streaming_probe(
            _benchmark_case(settings, args.case_id)
        )
    elif args.command == "load-probe":
        from giis_prod.evaluation import SystemEvaluator, concurrent_probe

        evaluator = SystemEvaluator(settings)
        users = args.users or int(settings.get("quality_gates.expected_concurrent_users"))
        result = concurrent_probe(
            evaluator,
            _benchmark_case(settings, args.case_id),
            users,
        )
    elif args.command == "compare":
        from giis_prod.evaluation import compare_runs

        result = compare_runs(settings, args.baseline.resolve(), args.candidate.resolve())
    else:
        stages: dict[str, Any] = {}
        stages["corpus"] = build_corpus(settings)
        stages["dataset"] = build_dataset(settings, args.expected_documents)
        stages["training"] = train(settings, args.resume)
        stages["merge"] = merge(settings)
        result = stages
    _print(result)


if __name__ == "__main__":
    main()
