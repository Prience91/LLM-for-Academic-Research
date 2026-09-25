from __future__ import annotations

import os
import subprocess
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen

from .config import Settings

_HARMONY_ENCODINGS = {
    "o200k_base.tiktoken": "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d",
    "cl100k_base.tiktoken": "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
}
_HARMONY_ENCODINGS_URL = "https://openaipublic.blob.core.windows.net/encodings"


def vllm_command(settings: Settings) -> list[str]:
    serving = settings.section("serving")
    configured_model = str(settings.get("models.served"))
    possible_path = Path(configured_model)
    model = str((settings.root / possible_path).resolve()) if not possible_path.is_absolute() and (settings.root / possible_path).exists() else configured_model
    api_key = os.getenv("VLLM_API_KEY")
    if not api_key or api_key == "replace-me":
        raise RuntimeError("Set a non-default VLLM_API_KEY before serving")
    command = [
        "vllm",
        "serve",
        model,
        "--served-model-name",
        configured_model,
        "--host",
        str(serving["host"]),
        "--port",
        str(serving["vllm_port"]),
        "--api-key",
        api_key,
        "--gpu-memory-utilization",
        str(serving["gpu_memory_utilization"]),
        "--max-model-len",
        str(serving["max_model_len"]),
        "--max-num-seqs",
        str(serving["max_num_seqs"]),
    ]
    if bool(serving["enable_prefix_caching"]):
        command.append("--enable-prefix-caching")
    if bool(serving.get("language_model_only", False)):
        command.append("--language-model-only")
    if str(serving["quantization"]).strip():
        command.extend(["--quantization", str(serving["quantization"])])
    return command


def teacher_server_profile(settings: Settings, profile: str = "default") -> dict[str, object]:
    serving = settings.section("teacher_server")
    if profile == "default":
        return serving
    if profile not in {"ab-baseline", "ab-candidate"}:
        raise ValueError(f"Unknown teacher server profile: {profile}")
    experiment = settings.section("teacher_ab")
    prefix = "baseline" if profile == "ab-baseline" else "candidate"
    serving["max_model_len"] = int(experiment[f"{prefix}_max_model_len"])
    serving["max_num_seqs"] = int(experiment[f"{prefix}_max_num_seqs"])
    return serving


def teacher_vllm_command(settings: Settings, profile: str = "default") -> list[str]:
    serving = teacher_server_profile(settings, profile)
    api_key = os.getenv("TEACHER_API_KEY")
    if not api_key or api_key == "replace-me":
        raise RuntimeError("Set a non-default TEACHER_API_KEY before serving the teacher")
    command = [
        "vllm",
        "serve",
        str(settings.get("models.teacher_source")),
        "--served-model-name",
        str(settings.get("models.teacher")),
        "--host",
        str(serving["host"]),
        "--port",
        str(serving["port"]),
        "--api-key",
        api_key,
        "--gpu-memory-utilization",
        str(serving["gpu_memory_utilization"]),
        "--max-model-len",
        str(serving["max_model_len"]),
        "--max-num-seqs",
        str(serving["max_num_seqs"]),
    ]
    if serving.get("max_num_batched_tokens"):
        command.extend(["--max-num-batched-tokens", str(serving["max_num_batched_tokens"])])
    if str(serving.get("kv_cache_dtype") or "").strip():
        command.extend(["--kv-cache-dtype", str(serving["kv_cache_dtype"])])
    if bool(serving["enable_prefix_caching"]):
        command.append("--enable-prefix-caching")
    if bool(serving.get("enable_chunked_prefill", False)):
        command.append("--enable-chunked-prefill")
    if bool(serving.get("enforce_eager", False)):
        command.append("--enforce-eager")
    if bool(serving.get("disable_log_requests", False)):
        command.append("--no-enable-log-requests")
    if str(serving.get("generation_config") or "").strip():
        command.extend(["--generation-config", str(serving["generation_config"])])
    if str(serving["quantization"]).strip():
        command.extend(["--quantization", str(serving["quantization"])])
    return command


def serve_vllm(settings: Settings) -> None:
    _run_vllm(vllm_command(settings), "production")


def serve_teacher_vllm(settings: Settings, profile: str = "default") -> None:
    teacher_source = str(settings.get("models.teacher_source")).casefold()
    if "gpt-oss" in teacher_source:
        environment = os.environ.copy()
        environment["TIKTOKEN_ENCODINGS_BASE"] = str(_ensure_harmony_encodings(settings))
        _run_vllm(teacher_vllm_command(settings, profile), "teacher", env=environment)
        return
    _run_vllm(teacher_vllm_command(settings, profile), "teacher")


def _run_vllm(
    command: list[str], service_name: str, *, env: dict[str, str] | None = None
) -> None:
    # CalledProcessError includes the complete argv in its message. Because vLLM receives its API
    # key through argv, raising that exception would disclose the key in terminal logs.
    completed = subprocess.run(command, check=False, env=env)
    if completed.returncode:
        raise RuntimeError(
            f"The {service_name} vLLM process exited with status {completed.returncode}. "
            "Review the first engine error above; API credentials have been omitted."
        )


def _ensure_harmony_encodings(settings: Settings) -> Path:
    configured_base = os.getenv("TIKTOKEN_ENCODINGS_BASE")
    base = (
        Path(configured_base).expanduser()
        if configured_base
        else settings.root / "artifacts" / "tokenizers" / "tiktoken_encodings"
    )
    base = base.resolve()
    base.mkdir(parents=True, exist_ok=True)

    for filename, expected_hash in _HARMONY_ENCODINGS.items():
        destination = base / filename
        if destination.is_file() and _file_sha256(destination) == expected_hash:
            continue

        partial = destination.with_suffix(f"{destination.suffix}.part")
        partial.unlink(missing_ok=True)
        digest = sha256()
        url = f"{_HARMONY_ENCODINGS_URL}/{filename}"
        try:
            with urlopen(url, timeout=60) as source, partial.open("wb") as target:  # noqa: S310
                while block := source.read(1024 * 1024):
                    target.write(block)
                    digest.update(block)
            computed_hash = digest.hexdigest()
            if computed_hash != expected_hash:
                raise RuntimeError(
                    f"Integrity check failed for {filename}: expected {expected_hash}, "
                    f"received {computed_hash}"
                )
            partial.replace(destination)
        except Exception:
            partial.unlink(missing_ok=True)
            raise

    return base


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()
