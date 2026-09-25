from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from .config import Settings
from .io import write_json_atomic


def merge_adapter(settings: Settings) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    base_model_id = str(settings.get("models.base"))
    adapter_path = settings.path("models.adapter_output") / "final"
    output_path = settings.path("models.merged_output")
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    base_config = AutoConfig.from_pretrained(base_model_id)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    merged = PeftModel.from_pretrained(base, str(adapter_path)).merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    merged.save_pretrained(output_path, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(output_path)
    language_model_only = False
    base_text_config = getattr(base_config, "text_config", None)
    if (
        base_text_config is not None
        and getattr(base_text_config, "model_type", None)
        == getattr(merged.config, "model_type", None)
    ):
        # AutoModelForCausalLM extracts the language component of composite Qwen
        # checkpoints. Restore the outer config for vLLM while explicitly disabling
        # the absent multimodal encoder and preserving the merged text configuration.
        composite_config = base_config.to_dict()
        composite_config["text_config"] = merged.config.to_dict()
        composite_config["language_model_only"] = True
        write_json_atomic(output_path / "config.json", composite_config)
        language_model_only = True
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "base_model": base_model_id,
        "adapter_path": str(adapter_path),
        "merged_model": str(output_path),
        "dtype": "bfloat16",
        "quantization": "none",
        "language_model_only": language_model_only,
        "note": "Quantized candidates must pass the deployment comparison benchmark before activation.",
    }
    write_json_atomic(output_path / "package_manifest.json", manifest)
    return manifest
