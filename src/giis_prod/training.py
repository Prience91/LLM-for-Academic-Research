from __future__ import annotations

import json
import os
from hashlib import sha256
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .benchmark_freeze import validate_benchmark_freeze
from .config import Settings
from .dataset_release import dataset_release_path, validate_dataset_release
from .io import write_json_atomic


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def training_precision_policy(configuration: dict[str, Any]) -> dict[str, Any]:
    """Describe the base-weight precision used while training the LoRA adapter."""
    quantized_base = bool(configuration.get("use_4bit", False))
    return {
        "method": "qlora_nf4" if quantized_base else "lora_bf16",
        "base_weight_precision": "nf4_4bit" if quantized_base else "bfloat16",
        "adapter_precision": "bfloat16",
        "quantized_base": quantized_base,
        "full_parameter_finetuning": False,
    }


def longest_training_example_indices(tokenized_split: Any, count: int) -> list[int]:
    """Return stable indices for the longest tokenized examples in a split."""
    if count < 0:
        raise ValueError("training.worst_case_canary_examples cannot be negative")
    if count > len(tokenized_split):
        raise ValueError(
            "training.worst_case_canary_examples exceeds the training split size"
        )
    ranked = sorted(
        range(len(tokenized_split)),
        key=lambda index: (-len(tokenized_split[index]["input_ids"]), index),
    )
    return ranked[:count]


def _load_and_validate_sft_exclusions(settings: Settings) -> dict[str, Any]:
    """Load an audit exclusion overlay without changing the immutable dataset release."""
    path = settings.path("paths.sft_exclusions")
    if not path.exists():
        raise FileNotFoundError(
            f"Required SFT semantic-exclusion manifest is missing: {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("SFT semantic-exclusion manifest must use schema_version 1")
    entries = payload.get("excluded_examples")
    if not isinstance(entries, list):
        raise ValueError("excluded_examples must be a list")
    excluded_ids: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Every semantic exclusion must be an object")
        example_id = str(entry.get("example_id") or "").strip()
        reason_codes = entry.get("reason_codes")
        if not example_id or not isinstance(reason_codes, list) or not reason_codes:
            raise ValueError("Every semantic exclusion needs an example_id and reason_codes")
        excluded_ids.append(example_id)
    if len(excluded_ids) != len(set(excluded_ids)):
        raise ValueError("SFT semantic-exclusion manifest contains duplicate example IDs")
    expected_count = int(payload.get("expected_exclusion_count", -1))
    if expected_count != len(excluded_ids):
        raise ValueError(
            "SFT semantic-exclusion count does not match expected_exclusion_count"
        )

    queue_path = settings.path("paths.semantic_audit_queue")
    if not queue_path.is_file():
        raise FileNotFoundError(f"Semantic-audit queue is missing: {queue_path}")
    expected_queue_sha256 = str(payload.get("source_audit_queue_sha256") or "")
    actual_queue_sha256 = _sha256_file(queue_path)
    if expected_queue_sha256 != actual_queue_sha256:
        raise ValueError("Semantic-audit queue digest does not match the exclusion manifest")

    occurrences: dict[str, list[str]] = {example_id: [] for example_id in excluded_ids}
    sft_dir = settings.path("paths.sft_dir")
    for split in ("train", "validation"):
        split_path = sft_dir / f"{split}.jsonl"
        with split_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                example_id = str(record.get("example_id") or "")
                if example_id in occurrences:
                    occurrences[example_id].append(f"{split}:{line_number}")
    invalid = {
        example_id: locations
        for example_id, locations in occurrences.items()
        if len(locations) != 1
    }
    if invalid:
        raise ValueError(
            "Every excluded example must occur exactly once in the immutable release: "
            + json.dumps(invalid, sort_keys=True)
        )
    split_counts = {"train": 0, "validation": 0}
    for locations in occurrences.values():
        split_counts[locations[0].split(":", 1)[0]] += 1
    return {
        "manifest_path": str(path),
        "manifest_sha256": _sha256_file(path),
        "source_audit_queue_path": str(queue_path),
        "source_audit_queue_sha256": actual_queue_sha256,
        "excluded_ids": excluded_ids,
        "excluded_count": len(excluded_ids),
        "split_counts": split_counts,
    }


def _messages_to_prompt_completion(example: dict[str, Any]) -> dict[str, Any]:
    """Convert one GIIS chat record to TRL's conversational completion format.

    Granite's upstream chat template does not expose an assistant-token mask.  TRL can
    still apply answer-only loss deterministically when the final assistant message is
    represented as a completion and every preceding message is represented as its prompt.
    """
    messages = example.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("SFT record must contain at least two chat messages")
    if not all(isinstance(message, dict) for message in messages):
        raise ValueError("SFT record messages must be objects")
    final = messages[-1]
    if final.get("role") != "assistant" or not str(final.get("content") or "").strip():
        raise ValueError("SFT record must end with a non-empty assistant message")
    prompt = [dict(message) for message in messages[:-1]]
    completion = [dict(final)]
    return {"prompt": prompt, "completion": completion}


def _tokenize_for_completion_loss(
    example: dict[str, Any],
    tokenizer: Any,
    *,
    enable_thinking: bool | None = None,
) -> dict[str, list[int]]:
    """Tokenize one record and construct an exact answer-only loss mask.

    Some Granite tokenizers return ``BatchEncoding`` from ``apply_chat_template`` even
    when TRL expects a plain token list. Precomputing both arrays avoids TRL mistaking
    the two mapping keys for a two-token prompt and silently training on corpus evidence.
    """
    converted = _messages_to_prompt_completion(example)
    template_kwargs: dict[str, Any] = {}
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = enable_thinking
    prompt_encoded = tokenizer.apply_chat_template(
        converted["prompt"],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        **template_kwargs,
    )
    full_encoded = tokenizer.apply_chat_template(
        converted["prompt"] + converted["completion"],
        tokenize=True,
        return_dict=True,
        **template_kwargs,
    )
    prompt_ids = list(prompt_encoded["input_ids"])
    full_ids = list(full_encoded["input_ids"])
    if prompt_ids and isinstance(prompt_ids[0], list):
        prompt_ids = list(prompt_ids[0])
    if full_ids and isinstance(full_ids[0], list):
        full_ids = list(full_ids[0])
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Granite prompt tokens are not a prefix of the completed chat")
    completion_tokens = len(full_ids) - len(prompt_ids)
    if completion_tokens < 1:
        raise ValueError("SFT record produced no assistant completion tokens")
    return {
        "input_ids": full_ids,
        "completion_mask": [0] * len(prompt_ids) + [1] * completion_tokens,
    }


def train_sft(
    settings: Settings,
    resume_from_checkpoint: str | bool | None = None,
) -> dict[str, Any]:
    # Validate before importing torch/Transformers or loading any model weights. A missing,
    # interrupted, stale, or mutated dataset must fail quickly and cheaply.
    dataset_release = validate_dataset_release(settings)
    semantic_exclusions = _load_and_validate_sft_exclusions(settings)
    validate_benchmark_freeze(settings)

    configuration = settings.section("training")
    precision_policy = training_precision_policy(configuration)
    maximum_steps = int(configuration.get("max_steps", -1))
    gradient_accumulation_steps = int(configuration["gradient_accumulation_steps"])
    worst_case_canary_examples = int(
        configuration.get("worst_case_canary_examples", 0)
    )
    if worst_case_canary_examples:
        if maximum_steps != 1:
            raise ValueError(
                "training.worst_case_canary_examples requires training.max_steps = 1"
            )
        if worst_case_canary_examples != gradient_accumulation_steps:
            raise ValueError(
                "Worst-case canary size must equal gradient_accumulation_steps so the "
                "single optimizer step exercises every selected example"
            )
    torch_empty_cache_steps = int(configuration.get("torch_empty_cache_steps", 0))
    if torch_empty_cache_steps < 0:
        raise ValueError("training.torch_empty_cache_steps cannot be negative")

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer

    set_seed(int(configuration["seed"]))
    wandb_enabled = bool(configuration.get("wandb_enabled", False))
    wandb_module = None
    run_name = str(configuration.get("wandb_run_name") or "").strip() or (
        "smoke-run-ii-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    if wandb_enabled:
        import wandb

        if not wandb.login():
            raise RuntimeError(
                "W&B login is required. Run `wandb login` in this tmux environment and retry."
            )
        wandb_module = wandb
        os.environ.setdefault("WANDB_PROJECT", str(configuration["wandb_project"]))
        os.environ.setdefault("WANDB_TAGS", str(configuration.get("wandb_tags") or ""))
    base_model = str(settings.get("models.base"))
    output_dir = settings.path("models.adapter_output")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(settings.path("paths.sft_dir") / "train.jsonl"),
            "validation": str(settings.path("paths.sft_dir") / "validation.jsonl"),
        },
    )
    excluded_ids = set(semantic_exclusions["excluded_ids"])
    original_split_sizes = {split: len(dataset[split]) for split in dataset}
    if excluded_ids:
        for split in dataset:
            dataset[split] = dataset[split].filter(
                lambda example: example.get("example_id") not in excluded_ids,
                desc=f"Applying independent semantic-audit exclusions to {split}",
            )
    filtered_split_sizes = {split: len(dataset[split]) for split in dataset}
    observed_removed = sum(original_split_sizes.values()) - sum(
        filtered_split_sizes.values()
    )
    if observed_removed != semantic_exclusions["excluded_count"]:
        raise RuntimeError(
            "Applied semantic-exclusion count differs from the validated manifest"
        )
    template_enable_thinking = configuration.get("enable_thinking")
    if template_enable_thinking is not None and not isinstance(
        template_enable_thinking, bool
    ):
        raise ValueError("training.enable_thinking must be a boolean when configured")
    # Precompute exact completion masks before loading the model. Granite's tokenizer
    # returns BatchEncoding from apply_chat_template, while TRL 0.24's conversational
    # prompt/completion branch assumes a plain list and would infer a two-token prompt.
    for split in dataset:
        dataset[split] = dataset[split].map(
            _tokenize_for_completion_loss,
            fn_kwargs={
                "tokenizer": tokenizer,
                "enable_thinking": template_enable_thinking,
            },
            remove_columns=dataset[split].column_names,
            desc=f"Tokenizing {split} with verified completion masks",
        )
    worst_case_canary_lengths: list[int] = []
    if worst_case_canary_examples:
        selected_indices = longest_training_example_indices(
            dataset["train"], worst_case_canary_examples
        )
        worst_case_canary_lengths = [
            len(dataset["train"][index]["input_ids"]) for index in selected_indices
        ]
        dataset["train"] = dataset["train"].select(selected_indices)
        print(
            json.dumps(
                {
                    "worst_case_canary_examples": worst_case_canary_examples,
                    "minimum_selected_tokens": min(worst_case_canary_lengths),
                    "maximum_selected_tokens": max(worst_case_canary_lengths),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    model_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
    }
    attention_implementation = str(
        configuration.get("attention_implementation") or ""
    ).strip()
    if attention_implementation:
        model_kwargs["attn_implementation"] = attention_implementation
    if precision_policy["quantized_base"]:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    if precision_policy["quantized_base"]:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(configuration["gradient_checkpointing"]),
        )
    model.config.use_cache = False

    report_to = "wandb" if wandb_enabled else "none"
    hub_model_id = str(configuration.get("hub_model_id") or "")
    trainer_config = SFTConfig(
        output_dir=str(output_dir),
        max_length=int(configuration["max_length"]),
        per_device_train_batch_size=int(configuration["train_batch_size"]),
        per_device_eval_batch_size=int(configuration["eval_batch_size"]),
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=float(configuration["learning_rate"]),
        num_train_epochs=float(configuration["epochs"]),
        max_steps=maximum_steps,
        bf16=True,
        gradient_checkpointing=bool(configuration["gradient_checkpointing"]),
        gradient_checkpointing_kwargs={
            "use_reentrant": bool(
                configuration.get("gradient_checkpointing_use_reentrant", True)
            )
        },
        assistant_only_loss=False,
        completion_only_loss=True,
        packing=False,
        prediction_loss_only=bool(configuration.get("prediction_loss_only", True)),
        torch_empty_cache_steps=torch_empty_cache_steps or None,
        eval_strategy="steps",
        eval_steps=int(configuration["eval_steps"]),
        save_strategy="steps",
        save_steps=int(configuration["save_steps"]),
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=int(configuration.get("logging_steps", 10)),
        report_to=report_to,
        run_name=run_name if wandb_enabled else None,
        seed=int(configuration["seed"]),
        data_seed=int(configuration["seed"]),
        push_to_hub=bool(hub_model_id),
        hub_model_id=hub_model_id or None,
    )
    lora = LoraConfig(
        r=int(configuration["lora_rank"]),
        lora_alpha=int(configuration["lora_alpha"]),
        lora_dropout=float(configuration["lora_dropout"]),
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    trainer = SFTTrainer(
        model=model,
        args=trainer_config,
        peft_config=lora,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )
    result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    metrics = dict(result.metrics)
    wandb_record: dict[str, Any] | None = None
    if wandb_module is not None and wandb_module.run is not None:
        wandb_module.config.update(
            {
                "giis_generation_policy_version": dataset_release[
                    "generation_policy_version"
                ],
                "giis_policy_fingerprint": dataset_release["policy_fingerprint"],
                "giis_training_selection_digest": dataset_release["training_selection"][
                    "selection_digest"
                ],
                "giis_loss_scope": "completion_only",
                "giis_training_precision": precision_policy["method"],
                "giis_base_weight_precision": precision_policy["base_weight_precision"],
            },
            allow_val_change=True,
        )
        wandb_module.run.summary.update(metrics)
        wandb_record = {
            "entity": wandb_module.run.entity,
            "project": wandb_module.run.project,
            "run_id": wandb_module.run.id,
            "run_name": wandb_module.run.name,
            "url": wandb_module.run.url,
        }
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "base_model": base_model,
        "adapter_path": str(output_dir / "final"),
        "loss_scope": "completion_only",
        "assistant_only_loss": False,
        "completion_only_loss": True,
        "worst_case_canary": {
            "enabled": bool(worst_case_canary_examples),
            "selected_examples": worst_case_canary_examples,
            "selected_token_lengths": worst_case_canary_lengths,
        },
        "training_precision": precision_policy,
        "dataset_release": {
            "manifest_path": str(dataset_release_path(settings)),
            "generation_policy_version": dataset_release["generation_policy_version"],
            "policy_fingerprint": dataset_release["policy_fingerprint"],
            "training_selection_digest": dataset_release["training_selection"][
                "selection_digest"
            ],
            "file_hashes": {
                name: descriptor["sha256"]
                for name, descriptor in dataset_release["files"].items()
            },
        },
        "semantic_exclusions": {
            **semantic_exclusions,
            "original_split_sizes": original_split_sizes,
            "filtered_split_sizes": filtered_split_sizes,
        },
        "configuration": configuration,
        "metrics": metrics,
        "wandb": wandb_record,
    }
    write_json_atomic(output_dir / "training_manifest.json", manifest)
    return manifest
