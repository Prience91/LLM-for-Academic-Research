from __future__ import annotations

import pytest

from giis_prod.training import longest_training_example_indices, training_precision_policy


def test_selected_bf16_lora_policy_has_no_quantized_or_full_parameter_base() -> None:
    policy = training_precision_policy({"use_4bit": False})
    assert policy == {
        "method": "lora_bf16",
        "base_weight_precision": "bfloat16",
        "adapter_precision": "bfloat16",
        "quantized_base": False,
        "full_parameter_finetuning": False,
    }


def test_optional_qlora_policy_remains_explicit() -> None:
    policy = training_precision_policy({"use_4bit": True})
    assert policy["method"] == "qlora_nf4"
    assert policy["base_weight_precision"] == "nf4_4bit"
    assert policy["quantized_base"] is True
    assert policy["full_parameter_finetuning"] is False


def test_longest_training_example_indices_are_stable_and_descending() -> None:
    split = [
        {"input_ids": [1, 2]},
        {"input_ids": [1, 2, 3, 4]},
        {"input_ids": [1, 2, 3, 4]},
        {"input_ids": [1]},
    ]
    assert longest_training_example_indices(split, 3) == [1, 2, 0]


def test_longest_training_example_indices_reject_invalid_count() -> None:
    split = [{"input_ids": [1]}]
    with pytest.raises(ValueError, match="cannot be negative"):
        longest_training_example_indices(split, -1)
    with pytest.raises(ValueError, match="exceeds the training split size"):
        longest_training_example_indices(split, 2)
