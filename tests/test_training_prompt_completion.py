from __future__ import annotations

import unittest

from giis_prod.training import (
    _messages_to_prompt_completion,
    _tokenize_for_completion_loss,
)


class FakeGraniteTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        return_dict: bool,
        add_generation_prompt: bool = False,
        enable_thinking: bool | None = None,
    ) -> dict[str, list[int]]:
        ids = [10, 11]
        if add_generation_prompt:
            ids.append(12)
        elif messages[-1]["role"] == "assistant":
            ids.extend([12, 20, 21])
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class PromptCompletionConversionTests(unittest.TestCase):
    def test_final_assistant_message_becomes_only_completion(self) -> None:
        source = {
            "example_id": "sft_example",
            "messages": [
                {"role": "system", "content": "Ground answers."},
                {"role": "user", "content": "Evidence and question"},
                {"role": "assistant", "content": "Grounded answer [E1]."},
            ],
        }

        converted = _messages_to_prompt_completion(source)

        self.assertEqual(converted["prompt"], source["messages"][:2])
        self.assertEqual(converted["completion"], [source["messages"][-1]])
        self.assertNotIn("example_id", converted)

    def test_conversion_does_not_alias_source_messages(self) -> None:
        source = {
            "messages": [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer"},
            ]
        }

        converted = _messages_to_prompt_completion(source)
        converted["completion"][0]["content"] = "Changed"

        self.assertEqual(source["messages"][-1]["content"], "Answer")

    def test_missing_messages_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two"):
            _messages_to_prompt_completion({})

    def test_final_message_must_be_nonempty_assistant(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty assistant"):
            _messages_to_prompt_completion(
                {
                    "messages": [
                        {"role": "system", "content": "System"},
                        {"role": "user", "content": "Question"},
                    ]
                }
            )

    def test_tokenization_builds_exact_completion_mask(self) -> None:
        source = {
            "messages": [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer"},
            ]
        }

        tokenized = _tokenize_for_completion_loss(source, FakeGraniteTokenizer())

        self.assertEqual(tokenized["input_ids"], [10, 11, 12, 20, 21])
        self.assertEqual(tokenized["completion_mask"], [0, 0, 0, 1, 1])

    def test_tokenization_forwards_qwen_no_thinking_control(self) -> None:
        class RecordingTokenizer(FakeGraniteTokenizer):
            def __init__(self) -> None:
                self.thinking_values: list[bool | None] = []

            def apply_chat_template(self, messages, **kwargs):
                self.thinking_values.append(kwargs.get("enable_thinking"))
                return super().apply_chat_template(messages, **kwargs)

        source = {
            "messages": [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer"},
            ]
        }
        tokenizer = RecordingTokenizer()

        tokenized = _tokenize_for_completion_loss(
            source, tokenizer, enable_thinking=False
        )

        self.assertEqual(tokenizer.thinking_values, [False, False])
        self.assertEqual(tokenized["completion_mask"], [0, 0, 0, 1, 1])

    def test_tokenization_rejects_nonprefix_prompt(self) -> None:
        class NonPrefixTokenizer(FakeGraniteTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                result = super().apply_chat_template(messages, **kwargs)
                if not kwargs.get("add_generation_prompt"):
                    result["input_ids"][0] = 99
                return result

        source = {
            "messages": [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer"},
            ]
        }

        with self.assertRaisesRegex(ValueError, "not a prefix"):
            _tokenize_for_completion_loss(source, NonPrefixTokenizer())
        with self.assertRaisesRegex(ValueError, "non-empty assistant"):
            _messages_to_prompt_completion(
                {
                    "messages": [
                        {"role": "user", "content": "Question"},
                        {"role": "assistant", "content": "  "},
                    ]
                }
            )


if __name__ == "__main__":
    unittest.main()
