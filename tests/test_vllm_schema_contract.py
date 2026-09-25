from __future__ import annotations

import unittest

from jsonschema import Draft202012Validator
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

from giis_prod.teacher import _grounding_schema


class VllmSchemaContractTests(unittest.TestCase):
    def test_grounding_schema_is_valid_for_installed_openai_protocol(self):
        response_format = _grounding_schema(
            exact_items=2,
            maximum_evidence_index=8,
        )
        Draft202012Validator.check_schema(
            response_format["json_schema"]["schema"]
        )
        request = ChatCompletionRequest(
            model="gpt-oss-120b",
            messages=[{"role": "user", "content": "schema validation only"}],
            response_format=response_format,
        )
        self.assertEqual(
            request.response_format.json_schema.name,
            "giis_semantic_verdicts",
        )


if __name__ == "__main__":
    unittest.main()
