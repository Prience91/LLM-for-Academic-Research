from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import giis_pipeline
from giis_prod.config import Settings
from giis_prod.server import serve_teacher_vllm, teacher_vllm_command, vllm_command


class ConsolidatedRunnerTests(unittest.TestCase):
    def test_all_operator_commands_are_registered(self):
        parser = giis_pipeline.build_parser()
        expected = {
            "status",
            "audit-corpus",
            "corpus",
            "plan-heldout",
            "materialize-training",
            "author-heldout",
            "heldout-pipeline",
            "dataset",
            "teacher-serve",
            "teacher-ab-plan",
            "teacher-ab-run",
            "teacher-ab-compare",
            "creative-audit",
            "train",
            "merge",
            "plan-semantic-audit",
            "serve",
            "ui",
            "evaluate",
            "stream-probe",
            "load-probe",
            "compare",
            "build-all",
        }
        subcommands = next(
            action.choices
            for action in parser._actions
            if getattr(action, "choices", None)
        )
        self.assertEqual(set(subcommands), expected)

    def test_vllm_key_is_redacted(self):
        self.assertEqual(
            giis_pipeline._redacted_command(
                ["vllm", "serve", "model", "--api-key", "secret", "--port", "8000"]
            ),
            ["vllm", "serve", "model", "--api-key", "***", "--port", "8000"],
        )

    def test_status_provides_next_step(self):
        status = giis_pipeline.pipeline_status(Settings.load())
        self.assertTrue(status["next_step"])
        self.assertIn("artifacts", status)

    def test_teacher_server_command_uses_expected_endpoint(self):
        with patch.dict(os.environ, {"TEACHER_API_KEY": "test-secret"}, clear=False):
            command = teacher_vllm_command(Settings.load())
        self.assertEqual(command[2], "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4")
        self.assertEqual(
            command[command.index("--served-model-name") + 1],
            "qwen3.5-122b-a10b-gptq-int4-teacher",
        )
        self.assertEqual(command[command.index("--port") + 1], "8800")

    def test_teacher_ab_profiles_change_only_capacity_controls(self):
        with patch.dict(os.environ, {"TEACHER_API_KEY": "test-secret"}, clear=False):
            baseline = teacher_vllm_command(Settings.load(), profile="ab-baseline")
            candidate = teacher_vllm_command(Settings.load(), profile="ab-candidate")
        self.assertEqual(baseline[baseline.index("--max-model-len") + 1], "32768")
        self.assertEqual(baseline[baseline.index("--max-num-seqs") + 1], "1")
        self.assertEqual(candidate[candidate.index("--max-model-len") + 1], "16384")
        self.assertEqual(candidate[candidate.index("--max-num-seqs") + 1], "2")
        ignored = {"--max-model-len", "32768", "16384", "--max-num-seqs", "1", "2"}
        self.assertEqual(
            [value for value in baseline if value not in ignored],
            [value for value in candidate if value not in ignored],
        )

    def test_student_server_registers_configured_model_alias(self):
        with patch.dict(os.environ, {"VLLM_API_KEY": "test-secret"}, clear=False):
            command = vllm_command(Settings.load())
        self.assertEqual(
            command[command.index("--served-model-name") + 1],
            "models/qwen3.8-27b-mini-v17-turbo-qlora-nf4-merged",
        )

    def test_teacher_server_failure_does_not_disclose_api_key(self):
        secret = "must-not-appear"
        completed = __import__("subprocess").CompletedProcess([], 1)
        with patch.dict(os.environ, {"TEACHER_API_KEY": secret}, clear=False):
            with patch("giis_prod.server._ensure_harmony_encodings") as harmony:
                with patch("giis_prod.server.subprocess.run", return_value=completed) as run:
                    with self.assertRaises(RuntimeError) as raised:
                        serve_teacher_vllm(Settings.load())
        self.assertNotIn(secret, str(raised.exception))
        harmony.assert_not_called()
        self.assertIsNone(run.call_args.kwargs["env"])


if __name__ == "__main__":
    unittest.main()

