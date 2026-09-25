from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import gradio as gr

from giis_prod.ui import (
    _cited_evidence,
    _clear_ui_state,
    _display_text,
    _format_summary_sections,
    _progressive_text,
    _stage_user_message,
    _stream_summary_output,
    build_app,
    launch_app,
)


class UiCompatibilityTests(unittest.TestCase):
    def test_user_question_is_staged_immediately_without_duplication(self):
        prior = [{"role": "assistant", "content": "Earlier answer"}]

        cleared, history, pending = _stage_user_message("  New question?  ", prior)

        self.assertEqual(cleared, "")
        self.assertEqual(pending, "New question?")
        self.assertEqual(history[-1], {"role": "user", "content": "New question?"})
        self.assertEqual(len(history), 2)
        self.assertEqual(prior, [{"role": "assistant", "content": "Earlier answer"}])

    def test_display_text_normalizes_stroked_z_glyphs(self):
        self.assertEqual(_display_text("Ƶambia and ƶebra"), "Zambia and zebra")

    def test_summary_sections_are_bold_and_content_starts_on_next_line(self):
        raw = "Purpose: Explain the scope.\nApproach: Analyze the evidence.\nFindings: Report results."

        rendered = _format_summary_sections(raw)

        self.assertIn("**Purpose**\n\nExplain the scope.", rendered)
        self.assertIn("**Approach**\n\nAnalyze the evidence.", rendered)
        self.assertIn("**Findings**\n\nReport results.", rendered)

    def test_validated_text_is_progressively_rendered_and_committed_exactly(self):
        answer = "One two three four five six seven eight nine ten."

        updates = list(_progressive_text(answer, words_per_update=4))

        self.assertGreater(len(updates), 1)
        self.assertEqual(updates[-1], answer)

    def test_citation_popovers_include_only_markers_used_in_answer(self):
        evidence = [
            SimpleNamespace(evidence_id="E1"),
            SimpleNamespace(evidence_id="E2"),
            SimpleNamespace(evidence_id="E3"),
        ]
        result = SimpleNamespace(raw_marker_answer="Supported claim [E2].", evidence=evidence)

        cited = _cited_evidence(result)

        self.assertEqual([item.evidence_id for item in cited], ["E2"])

    def test_clear_state_removes_and_disables_stale_source_selection(self):
        state = _clear_ui_state()
        selector = state[3]

        self.assertEqual(state[0], [])
        self.assertIsNone(selector.value)
        self.assertEqual(selector.choices, [])
        self.assertFalse(selector.interactive)
        self.assertIsNone(state[5])

    def test_summary_stream_commits_last_non_empty_value(self):
        generator = SimpleNamespace(
            summarize_source_stream=lambda document_id: iter(
                ["Purpose: first", "", "Purpose: first\n\nFindings: final", ""]
            )
        )

        output = list(_stream_summary_output(generator, "doc-1"))

        self.assertEqual(output[-1], "**Purpose**\n\nfirst\n**Findings**\n\nfinal")
        self.assertEqual(output[-2], output[-1])

    def test_summary_stream_keeps_partial_text_if_backend_raises(self):
        def interrupted(_document_id):
            yield "Purpose: retained"
            raise RuntimeError("stream interrupted")

        generator = SimpleNamespace(summarize_source_stream=interrupted)

        output = list(_stream_summary_output(generator, "doc-1"))

        self.assertEqual(output[-1], "**Purpose**\n\nretained")

    def test_app_builds_with_supported_gradio_chatbot_api(self):
        with patch("giis_prod.ui.GroundedGenerator"):
            app = build_app(object())

        self.assertIsInstance(app, gr.Blocks)
        serialized = str(app.get_config_file())
        self.assertIn("Climate and Ecology", serialized)
        self.assertIn("'# Climate and Ecology'", serialized)
        self.assertNotIn("**# Climate and Ecology**", serialized)
        self.assertIn("Your research question", serialized)
        self.assertIn("Clear Chat History", serialized)
        self.assertIn("Find your references here", serialized)
        self.assertIn("Summarize source", serialized)
        self.assertIn("Sources ready for summarization", serialized)
        self.assertIn("Reference Panel", serialized)
        self.assertIn("giis-split-layout", serialized)
        self.assertIn("giis-splitter", serialized)
        self.assertIn("giis-reference-panel", serialized)
        self.assertIn("giis-source-summary", serialized)
        self.assertIn("Resize chat and reference panels", serialized)
        self.assertNotIn("sidebar", serialized.lower())

        config = app.get_config_file()
        components = {item["id"]: item for item in config["components"]}
        selector = next(
            item
            for item in config["components"]
            if item["type"] == "dropdown"
            and item["props"].get("label") == "Sources ready for summarization"
        )
        self.assertTrue(selector["props"]["allow_custom_value"])
        summarize_dependency = next(
            item for item in config["dependencies"] if item["api_name"] == "summarize"
        )
        self.assertEqual(components[summarize_dependency["inputs"][0]]["type"], "state")
        self.assertEqual(len(summarize_dependency["outputs"]), 1)
        self.assertEqual(components[summarize_dependency["outputs"][0]]["type"], "markdown")

    def test_launch_passes_gradio_six_theme_and_head(self):
        app = MagicMock()
        values = {"serving.host": "127.0.0.1", "serving.ui_port": 7860}
        settings = MagicMock()
        settings.get.side_effect = values.__getitem__

        with (
            patch("giis_prod.ui.build_app", return_value=app),
            patch.dict("os.environ", {}, clear=True),
        ):
            launch_app(settings)

        app.launch.assert_called_once()
        kwargs = app.launch.call_args.kwargs
        self.assertEqual(kwargs["server_name"], "0.0.0.0")
        self.assertEqual(kwargs["server_port"], 7860)
        self.assertTrue(kwargs["share"])
        self.assertIsInstance(kwargs["theme"], gr.themes.Base)
        self.assertIn("GIIS Climate and Ecology", kwargs["head"])
        self.assertIn("overflow: hidden", kwargs["css"])
        self.assertIn("#giis-chatbot {", kwargs["css"])
        self.assertIn("#giis-chatbot::-webkit-scrollbar", kwargs["css"])
        self.assertNotIn("#giis-chatbot .bubble-wrap", kwargs["css"])
        self.assertIn("scrollbar-width: none", kwargs["css"])
        self.assertIn("--giis-chat-percent: 65%", kwargs["css"])
        self.assertIn("#giis-source-summary", kwargs["css"])
        self.assertIn("overflow-x: hidden", kwargs["css"])
        self.assertIn("font-family: Arial, Helvetica, sans-serif", kwargs["css"])
        self.assertIn("grid-template-columns", kwargs["css"])
        self.assertIn("Math.max(45, Math.min(80, value))", kwargs["js"])
        self.assertIn('splitter.setAttribute("role", "separator")', kwargs["js"])
        self.assertIn('aria-valuenow', kwargs["js"])
        self.assertEqual(kwargs["footer_links"], [])


if __name__ == "__main__":
    unittest.main()
