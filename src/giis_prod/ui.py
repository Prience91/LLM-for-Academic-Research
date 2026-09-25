from __future__ import annotations

import os
import re
from typing import Any

from .citations import render_citation_popovers
from .config import Settings
from .generation import GroundedGenerator

_GIIS_LOGO_URL = "https://www.thegiis.org/frontend/assets/images/GIIS.png"
_EVIDENCE_MARKER = re.compile(r"\[(E\d+)\]")
_SUMMARY_SECTION = re.compile(
    r"(?im)^\s*(?:\*\*)?(Purpose|Approach|Findings|"
    r"Limitations(?:\s+or\s+Implications)?|Implications)(?:\*\*)?\s*:\s*"
)
_DISPLAY_GLYPHS = str.maketrans({"Ƶ": "Z", "ƶ": "z"})
_REFERENCE_PLACEHOLDER = (
    "## Reference Panel\n\n"
    "*Find your references here*"
)
_CUSTOM_HEAD = f"""
<meta property="og:site_name" content="GIIS Climate and Ecology" />
<meta property="og:title" content="GIIS Climate and Ecology" />
<meta property="og:description" content="Synthesizing climate and social-ecological systems research" />
<meta property="og:image" content="{_GIIS_LOGO_URL}" />
<meta property="og:image:secure_url" content="{_GIIS_LOGO_URL}" />
<meta property="og:image:type" content="image/png" />
<meta property="og:type" content="website" />
<meta name="twitter:card" content="summary_large_image" />
<meta name="twitter:title" content="GIIS Climate and Ecology" />
<meta name="twitter:description" content="Synthesizing climate and social-ecological systems research" />
<meta name="twitter:image" content="{_GIIS_LOGO_URL}" />
"""
_SPLIT_VIEW_JS = r"""
(() => {
  const initializeSplitView = () => {
    const layout = document.getElementById("giis-split-layout");
    const splitter = document.getElementById("giis-splitter");
    if (!layout || !splitter || splitter.dataset.giisReady === "true") return;
    splitter.dataset.giisReady = "true";
    const nestedSeparator = splitter.querySelector('[role="separator"]');
    if (nestedSeparator) {
      nestedSeparator.setAttribute("aria-hidden", "true");
      nestedSeparator.setAttribute("tabindex", "-1");
    }
    splitter.setAttribute("role", "separator");
    splitter.setAttribute("aria-label", "Resize chat and reference panels");
    splitter.setAttribute("aria-orientation", "vertical");
    splitter.setAttribute("aria-valuemin", "45");
    splitter.setAttribute("aria-valuemax", "80");
    splitter.setAttribute("tabindex", "0");

    const setRatio = (value) => {
      const ratio = Math.max(45, Math.min(80, value));
      layout.style.setProperty("--giis-chat-percent", `${ratio}%`);
      splitter.setAttribute("aria-valuenow", String(Math.round(ratio)));
    };
    const ratioFromPointer = (event) => {
      const bounds = layout.getBoundingClientRect();
      setRatio(((event.clientX - bounds.left) / bounds.width) * 100);
    };

    splitter.addEventListener("pointerdown", (event) => {
      if (window.matchMedia("(max-width: 768px)").matches) return;
      splitter.setPointerCapture(event.pointerId);
      document.body.classList.add("giis-resizing");
      ratioFromPointer(event);
    });
    splitter.addEventListener("pointermove", (event) => {
      if (splitter.hasPointerCapture(event.pointerId)) ratioFromPointer(event);
    });
    const stopResize = (event) => {
      if (splitter.hasPointerCapture(event.pointerId)) {
        splitter.releasePointerCapture(event.pointerId);
      }
      document.body.classList.remove("giis-resizing");
    };
    splitter.addEventListener("pointerup", stopResize);
    splitter.addEventListener("pointercancel", stopResize);
    splitter.addEventListener("keydown", (event) => {
      const current = parseFloat(
        layout.style.getPropertyValue("--giis-chat-percent") || "65"
      );
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        setRatio(current - 2);
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        setRatio(current + 2);
      } else if (event.key === "Home") {
        event.preventDefault();
        setRatio(45);
      } else if (event.key === "End") {
        event.preventDefault();
        setRatio(80);
      }
    });
    setRatio(65);
  };

  initializeSplitView();
  const observer = new MutationObserver(initializeSplitView);
  observer.observe(document.body, {childList: true, subtree: true});
  return () => observer.disconnect();
})()
"""
_CUSTOM_CSS = """
footer, .gradio-footer { display: none !important; }
.gradio-container, .gradio-container * {
  font-family: Arial, Helvetica, sans-serif !important;
}
.giis-modal-host { height: 0 !important; min-height: 0 !important; overflow: visible !important; }
#giis-chatbot {
  scrollbar-width: none !important;
  -ms-overflow-style: none !important;
}
#giis-chatbot::-webkit-scrollbar {
  display: none !important;
  width: 0 !important;
  height: 0 !important;
}
@media (min-width: 769px) {
  html, body { height: 100%; overflow: hidden !important; }
  .gradio-container { height: 100vh !important; max-height: 100vh !important;
    overflow: hidden !important; }
  #giis-split-layout {
    --giis-chat-percent: 65%;
    display: grid !important;
    grid-template-columns: minmax(0, var(--giis-chat-percent)) 10px minmax(0, 1fr) !important;
    gap: 0 !important;
    align-items: stretch !important;
    width: 100% !important;
  }
  #giis-main-chat, #giis-reference-panel { min-width: 0 !important; }
  #giis-reference-panel {
    max-height: calc(100vh - 5rem) !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    padding-left: 1rem !important;
  }
  #giis-source-summary {
    display: block !important;
    grid-column: 1 / -1 !important;
    min-width: 0 !important;
    width: 100% !important;
    max-width: 100% !important;
    overflow-wrap: anywhere !important;
  }
  #giis-source-summary .prose {
    max-width: none !important;
    width: 100% !important;
  }
  #giis-splitter {
    align-self: stretch !important;
    cursor: col-resize !important;
    min-width: 10px !important;
    position: relative !important;
    touch-action: none !important;
  }
  #giis-splitter::before {
    background: var(--border-color-primary, #4b5563);
    border-radius: 999px;
    content: "";
    inset: 0 4px;
    position: absolute;
    transition: background 120ms ease, inset 120ms ease;
  }
  #giis-splitter:hover::before, #giis-splitter:focus-visible::before {
    background: var(--primary-500, #6366f1);
    inset: 0 3px;
  }
  body.giis-resizing { cursor: col-resize !important; user-select: none !important; }
}
@media (max-width: 768px) {
  html, body, .gradio-container { overflow-y: auto !important; }
  #giis-split-layout { display: flex !important; flex-direction: column !important; }
  #giis-splitter { display: none !important; }
  #giis-reference-panel { max-height: none !important; overflow: visible !important; }
  #giis-chatbot { height: 60vh !important; }
}
"""


def _display_text(value: str) -> str:
    return value.translate(_DISPLAY_GLYPHS)


def _format_summary_sections(value: str) -> str:
    canonical = {
        "purpose": "Purpose",
        "approach": "Approach",
        "findings": "Findings",
        "limitations": "Limitations",
        "limitations or implications": "Limitations or Implications",
        "implications": "Implications",
    }

    def replace(match: re.Match[str]) -> str:
        label = canonical[" ".join(match.group(1).casefold().split())]
        return f"**{label}**\n\n"

    return _SUMMARY_SECTION.sub(replace, value)


def _progressive_text(value: str, words_per_update: int = 8):
    """Yield accumulating, already-validated text in small rendering groups."""

    clean = value.strip()
    if not clean:
        return
    pieces = re.findall(r"\S+\s*", clean)
    accumulated = ""
    last_emitted = ""
    for index, piece in enumerate(pieces, start=1):
        accumulated += piece
        if index % words_per_update == 0:
            last_emitted = accumulated.rstrip()
            yield last_emitted
    if clean != last_emitted:
        yield clean


def _cited_evidence(result: Any) -> list[Any]:
    """Return only evidence whose immutable marker appears in the answer draft."""

    used = set(_EVIDENCE_MARKER.findall(result.raw_marker_answer or ""))
    return [item for item in result.evidence if item.evidence_id in used]


def _stage_user_message(
    message: str, history: list[dict[str, str]] | None
) -> tuple[str, list[dict[str, str]], str]:
    """Render a submitted question immediately before slow generation begins."""

    clean = (message or "").strip()
    staged = list(history or [])
    if clean:
        staged.append({"role": "user", "content": clean})
    return "", staged, clean


def _clear_ui_state() -> tuple[Any, ...]:
    """Return a complete reset, including the source selector's value and state."""

    import gradio as gr

    return (
        [],
        render_citation_popovers([]),
        _REFERENCE_PLACEHOLDER,
        gr.Dropdown(
            choices=[], value=None, interactive=False, allow_custom_value=True
        ),
        "",
        None,
    )


def _stream_summary_output(generator: Any, document_id: str | None):
    """Yield only non-empty summaries and commit the last visible value."""

    if not document_id:
        yield "Select a retrieved source first."
        return

    last_display = ""
    try:
        for partial in generator.summarize_source_stream(document_id):
            display = _format_summary_sections(_display_text(partial)).strip()
            if not display:
                continue
            last_display = display
            yield display
    except Exception:
        if not last_display:
            yield (
                "The brief source summary could not complete within the current server limit. "
                "The standard grounded answer path remains available."
            )
            return

    # Gradio receives a deterministic terminal value instead of relying on a
    # second State component or a page-load restoration callback.
    if last_display:
        yield last_display


def build_app(settings: Settings) -> Any:
    import gradio as gr

    generator = GroundedGenerator(settings)

    def respond(clean: str, history: list[dict[str, str]] | None):
        clean = (clean or "").strip()
        history = list(history or [])
        if not clean:
            yield (
                history,
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
            )
        # The final user item was inserted by the immediate UI stage. Exclude that
        # duplicate from retrieval history while retaining earlier conversation turns.
        prior_history = history[:-1]
        result = generator.answer(clean, prior_history)
        documents: dict[str, str] = {}
        for item in result.evidence:
            documents.setdefault(item.document_id, _display_text(item.title))
        choices = [(title, document_id) for document_id, title in documents.items()]
        selector = gr.Dropdown(
            choices=choices,
            value=choices[0][1] if choices else None,
            interactive=bool(choices),
            allow_custom_value=True,
        )
        answer = _display_text(result.answer)
        updates = _progressive_text(answer)
        first_update = True
        for partial in updates:
            streamed_history = list(history)
            streamed_history.append({"role": "assistant", "content": partial})
            yield (
                streamed_history,
                (
                    render_citation_popovers(_cited_evidence(result))
                    if first_update
                    else gr.skip()
                ),
                (
                    _display_text(result.references_markdown)
                    if first_update
                    else gr.skip()
                ),
                selector if first_update else gr.skip(),
                "" if first_update else gr.skip(),
                choices[0][1] if first_update and choices else (
                    None if first_update else gr.skip()
                ),
            )
            first_update = False

    def summarize(document_id: str | None):
        yield from _stream_summary_output(generator, document_id)

    with gr.Blocks(title="Climate and Ecology", fill_height=True) as app:
        with gr.Row(equal_height=True):
            with gr.Column(scale=4):
                gr.Markdown("# Climate and Ecology")
            
        with gr.Row(equal_height=True, elem_id="giis-split-layout"):
            with gr.Column(elem_id="giis-main-chat", min_width=0):
                # Gradio 6 accepts message dictionaries directly and removed the legacy
                # ``type="messages"`` constructor argument.
                chatbot = gr.Chatbot(
                    height="calc(100vh - 20rem)",
                    min_height=360,
                    show_label=False,
                    elem_id="giis-chatbot",
                )
                with gr.Row():
                    message = gr.Textbox(
                        label="Your research question",
                        lines=2,
                        scale=4,
                        placeholder="Explore here... Press Shift + Enter to submit",
                    )
                    submit = gr.Button("Submit", variant="primary", scale=1)
                clear = gr.Button("Clear Chat History", variant="secondary")

            gr.HTML(
                '<div role="separator" aria-orientation="vertical" '
                'aria-label="Resize chat and reference panels" aria-valuemin="45" '
                'aria-valuemax="80" aria-valuenow="65" tabindex="0"></div>',
                elem_id="giis-splitter",
                container=False,
            )

            with gr.Column(elem_id="giis-reference-panel", min_width=0):
                references = gr.Markdown(_REFERENCE_PLACEHOLDER, label="Reference Panel")
                source_selector = gr.Dropdown(
                    label="Sources ready for summarization",
                    choices=[],
                    value=None,
                    interactive=False,
                    allow_custom_value=True,
                )
                summary_button = gr.Button("Summarize source")
                # Keep the summary as the final full-width item in this column. The
                # explicit ID prevents long Markdown from escaping to the right.
                summary_output = gr.Markdown(elem_id="giis-source-summary")

        pending_query = gr.State("")
        # The dropdown is presentation state. Keep the selected immutable corpus ID
        # separately so a UI restart or stale browser cannot invalidate summary input.
        selected_document_id = gr.State(None)

        # The metadata modals stay outside the split layout so they can overlay both panels.
        citation_popovers = gr.HTML(
            render_citation_popovers([]), container=False, elem_classes=["giis-modal-host"]
        )

        message_event = message.submit(
            _stage_user_message,
            inputs=[message, chatbot],
            outputs=[message, chatbot, pending_query],
            queue=False,
            show_progress="hidden",
        )
        message_event.then(
            respond,
            inputs=[pending_query, chatbot],
            outputs=[
                chatbot,
                citation_popovers,
                references,
                source_selector,
                summary_output,
                selected_document_id,
            ],
            show_progress="minimal",
        )
        submit_event = submit.click(
            _stage_user_message,
            inputs=[message, chatbot],
            outputs=[message, chatbot, pending_query],
            queue=False,
            show_progress="hidden",
        )
        submit_event.then(
            respond,
            inputs=[pending_query, chatbot],
            outputs=[
                chatbot,
                citation_popovers,
                references,
                source_selector,
                summary_output,
                selected_document_id,
            ],
            show_progress="minimal",
        )
        source_selector.change(
            lambda document_id: document_id,
            inputs=source_selector,
            outputs=selected_document_id,
            queue=False,
            show_progress="hidden",
        )
        clear.click(
            _clear_ui_state,
            outputs=[
                chatbot,
                citation_popovers,
                references,
                source_selector,
                summary_output,
                selected_document_id,
            ],
        )
        summary_event = summary_button.click(
            lambda document_id: (
                (
                    "Generating summary..."
                    if document_id
                    else "Select a retrieved source first."
                ),
                gr.Button(interactive=False),
            ),
            inputs=selected_document_id,
            outputs=[summary_output, summary_button],
            queue=False,
            show_progress="hidden",
        )
        stream_event = summary_event.then(
            summarize,
            inputs=selected_document_id,
            outputs=summary_output,
            show_progress="hidden",
            concurrency_limit=1,
            concurrency_id="source-summary",
        )
        stream_event.then(
            lambda: gr.Button(interactive=True),
            outputs=summary_button,
            queue=False,
            show_progress="hidden",
        )
    return app


def launch_app(settings: Settings) -> None:
    import gradio as gr

    username = os.getenv("GIIS_UI_USERNAME")
    password = os.getenv("GIIS_UI_PASSWORD")
    auth = (username, password) if username and password else None
    build_app(settings).launch(
        # Match the proven 5b deployment entry point: bind the Gradio UI on all
        # interfaces, while vLLM remains independently configured.
        server_name="0.0.0.0",
        server_port=int(settings.get("serving.ui_port")),
        share=True,
        auth=auth,
        theme=gr.themes.Soft(),
        head=_CUSTOM_HEAD,
        css=_CUSTOM_CSS,
        js=_SPLIT_VIEW_JS,
        footer_links=[],
    )
