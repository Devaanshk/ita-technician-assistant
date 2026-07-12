"""
app.py — ITA Technician Assistant

Gradio front-end for the hybrid RAG engine over ITA's vendor manuals
(Lutron, Sonos, Araknis, Josh.ai).

Run:
    python app.py

Then open:
    http://localhost:7860

For a synthesized answer:
  - Set ANTHROPIC_API_KEY before launching, or
  - Run Ollama locally and pull llama3.2:1b
"""

import html as html_lib
import base64
import os

import gradio as gr

from rag_engine import RagEngine


def _img_data_uri(path: str) -> str:
    """Inline a local figure as a base64 data URI so it renders in a
    gr.HTML block without needing a static file server / allowed_paths
    config. Figures are small rasterized crops, so base64 overhead is a
    non-issue here."""
    try:
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
        return f"data:image/png;base64,{data}"
    except Exception:
        return ""


def _figure_strip_html(image_paths):
    """Render a horizontal strip of click-to-zoom figure thumbnails."""
    if not image_paths:
        return ""
    thumbs = ""
    for p in image_paths:
        if not os.path.exists(p):
            continue
        uri = _img_data_uri(p)
        if not uri:
            continue
        thumbs += (
            '<img src="' + uri + '" '
            'style="max-height:140px;border:1px solid #d4e2ea;'
            'border-radius:6px;margin:4px 8px 4px 0;cursor:zoom-in;" '
            "onclick=\"this.style.maxHeight = this.style.maxHeight==='140px' ? 'none' : '140px';\" "
            "/>"
        )
    if not thumbs:
        return ""
    return (
        '<div style="display:flex;flex-wrap:wrap;align-items:flex-start;'
        'margin-top:10px;padding-top:10px;border-top:1px dashed #d4e2ea;">'
        + thumbs + "</div>"
    )


engine = RagEngine()


VENDOR_COLOR = {
    "Lutron": "#c9a227",
    "Sonos": "#2c5364",
    "Josh.ai": "#7a4fb5",
    "Araknis Networks (Snap One)": "#1f7a5c",
}


VENDOR_ICON = {
    "Lutron": "🔆",
    "Sonos": "🔊",
    "Josh.ai": "🎙️",
    "Araknis Networks (Snap One)": "🌐",
}


EXAMPLE_QUESTIONS = [
    "How do I factory reset an Araknis switch?",
    "Sonos speaker won't show up during setup, what should I check?",
    "Josh Core has power but isn't showing up in the app",
    "Customer's Lutron hub lost scheduled lighting scenes but manual control still works",
    "What's the PoE power budget I need to worry about?",
    "Difference between a switch reboot and a full factory reset on Araknis",
]


def _esc(text):
    return html_lib.escape(text or "")


def _vendor_badge(vendor):
    color = VENDOR_COLOR.get(vendor, "#555")
    icon = VENDOR_ICON.get(vendor, "🔧")

    return (
        f'<span style="background:{color}22;'
        f"color:{color};"
        f"border:1px solid {color}55;"
        f"padding:3px 10px;"
        f"border-radius:20px;"
        f"font-size:0.78rem;"
        f"font-weight:600;"
        f'white-space:nowrap;">'
        f"{icon} {_esc(vendor)}"
        f"</span>"
    )


def _step_list_html(steps):
    items = ""

    for i, step in enumerate(steps, 1):
        items += (
            '<div style="display:flex;'
            "gap:12px;"
            "padding:9px 0;"
            "border-bottom:1px solid #e3edf3;"
            'align-items:flex-start;">'
            '<div style="flex:0 0 26px;'
            "height:26px;"
            "border-radius:50%;"
            "background:#0f2027;"
            "color:#c9a227;"
            "font-weight:700;"
            "font-size:0.85rem;"
            "display:flex;"
            "align-items:center;"
            'justify-content:center;">'
            f"{i}"
            "</div>"
            '<div style="padding-top:3px;'
            "color:#1a1a1a;"
            "font-size:0.95rem;"
            'line-height:1.4;">'
            f"{_esc(step)}"
            "</div>"
            "</div>"
        )

    return f'<div style="margin-top:6px;">{items}</div>'


def _card_wrapper(inner_html, title=None, vendor=None):
    header = ""

    if title or vendor:
        badge = _vendor_badge(vendor) if vendor else ""

        header = (
            '<div style="display:flex;'
            "justify-content:space-between;"
            "align-items:center;"
            'margin-bottom:8px;">'
            '<div style="font-weight:700;'
            "font-size:1.02rem;"
            'color:#0f2027;">'
            f"{_esc(title or '')}"
            "</div>"
            f"{badge}"
            "</div>"
        )

    return (
        '<div style="background:#ffffff;'
        "border:1px solid #d4e2ea;"
        "border-left:4px solid #c9a227;"
        "border-radius:8px;"
        "padding:16px 18px;"
        "margin-bottom:14px;"
        'box-shadow:0 2px 6px rgba(15,32,39,0.07);">'
        f"{header}"
        f"{inner_html}"
        "</div>"
    )


def render_answer(result: dict) -> str:
    mode = result.get("mode")

    if mode == "none":
        return _card_wrapper(
            '<div style="color:#666;">'
            "No matching manual content found. "
            "Try rephrasing, or this may not be covered in the current corpus."
            "</div>"
        )

    if mode == "llm_steps":
        parts = _step_list_html(result["steps"])

        if result.get("warning"):
            parts += (
                '<div style="margin-top:12px;'
                "background:#fff4e5;"
                "border:1px solid #f0c36d;"
                "border-radius:6px;"
                "padding:10px 14px;"
                "color:#7a4a00;"
                'font-size:0.88rem;">'
                "⚠️ <strong>Warning:</strong> "
                f"{_esc(result['warning'])}"
                "</div>"
            )

        if result.get("escalate"):
            parts += (
                '<div style="margin-top:10px;'
                "color:#555;"
                'font-size:0.85rem;">'
                "<strong>If that doesn’t fix it:</strong> "
                f"{_esc(result['escalate'])}"
                "</div>"
            )

        backend_tag = {
            "anthropic": "Claude",
            "ollama": "Local Ollama",
        }.get(result.get("backend"), "")

        footer = ""

        if backend_tag:
            footer = (
                '<div style="margin-top:12px;'
                "font-size:0.75rem;"
                'color:#999;">'
                f"Generated via {backend_tag}"
                "</div>"
            )

        return _card_wrapper(
            parts + footer,
            title=result.get("issue"),
            vendor=result.get("vendor"),
        )

    if mode == "llm_raw":
        return _card_wrapper(
            '<div style="white-space:pre-wrap;'
            "color:#1a1a1a;"
            'font-size:0.92rem;">'
            f"{_esc(result['text'])}"
            "</div>"
        )

    if mode == "extractive":
        output = (
            '<div style="background:#e8f2f8;'
            "border:1px solid #c5dce9;"
            "border-radius:6px;"
            "padding:9px 14px;"
            "margin-bottom:12px;"
            "color:#2c5364;"
            'font-size:0.82rem;">'
            "ℹ️ No LLM connected — showing matched manual sections directly. "
            "Run Ollama locally or set ANTHROPIC_API_KEY for a synthesized "
            "step-by-step answer."
            "</div>"
        )

        for card in result["cards"]:
            card_figures = _figure_strip_html(card.get("image_paths", []))
            if card["type"] == "steps":
                inner = _step_list_html(card["steps"]) + card_figures

            elif card["type"] == "qa":
                inner = ""

                for question, answer in card["pairs"]:
                    inner += (
                        '<div style="margin-bottom:10px;">'
                        '<div style="font-weight:600;'
                        "color:#0f2027;"
                        'font-size:0.9rem;">'
                        f"Q: {_esc(question)}"
                        "</div>"
                        '<div style="color:#333;'
                        "font-size:0.9rem;"
                        'margin-top:2px;">'
                        f"A: {_esc(answer)}"
                        "</div>"
                        "</div>"
                    )
                inner += card_figures

            else:
                inner = (
                    '<div style="color:#333;'
                    "font-size:0.9rem;"
                    'line-height:1.5;">'
                    f"{_esc(card['text'])}"
                    "</div>"
                    + card_figures
                )

            output += _card_wrapper(
                inner,
                title=card["title"],
                vendor=card["vendor"],
            )

        return output

    return _card_wrapper(
        '<div style="color:#666;">Unexpected response format.</div>'
    )


def format_sources_html(contexts):
    if not contexts:
        return (
            '<div style="color:#7990a0;font-size:0.85rem;">'
            "Sources will appear here after you ask a question."
            "</div>"
        )

    rows = ""

    for context in contexts:
        color = VENDOR_COLOR.get(context["vendor"], "#555")
        figure_html = _figure_strip_html(context.get("image_paths", []))

        rows += (
            '<div style="padding:8px 0;'
            'border-bottom:1px solid #dce8ef;">'
            '<div style="font-size:0.85rem;'
            "font-weight:600;"
            'color:#0f2027;">'
            f"{_esc(context['section_title'])}"
            "</div>"
            '<div style="font-size:0.75rem;'
            f"color:{color};"
            'margin-top:2px;">'
            f"{_esc(context['vendor'])}"
            '<span style="color:#7c8f9b;">'
            f" · relevance {context['score']:.2f}"
            f" · {_esc(context['source_file'])}"
            "</span>"
            "</div>"
            f"{figure_html}"
            "</div>"
        )

    return rows


def respond(message, history):
    if not message or not message.strip():
        placeholder = _card_wrapper(
            '<div style="color:#666;">'
            "Ask about a Lutron, Sonos, Araknis, or Josh.ai "
            "install/support issue above."
            "</div>"
        )

        return placeholder, format_sources_html([])

    result, contexts = engine.answer(message)

    return render_answer(result), format_sources_html(contexts)

CUSTOM_CSS = """
:root, .dark {
    color-scheme: light !important;
    --body-background-fill: #eef4f8 !important;
    --background-fill-primary: #eef4f8 !important;
    --background-fill-secondary: #f7fbfd !important;
    --block-background-fill: #ffffff !important;
    --body-text-color: #1a1a1a !important;
    --body-text-color-subdued: #555555 !important;
    --block-label-text-color: #0f2027 !important;
    --block-title-text-color: #0f2027 !important;
    --border-color-primary: #d4e2ea !important;
    --input-background-fill: #ffffff !important;
    --button-secondary-background-fill: #ffffff !important;
    --button-secondary-text-color: #0f2027 !important;
}

body,
.gradio-container {
    background: #eef4f8 !important;
    color: #1a1a1a !important;
    font-family: "Segoe UI", Roboto, sans-serif;
}

.gradio-container * {
    color: #1a1a1a;
}

.gradio-container .block,
.gradio-container .form,
.gradio-container .panel {
    border-color: #d4e2ea !important;
}

label,
.label-wrap span {
    color: #0f2027 !important;
    font-weight: 600 !important;
}

textarea,
input {
    background: #ffffff !important;
    color: #1a1a1a !important;
    border-color: #b8ceda !important;
}

textarea::placeholder,
input::placeholder {
    color: #7d929f !important;
}

textarea:hover,
input:hover {
    background: #f4f9fc !important;
    border-color: #c9a227 !important;
}

textarea:focus,
input:focus {
    background: #ffffff !important;
    border-color: #c9a227 !important;
    box-shadow: 0 0 0 2px #c9a22733 !important;
}

/* Example question buttons */
#example-questions button,
#example-questions .gallery-item,
#example-questions .example,
#example-questions table tr,
#example-questions table td {
    background: #ffffff !important;
    color: #1a1a1a !important;
    border-color: #c7dce8 !important;
    transition:
        background-color 0.2s ease,
        border-color 0.2s ease,
        color 0.2s ease !important;
}

#example-questions button:hover,
#example-questions .gallery-item:hover,
#example-questions .example:hover,
#example-questions table tr:hover,
#example-questions table td:hover {
    background: #cfe8f6 !important;
    color: #0f2027 !important;
    border-color: #8fbfd6 !important;
}

#example-questions button:hover *,
#example-questions .gallery-item:hover *,
#example-questions .example:hover *,
#example-questions table tr:hover *,
#example-questions table td:hover * {
    background: transparent !important;
    color: #0f2027 !important;
}

#ita-header,
#ita-header * {
    color: #ffffff !important;
}

#ita-header .tagline {
    color: #c9a227 !important;
}

#ita-header {
    background: linear-gradient(
        120deg,
        #0b141c 0%,
        #0f2027 45%,
        #203a43 100%
    );
    padding: 30px 32px;
    border-radius: 14px;
    margin-bottom: 18px;
    border: 1px solid #c9a22755;
    box-shadow: 0 5px 18px rgba(15, 32, 39, 0.16);
}

#ita-header .tagline {
    letter-spacing: 0.06em;
    font-size: 0.78rem;
    text-transform: uppercase;
    font-weight: 600;
    margin-bottom: 6px;
}

#ita-header h1 {
    margin: 0 0 6px 0;
    font-size: 1.65rem;
    font-weight: 700;
}

#ita-header p {
    margin: 0;
    opacity: 0.85;
    font-size: 0.88rem;
}

#ask-btn,
#ask-btn * {
    background: #0f2027 !important;
    border: 1px solid #c9a227 !important;
    color: #c9a227 !important;
    font-weight: 700 !important;
}

#ask-btn:hover,
#ask-btn:hover * {
    background: #c9a227 !important;
    color: #0f2027 !important;
}

#sidebar-info,
#sidebar-info * {
    color: #333333 !important;
}
"""


with gr.Blocks(title="ITA Technician Assistant") as demo:
    gr.HTML(
        """
        <div id="ita-header">
            <div class="tagline">
                Innovative Technologies
            </div>

            <h1>
                &#128295; ITA Technician Assistant
            </h1>

            <p>
                Capturing Expectations, Delivering Excellence&trade;
                &mdash; now for support calls too.
                Cited, step-by-step answers from the Lutron, Sonos,
                Araknis &amp; Josh.ai manuals.
            </p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            question = gr.Textbox(
                label="Ask a technician question",
                placeholder="e.g. How do I factory reset an Araknis switch?",
                lines=2,
            )

            ask_btn = gr.Button(
                "Ask",
                elem_id="ask-btn",
            )

            answer_html = gr.HTML(
                value=(
                    '<div style="color:#7d929f;">'
                    "Answer will appear here."
                    "</div>"
                )
            )

            gr.Examples(
            examples=EXAMPLE_QUESTIONS,
            inputs=question,
            label="Try one of these",
            elem_id="example-questions",
            )
        with gr.Column(scale=2):
            gr.HTML(
                '<div style="font-weight:700;'
                "color:#0f2027;"
                "font-size:1.05rem;"
                'margin-bottom:6px;">'
                "📚 Sources used"
                "</div>"
            )

            sources_html = gr.HTML(
                value=format_sources_html([])
            )

            gr.HTML(
                '<div id="sidebar-info" '
                'style="margin-top:14px;'
                "padding-top:14px;"
                "border-top:1px solid #cbdde7;"
                "font-size:0.85rem;"
                "color:#333;"
                'line-height:1.6;">'
                '<strong style="color:#0f2027;">Corpus:</strong> '
                "Lutron Caseta &middot; Sonos &middot; "
                "Araknis Networks &middot; Josh.ai"
                "<br>"
                '<strong style="color:#0f2027;">Retrieval:</strong> '
                "hybrid BM25 (lexical) + TF-IDF/LSA (semantic)"
                "<br>"
                '<strong style="color:#0f2027;">Generation:</strong> '
                "Anthropic API &rarr; local Ollama &rarr; "
                "direct cited excerpts, in that order, "
                "whichever is available"
                "</div>"
            )

    ask_btn.click(
        fn=respond,
        inputs=[
            question,
            gr.State([]),
        ],
        outputs=[
            answer_html,
            sources_html,
        ],
    )

    question.submit(
        fn=respond,
        inputs=[
            question,
            gr.State([]),
        ],
        outputs=[
            answer_html,
            sources_html,
        ],
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        css=CUSTOM_CSS,
    )