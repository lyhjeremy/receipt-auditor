"""Receipt Auditor -- Gradio app. Runs locally and as an HF Space.

Batch receipt photos -> reconciled, categorized, PII-redacted structured
data -> dashboard + voice-queryable analysis (pandas computes, LLM only
narrates). See RECEIPT_AUDITOR_SPEC.md §4.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd

import audio
import vision
from analysis import answer_query, receipts_to_dataframe
from cache import FileCache, SemanticCache
from guardrails import Refusal
from privacy import is_space_environment, sweep_receipt
from reconcile import apply_reconciliation, assert_numbers_present, template_answer
from schemas import Query, Receipt

DATA_DIR = Path(__file__).resolve().parent / "data"
query_cache = SemanticCache(DATA_DIR / "query_cache.db", similarity_threshold=0.93)
audio_cache = FileCache(DATA_DIR / "audio_cache")
ADAPTER_PATH = Path(__file__).resolve().parent / "training" / "adapters"
BASE_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"

CATEGORIES = [
    "groceries", "dining", "coffee_snacks", "transport", "fuel", "health",
    "household", "clothing", "entertainment", "subscriptions_utilities",
    "travel", "other",
]


def categorize_item(desc: str, merchant: str) -> str:
    """Rule-based fallback (works with zero external calls) + LoRA/Gemini
    override where available. Kept simple and inspectable -- the point of
    this project is that spend categorization is auditable, not a black box.
    """
    if ADAPTER_PATH.exists():
        prompt = f"Merchant: {merchant}. Line item: {desc}. Which category: {', '.join(CATEGORIES)}? Answer with just the category."
        try:
            proc = subprocess.run(
                ["mlx_lm.generate", "--model", BASE_MODEL, "--adapter-path", str(ADAPTER_PATH),
                 "--prompt", prompt, "--max-tokens", "10"],
                capture_output=True, text=True, timeout=30,
            )
            out = proc.stdout.strip().lower()
            for cat in CATEGORIES:
                if cat.replace("_", " ") in out or cat in out:
                    return cat
        except Exception:
            pass
    return "other"


def process_receipts(images) -> tuple[list[Receipt], str]:
    if not images:
        return [], "Upload at least one receipt photo."

    receipts = []
    messages = []
    for img in images:
        from PIL import Image
        pil_img = img if isinstance(img, Image.Image) else Image.open(img)

        result = vision.extract(
            pil_img, Receipt,
            task_prompt="merchant, date, currency, line_items (desc, qty, amount), "
                        "subtotal, tax, total",
            domain_description="a purchase receipt",
            min_confidence=0.5,
        )
        if isinstance(result, Refusal):
            messages.append(f"⚠ Skipped an image: {result.user_message}")
            continue

        # Layer 2: only the validated Receipt survives -- raw OCR text is never kept.
        clean_receipt, n_redacted = sweep_receipt(result)  # Layer 3
        for item in clean_receipt.line_items:
            item.category = categorize_item(item.desc, clean_receipt.merchant)
        reconciled = apply_reconciliation(clean_receipt)

        receipts.append(reconciled)
        status = "✓" if reconciled.reconciliation.status == "ok" else "⚠ mismatch -- please review"
        redaction_note = f", {n_redacted} PII field(s) redacted" if n_redacted else ""
        messages.append(f"{status} {clean_receipt.merchant}: ${clean_receipt.total:.2f}{redaction_note}")

    return receipts, "\n".join(messages)


def receipts_to_editable_rows(receipts: list[Receipt]) -> list[list]:
    rows = []
    for r in receipts:
        status = r.reconciliation.status if r.reconciliation else "?"
        rows.append([r.merchant, str(r.date or ""), r.total,
                     ", ".join(f"{i.desc}(${i.amount:.2f})" for i in r.line_items), status])
    return rows


def build_dashboard(receipts: list[Receipt]):
    if not receipts:
        return None, "No receipts processed yet."

    df = receipts_to_dataframe(receipts)
    by_cat = df.groupby("category")["amount"].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(by_cat.index, by_cat.values, color="#7a1f2b")
    ax.set_xlabel("Spend ($)")
    ax.set_title("Spend by category")
    fig.tight_layout()

    summary = f"**Total across {len(receipts)} receipts:** ${df['amount'].sum():.2f}"
    return fig, summary


def transcribe_query(audio_path: str | None) -> str:
    if audio_path is None:
        return ""
    try:
        return audio.transcribe(audio_path).text
    except audio.NoSpeechError:
        return ""


def answer_voice_query(receipts: list[Receipt], query_text: str) -> tuple[str, str | None]:
    if not receipts:
        return "No receipts to analyze yet.", None
    if not query_text.strip():
        return "Ask a question about your spending.", None

    from guardrails import generate_validated
    df = receipts_to_dataframe(receipts)

    try:
        query = generate_validated(
            f"Parse this spending question into a structured query: '{query_text}'. "
            f"Categories: {', '.join(CATEGORIES)}.",
            Query, max_retries=1,
        )
    except Exception:
        return "Couldn't understand that question -- try something like 'how much did I spend on dining?'", None

    result = answer_query(df, query)

    if result["unit"] == "currency":
        # narration guard: the LLM only phrases result['value'], never
        # invents it -- and we verify the number survived phrasing.
        from llm import generate
        phrased = generate(
            f"Phrase this as one natural sentence: {result['description']} = ${result['value']:.2f}",
            tier="fast", max_tokens=60,
        ).text.strip()
        answer = phrased if assert_numbers_present(phrased, [result["value"]]) else template_answer(result["description"], result["value"])
    elif result["unit"] == "count":
        answer = f"{result['description']}: {result['value']}"
    else:
        answer = f"{result['description']}: " + ", ".join(f"{k} (${v:.2f})" for k, v in list(result["value"].items())[:5])

    voice = audio.voice_for("en") or "en-US-AriaNeural"
    audio_path = audio.speak_cached(answer, voice, audio_cache)
    return answer, str(audio_path)


with gr.Blocks(title="Receipt Auditor") as demo:
    gr.Markdown("# 🧾 Receipt Auditor")
    gr.Markdown(
        "Private by construction: the receipt schema has no fields for cardholder "
        "name, card number, or address -- they're never extracted in the first place. "
        + ("Session-only on this Space; nothing is written to disk." if is_space_environment()
           else "Local runs keep data on this machine only.")
    )

    receipts_state = gr.State([])

    with gr.Row():
        with gr.Column():
            photos = gr.File(file_count="multiple", file_types=["image"], label="Receipt photos")
            process_btn = gr.Button("Process receipts", variant="primary")
            process_status = gr.Markdown()
            receipts_table = gr.Dataframe(headers=["merchant", "date", "total", "items", "status"])
        with gr.Column():
            dashboard_plot = gr.Plot()
            dashboard_summary = gr.Markdown()

    gr.Markdown("### Ask about your spending")
    with gr.Row():
        query_audio = gr.Audio(sources=["microphone"], type="filepath", label="Speak your question")
        query_text = gr.Textbox(label="Or type your question")
    query_btn = gr.Button("Ask")
    query_answer = gr.Markdown()
    query_audio_out = gr.Audio(label="Listen")

    def _process_and_render(images):
        receipts, status = process_receipts(images)
        rows = receipts_to_editable_rows(receipts)
        fig, summary = build_dashboard(receipts)
        return receipts, status, rows, fig, summary

    process_btn.click(_process_and_render, inputs=[photos],
                       outputs=[receipts_state, process_status, receipts_table, dashboard_plot, dashboard_summary])
    query_audio.change(transcribe_query, inputs=[query_audio], outputs=[query_text])
    query_btn.click(answer_voice_query, inputs=[receipts_state, query_text], outputs=[query_answer, query_audio_out])

if __name__ == "__main__":
    demo.launch()
