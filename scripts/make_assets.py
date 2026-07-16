"""Banner + architecture diagram for the showcase page. matplotlib (not SVG)
-- matches _ai-gap-toolkit's documented reasoning (qlmanage crops SVG, no
libcairo for cairosvg). Money-green palette, distinct from sibling projects."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "assets"
MONEY = "#1F4D3A"
SLIP = "#3A5A46"
GOLD = "#B89340"
CREAM = "#F6F8F4"


def make_banner():
    fig, ax = plt.subplots(figsize=(12, 3), dpi=150)
    ax.set_xlim(0, 12); ax.set_ylim(0, 3); ax.axis("off")
    fig.patch.set_facecolor("#12241a")
    ax.set_facecolor("#12241a")
    ax.text(6, 1.9, "Receipt Auditor", ha="center", va="center", fontsize=42,
             color=CREAM, family="serif", weight="bold")
    ax.text(6, 1.1, "Photograph receipts. Get reconciled spending and a locally fine-tuned category guess.",
             ha="center", va="center", fontsize=14, color="#D8E3D6", family="serif", style="italic")
    fig.tight_layout()
    fig.savefig(OUT / "banner.png", facecolor=fig.get_facecolor())
    plt.close(fig)


def _box(ax, x, y, w, h, text, color):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.08",
                          facecolor=color, edgecolor="none")
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10.5,
             color="white", family="sans-serif", weight="bold", wrap=True)


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                  color="#7a8a80", linewidth=1.5))


def make_architecture():
    fig, ax = plt.subplots(figsize=(10.5, 5.5), dpi=150)
    ax.set_xlim(0, 10.5); ax.set_ylim(0, 5.5); ax.axis("off")
    fig.patch.set_facecolor(CREAM)

    _box(ax, 0.3, 4.3, 2.0, 0.8, "Receipt photo", MONEY)
    _box(ax, 2.9, 4.3, 2.2, 0.8, "Local OCR\n(tesseract)", SLIP)
    _box(ax, 5.7, 4.3, 2.2, 0.8, "claude -p\nextract + validate", GOLD)
    _box(ax, 8.4, 4.3, 1.9, 0.8, "Receipt", MONEY)

    _arrow(ax, 2.3, 4.7, 2.9, 4.7)
    _arrow(ax, 5.1, 4.7, 5.7, 4.7)
    _arrow(ax, 7.9, 4.7, 8.4, 4.7)

    _box(ax, 8.4, 2.8, 1.9, 0.8, "PII sweep\n+ redact", SLIP)
    _arrow(ax, 9.35, 4.3, 9.35, 3.6)

    _box(ax, 5.7, 2.8, 2.2, 0.8, "Reconciliation\n(plain code)", MONEY)
    _arrow(ax, 8.4, 3.2, 7.9, 3.2)

    _box(ax, 2.9, 2.8, 2.2, 0.8, "LoRA (Qwen2.5-1.5B)\ncategorize items", GOLD)
    _arrow(ax, 5.7, 3.2, 5.1, 3.2)

    _box(ax, 0.3, 1.0, 2.4, 0.8, "Spend question\n(voice or text)", MONEY)
    _box(ax, 3.2, 1.0, 2.6, 0.8, "pandas computes\nLLM only phrases", SLIP)
    _box(ax, 6.3, 1.0, 2.3, 0.8, "Narration guard\nverifies numbers", GOLD)
    _arrow(ax, 2.7, 1.4, 3.2, 1.4)
    _arrow(ax, 5.8, 1.4, 6.3, 1.4)

    ax.text(0.3, 0.3, "Guardrails: domain gate · reconciliation · 4-layer PII · narration guard · semantic cache",
            fontsize=9, color="#5f6f66", family="sans-serif", style="italic")

    fig.tight_layout()
    fig.savefig(OUT / "architecture.png", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    make_banner()
    make_architecture()
    print("Wrote banner.png + architecture.png to", OUT)
