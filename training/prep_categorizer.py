"""Line-item -> category classifier: dataset prep. See
RECEIPT_AUDITOR_SPEC.md §7.2.

Synthetic corpus (scripts/gen_line_items.py) trains the model. Jeremy's real
receipts (photographed, vision-extracted, HAND-VERIFIED by him) become a
separate held-out test set that is NEVER trained on -- the synthetic-to-real
accuracy gap is itself an honest, reportable result, not something to hide.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training" / "lora_harness"))

from prep import prep_dataset

SYNTHETIC_PATH = Path(__file__).resolve().parent.parent / "data" / "line_items.jsonl"
REAL_RECEIPTS_PATH = Path(__file__).resolve().parent.parent / "data" / "real_receipts_labeled.jsonl"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "lora"
EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"


def load_synthetic_rows() -> list[dict]:
    if not SYNTHETIC_PATH.exists():
        raise SystemExit(
            f"{SYNTHETIC_PATH} not found -- run scripts/gen_line_items.py first "
            "(queued behind other claude -p jobs as of Session 1; see roadmap)."
        )
    rows = []
    for line in SYNTHETIC_PATH.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_real_receipts_test_set() -> list[dict]:
    """Returns [] if Jeremy hasn't supplied labeled real receipts yet --
    the benchmark then reports 'no real-receipt column available' honestly
    rather than faking one. See RECEIPT_AUDITOR_SPEC.md §2 (open item)."""
    if not REAL_RECEIPTS_PATH.exists():
        return []
    return [json.loads(l) for l in REAL_RECEIPTS_PATH.read_text().splitlines() if l.strip()]


def to_messages(rec: dict) -> list[dict]:
    prompt = f"Merchant: {rec['merchant']}. Line item: {rec['desc']}. Which category is this?"
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": rec["category"]},
    ]


def build_dataset() -> dict:
    rows = load_synthetic_rows()
    print(f"{len(rows)} synthetic rows loaded")

    card = prep_dataset(
        rows,
        entity_key_fn=lambda r: r["merchant"],  # split by merchant, not row
        to_messages_fn=to_messages,
        out_dir=OUT_DIR,
        label_key="category",
    )

    real_test = load_real_receipts_test_set()
    card["real_receipts_held_out"] = len(real_test)
    if not real_test:
        print("No labeled real receipts yet -- the benchmark will report synthetic-only "
              "until Jeremy supplies + labels 10-20 real receipt photos (roadmap §2).")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "dataset_card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False))
    return card


if __name__ == "__main__":
    print(json.dumps(build_dataset(), indent=2))
