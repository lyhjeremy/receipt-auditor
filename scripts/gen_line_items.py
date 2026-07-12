"""Overnight resumable synthetic corpus generator for the spend-category
classifier. See RECEIPT_AUDITOR_SPEC.md §7.2. ~4-6k labeled line items across
realistic POS abbreviation styles ("SBUX #4821", "WM SUPERCENTER").

Jeremy's 10-20 real receipts (once supplied) become a SEPARATE held-out test
set, never included in this synthetic training corpus -- see
training/prep_categorizer.py.

Run under caffeinate:
  caffeinate -i python scripts/gen_line_items.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gen_data import DatasetGenerator

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "line_items.jsonl"

CATEGORIES = [
    "groceries", "dining", "coffee_snacks", "transport", "fuel", "health",
    "household", "clothing", "entertainment", "subscriptions_utilities",
    "travel", "other",
]

# Realistic merchant archetypes per category -- POS receipts abbreviate
# aggressively and inconsistently; the model needs to see that variety.
MERCHANT_STYLES = [
    "ALL CAPS with store number (e.g. 'WM SUPERCENTER #2847')",
    "abbreviated brand + location code (e.g. 'SBUX #4821', 'TGT 1147')",
    "POS chain prefix (e.g. 'TST* JOES PIZZA', 'SQ *CORNER CAFE')",
    "full readable business name (e.g. 'Corner Bakery Cafe')",
    "generic/local merchant with a city name appended",
]

N_PER_CATEGORY = 400  # 12 categories x 400 = ~4800 rows


def _stable_id(*parts) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def build_items() -> list[dict]:
    items = []
    for category in CATEGORIES:
        for i in range(N_PER_CATEGORY):
            style = MERCHANT_STYLES[i % len(MERCHANT_STYLES)]
            items.append({"id": _stable_id(category, i), "category": category, "seq": i, "style": style})
    return items


def build_prompt(item: dict) -> str:
    return f"""Invent one realistic receipt line item for the category "{item['category']}".
The merchant name style should be: {item['style']}.

Respond with ONLY JSON: {{"merchant": "...", "desc": "... (the line-item description as it
would appear on a real receipt, e.g. 'GROC ITEM 4823' or 'Latte Grande')",
"category": "{item['category']}"}}"""


def parse(raw: str) -> dict:
    return json.loads(raw)


def validate(parsed: dict) -> list[str]:
    violations = []
    if not parsed.get("merchant"):
        violations.append("missing merchant")
    if not parsed.get("desc"):
        violations.append("missing desc")
    if parsed.get("category") not in CATEGORIES:
        violations.append(f"invalid category '{parsed.get('category')}'")
    return violations


def main():
    items = build_items()
    print(f"Total items to generate: {len(items)}")

    def generate_fn(prompt: str) -> str:
        import llm
        return llm.generate(prompt, tier="fast", json_only=True, max_tokens=200).text

    gen = DatasetGenerator(
        name="line_items", out_path=OUT_PATH, items=items,
        build_prompt=build_prompt, parse=parse, validate=validate, generate_fn=generate_fn,
    )
    gen.run(max_consecutive_failures=3, sleep_between=0.5)


if __name__ == "__main__":
    main()
