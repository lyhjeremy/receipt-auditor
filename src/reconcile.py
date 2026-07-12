"""Deterministic totals reconciliation + the narration guard. See
RECEIPT_AUDITOR_SPEC.md §3. Both run in plain code -- never delegated to the
model, so they can't be fooled by a confidently wrong LLM output.
"""
from __future__ import annotations

import re

from schemas import Receipt, Reconciliation


def reconcile(receipt: Receipt) -> Reconciliation:
    """Computed total = sum(line items) + tax (subtotal is informational
    only, not trusted). status='mismatch' if it doesn't match the printed
    total within tolerance -- the row then needs a human's one-tap correction,
    never a silent "fix" by the model.
    """
    computed = sum(item.amount * item.qty for item in receipt.line_items)
    if receipt.tax:
        computed += receipt.tax

    tolerance = max(0.02, 0.005 * receipt.total)
    delta = round(computed - receipt.total, 2)
    status = "ok" if abs(delta) <= tolerance else "mismatch"

    return Reconciliation(
        computed_total=round(computed, 2), printed_total=receipt.total,
        delta=delta, status=status,
    )


def apply_reconciliation(receipt: Receipt) -> Receipt:
    return receipt.model_copy(update={"reconciliation": reconcile(receipt)})


def assert_numbers_present(sentence: str, required_numbers: list[float]) -> bool:
    """The narration guard: after the LLM phrases a computed answer, assert
    every number it was given (computed by pandas, not the model) appears
    verbatim in the output sentence. If not, the caller must fall back to a
    template sentence -- a hallucinated number is structurally impossible to
    ship, because the number never came from the model in the first place;
    this only catches the model DROPPING or ALTERING a number while phrasing.
    """
    # must START with a digit -- "[\d,]+" alone can match a bare trailing
    # comma (e.g. the "," right after "$120.00," in a sentence), producing an
    # empty string after stripping commas and crashing float(). Caught by a
    # test using a realistic multi-number sentence with a comma-separated list.
    found_numbers = {
        float(m.replace(",", "")) for m in re.findall(r"\d[\d,]*\.?\d*", sentence)
    }
    for n in required_numbers:
        # tolerate reasonable rounding (e.g. $42.00 vs $42) but not a changed value
        if not any(abs(n - f) < 0.01 for f in found_numbers):
            return False
    return True


def template_answer(query_desc: str, value: float, unit: str = "$") -> str:
    return f"{query_desc}: {unit}{value:,.2f}"
