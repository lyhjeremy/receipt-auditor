"""The four PII layers, wired together. See RECEIPT_AUDITOR_SPEC.md §2.

Layer 1: schema-as-guardrail (schemas.py Receipt has no PII-holding fields --
nothing to wire here, the absence IS the control).
Layer 2: raw OCR text is never kept -- only the validated Receipt object
survives past extraction (enforced by the caller: app.py never stores
vision.extract's raw response, only the parsed model).
Layer 3: belt-and-braces sweep (this module).
Layer 4: persistence policy (this module + app.py: Space = session-only).
"""
from __future__ import annotations

from guardrails import redact_pii
from schemas import Receipt


def sweep_receipt(receipt: Receipt) -> tuple[Receipt, int]:
    """Layer 3: run redact_pii over every string field before the receipt
    touches cache/persistence. Returns (clean_receipt, n_redactions) -- the
    count is surfaced in the app's dev panel, not hidden.
    """
    total_redactions = 0
    clean_merchant, n = redact_pii(receipt.merchant)
    total_redactions += n

    clean_items = []
    for item in receipt.line_items:
        clean_desc, n = redact_pii(item.desc)
        total_redactions += n
        clean_items.append(item.model_copy(update={"desc": clean_desc}))

    return receipt.model_copy(update={"merchant": clean_merchant, "line_items": clean_items}), total_redactions


def is_space_environment() -> bool:
    """Layer 4: on a hosted Space, receipts are session-only (in-memory),
    never written to disk. Locally, a git-ignored ledger opt-in is allowed
    (see app.py). This function is the single source of truth both paths
    check before persisting anything.
    """
    import os
    return bool(os.environ.get("SPACE_ID"))
