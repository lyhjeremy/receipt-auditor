"""Pydantic contracts for Receipt Auditor. See RECEIPT_AUDITOR_SPEC.md §1.

Note the schema itself is a guardrail: it has NO fields for cardholder name,
card number, address, or loyalty id, so the vision model is never even asked
to extract them. The cheapest PII control is a schema that can't hold PII.
"""
from __future__ import annotations

import datetime
from typing import Literal

from pydantic import BaseModel

Category = Literal[
    "groceries", "dining", "coffee_snacks", "transport", "fuel", "health",
    "household", "clothing", "entertainment", "subscriptions_utilities",
    "travel", "other",
]


class LineItem(BaseModel):
    desc: str
    qty: float = 1
    amount: float
    category: Category | None = None


class Reconciliation(BaseModel):
    computed_total: float
    printed_total: float
    delta: float
    status: Literal["ok", "mismatch"]


class Receipt(BaseModel):
    merchant: str
    date: datetime.date | None = None
    currency: str = "USD"
    line_items: list[LineItem]
    subtotal: float | None = None
    tax: float | None = None
    total: float
    extraction_confidence: float
    reconciliation: Reconciliation | None = None  # computed in code, never by the model


class Query(BaseModel):
    metric: Literal["sum", "count", "avg", "top_merchants", "by_category"]
    category: Category | None = None
    merchant: str | None = None
    date_from: datetime.date | None = None
    date_to: datetime.date | None = None
