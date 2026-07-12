"""Deterministic spend analysis. The LLM never computes a number -- it only
phrases an answer pandas already calculated, and the narration guard
(reconcile.assert_numbers_present) verifies the number survived phrasing
intact. See RECEIPT_AUDITOR_SPEC.md §4.
"""
from __future__ import annotations

import pandas as pd

from schemas import Query, Receipt


def receipts_to_dataframe(receipts: list[Receipt]) -> pd.DataFrame:
    rows = []
    for r in receipts:
        for item in r.line_items:
            rows.append({
                "merchant": r.merchant, "date": r.date, "currency": r.currency,
                "desc": item.desc, "amount": item.amount * item.qty,
                "category": item.category or "other",
            })
    return pd.DataFrame(rows)


def answer_query(df: pd.DataFrame, query: Query) -> dict:
    """Returns {value, unit, description} -- computed entirely in pandas.
    The caller passes `value` to the LLM only for phrasing, never for
    computation.
    """
    filtered = df
    if query.category:
        filtered = filtered[filtered["category"] == query.category]
    if query.merchant:
        filtered = filtered[filtered["merchant"].str.contains(query.merchant, case=False, na=False)]
    if query.date_from:
        filtered = filtered[filtered["date"] >= query.date_from]
    if query.date_to:
        filtered = filtered[filtered["date"] <= query.date_to]

    if query.metric == "sum":
        value = float(filtered["amount"].sum())
        desc = f"Total spend{_scope_desc(query)}"
        return {"value": value, "unit": "currency", "description": desc}

    if query.metric == "count":
        value = int(len(filtered))
        return {"value": value, "unit": "count", "description": f"Number of purchases{_scope_desc(query)}"}

    if query.metric == "avg":
        value = float(filtered["amount"].mean()) if len(filtered) else 0.0
        return {"value": value, "unit": "currency", "description": f"Average purchase{_scope_desc(query)}"}

    if query.metric == "top_merchants":
        top = filtered.groupby("merchant")["amount"].sum().sort_values(ascending=False)
        return {"value": top.to_dict(), "unit": "breakdown", "description": "Top merchants by spend"}

    if query.metric == "by_category":
        breakdown = filtered.groupby("category")["amount"].sum().sort_values(ascending=False)
        return {"value": breakdown.to_dict(), "unit": "breakdown", "description": "Spend by category"}

    raise ValueError(f"Unknown metric: {query.metric}")


def _scope_desc(query: Query) -> str:
    parts = []
    if query.category:
        parts.append(f" on {query.category}")
    if query.merchant:
        parts.append(f" at {query.merchant}")
    if query.date_from or query.date_to:
        parts.append(f" ({query.date_from or '...'} to {query.date_to or '...'})")
    return "".join(parts)
