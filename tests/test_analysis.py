import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analysis import answer_query, receipts_to_dataframe
from schemas import LineItem, Query, Receipt


def make_receipts():
    return [
        Receipt(
            merchant="Starbucks", date=datetime.date(2026, 6, 1),
            line_items=[LineItem(desc="Latte", amount=5.00, category="coffee_snacks")],
            total=5.00, extraction_confidence=0.9,
        ),
        Receipt(
            merchant="Whole Foods", date=datetime.date(2026, 6, 15),
            line_items=[
                LineItem(desc="Bread", amount=3.00, category="groceries"),
                LineItem(desc="Milk", amount=2.00, category="groceries"),
            ],
            total=5.00, extraction_confidence=0.9,
        ),
        Receipt(
            merchant="Starbucks", date=datetime.date(2026, 7, 1),
            line_items=[LineItem(desc="Latte", amount=5.50, category="coffee_snacks")],
            total=5.50, extraction_confidence=0.9,
        ),
    ]


def test_sum_by_category():
    df = receipts_to_dataframe(make_receipts())
    result = answer_query(df, Query(metric="sum", category="coffee_snacks"))
    assert result["value"] == 10.50


def test_sum_by_merchant():
    df = receipts_to_dataframe(make_receipts())
    result = answer_query(df, Query(metric="sum", merchant="Whole Foods"))
    assert result["value"] == 5.00


def test_sum_by_date_range():
    df = receipts_to_dataframe(make_receipts())
    result = answer_query(df, Query(
        metric="sum", date_from=datetime.date(2026, 6, 1), date_to=datetime.date(2026, 6, 30),
    ))
    assert result["value"] == 10.00  # excludes the July Starbucks purchase


def test_count():
    df = receipts_to_dataframe(make_receipts())
    result = answer_query(df, Query(metric="count", category="groceries"))
    assert result["value"] == 2


def test_avg():
    df = receipts_to_dataframe(make_receipts())
    result = answer_query(df, Query(metric="avg", category="coffee_snacks"))
    assert result["value"] == 5.25


def test_by_category_breakdown():
    df = receipts_to_dataframe(make_receipts())
    result = answer_query(df, Query(metric="by_category"))
    assert result["value"]["groceries"] == 5.00
    assert result["value"]["coffee_snacks"] == 10.50


def test_top_merchants():
    df = receipts_to_dataframe(make_receipts())
    result = answer_query(df, Query(metric="top_merchants"))
    merchants_by_spend = list(result["value"].keys())
    assert merchants_by_spend[0] == "Starbucks"  # 10.50 > 5.00
