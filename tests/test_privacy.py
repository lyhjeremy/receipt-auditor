import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from privacy import sweep_receipt
from schemas import LineItem, Receipt


def test_sweep_redacts_pii_in_line_items():
    receipt = Receipt(
        merchant="Test Store",
        line_items=[LineItem(desc="Card ending 4111111111111111 surcharge", amount=1.00)],
        total=1.00, extraction_confidence=0.9,
    )
    clean, n = sweep_receipt(receipt)
    assert n >= 1
    assert "4111111111111111" not in clean.line_items[0].desc


def test_sweep_leaves_clean_receipt_untouched():
    receipt = Receipt(
        merchant="Corner Cafe",
        line_items=[LineItem(desc="Latte", amount=4.50)],
        total=4.50, extraction_confidence=0.95,
    )
    clean, n = sweep_receipt(receipt)
    assert n == 0
    assert clean.merchant == "Corner Cafe"
    assert clean.line_items[0].desc == "Latte"


def test_sweep_does_not_mutate_original():
    receipt = Receipt(
        merchant="jeremy@example.com Store",  # contrived, but tests immutability
        line_items=[LineItem(desc="Item", amount=1.00)],
        total=1.00, extraction_confidence=0.9,
    )
    clean, n = sweep_receipt(receipt)
    assert receipt.merchant == "jeremy@example.com Store"  # original untouched
    assert clean.merchant != receipt.merchant
