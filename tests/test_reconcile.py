import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from reconcile import apply_reconciliation, assert_numbers_present, reconcile
from schemas import LineItem, Receipt


def make_receipt(items, tax=0.0, total=None):
    computed = sum(i.amount * i.qty for i in items) + tax
    return Receipt(
        merchant="Test Store", line_items=items, tax=tax,
        total=total if total is not None else computed,
        extraction_confidence=0.9,
    )


def test_reconciliation_ok_when_matching():
    items = [LineItem(desc="Bread", amount=3.50), LineItem(desc="Milk", amount=2.00)]
    receipt = make_receipt(items, tax=0.50)  # total = 6.00
    result = reconcile(receipt)
    assert result.status == "ok"
    assert result.computed_total == 6.00


def test_reconciliation_mismatch_on_corrupted_receipt():
    items = [LineItem(desc="Bread", amount=3.50), LineItem(desc="Milk", amount=2.00)]
    receipt = make_receipt(items, tax=0.50, total=999.00)  # deliberately corrupted total
    result = reconcile(receipt)
    assert result.status == "mismatch"
    assert result.delta != 0


def test_reconciliation_tolerance_absorbs_rounding():
    items = [LineItem(desc="Item", amount=10.001)]
    receipt = make_receipt(items, total=10.00)  # 0.001 rounding noise
    result = reconcile(receipt)
    assert result.status == "ok"


def test_reconciliation_respects_qty():
    items = [LineItem(desc="Apple", amount=0.50, qty=6)]  # 6 x $0.50 = $3.00
    receipt = make_receipt(items, total=3.00)
    result = reconcile(receipt)
    assert result.status == "ok"
    assert result.computed_total == 3.00


def test_apply_reconciliation_attaches_result():
    items = [LineItem(desc="Item", amount=5.00)]
    receipt = make_receipt(items, total=5.00)
    updated = apply_reconciliation(receipt)
    assert updated.reconciliation is not None
    assert updated.reconciliation.status == "ok"


def test_narration_guard_accepts_number_present():
    assert assert_numbers_present("You spent $42.50 on dining in June.", [42.50]) is True


def test_narration_guard_rejects_dropped_number():
    # the model was given 42.50 but wrote a different number -- a would-be
    # hallucination the guard must catch
    assert assert_numbers_present("You spent $99.00 on dining in June.", [42.50]) is False


def test_narration_guard_tolerates_rounding_format():
    assert assert_numbers_present("You spent $42 on dining in June.", [42.00]) is True


def test_narration_guard_multiple_required_numbers():
    sentence = "Groceries: $120.00, up from $95.50 last month."
    assert assert_numbers_present(sentence, [120.00, 95.50]) is True
    assert assert_numbers_present(sentence, [120.00, 999.00]) is False
