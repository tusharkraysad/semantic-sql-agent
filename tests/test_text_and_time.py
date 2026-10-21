from sqlagent.semantic.time_layer import label_of, parse_literal, shift
from sqlagent.text_utils import negated_terms, normalize, singularize


def test_normalize_and_singularize():
    assert normalize("  DOVE  Ultra! ") == "dove ultra"
    assert singularize("brands") == "brand"
    assert singularize("categories") == "category"
    assert singularize("skus") == "sku"


def test_negation_attaches_to_nearest_term_on_the_right():
    assert negated_terms("sales excluding January", ["january"]) == {"january"}
    assert negated_terms("excluding returns, sales in January", ["january"]) == set()
    assert negated_terms("turnover excluding Lumen in dec-25", ["lumen", "dec-25"]) == {"lumen"}


def test_period_literals():
    assert parse_literal("dec-25") == ("mon_yy", 202512)
    assert parse_literal("20250301") == ("yyyymmdd", 202503)
    assert parse_literal("20260101_F") is None  # forecast rows are not periods
    assert shift(202501, -1) == 202412
    assert label_of(202604) == "Apr-26"


def test_time_windows_bind_per_file(engine):
    periods = engine.loaded.periods
    w = periods.parse("turnover in the last 3 months")
    assert periods.resolve(w, "finance_pnl").label == "Oct-25..Dec-25"
    assert periods.resolve(w, "market_share").label == "Feb-26..Apr-26"
    neg = periods.parse("excluding dec-25")
    neg.negated = True
    assert "IS NULL OR" in periods.resolve(neg, "finance_pnl").filter_sql
