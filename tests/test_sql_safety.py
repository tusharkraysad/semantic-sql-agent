import pytest

from sqlagent.sql_validator import UnsafeSQL, assert_read_only


@pytest.mark.parametrize("sql", [
    "DROP TABLE finance_pnl",
    "SELECT 1; SELECT 2",
    "DELETE FROM finance_pnl",
    "INSERT INTO finance_pnl VALUES (1)",
])
def test_read_only(sql):
    with pytest.raises(UnsafeSQL):
        assert_read_only(sql)


def test_validator_rejects_the_classic_mistakes(engine):
    v = engine.loaded.validator
    assert not v.validate('SELECT SUM("cat_sales") AS x FROM market_share').ok          # repeated base
    assert not v.validate('SELECT * FROM finance_pnl').ok                               # star
    assert not v.validate('SELECT "Nope" FROM finance_pnl').ok                          # unknown column
    bad = v.validate("SELECT SUM(\"Total Turnover\") FROM finance_pnl WHERE \"Main Brand\" = 'spark'")
    assert not bad.ok and "Spark" in bad.errors[0]                                     # wrong case literal
    ok = v.validate("SELECT SUM(\"Total Turnover\") AS t FROM finance_pnl WHERE \"Main Brand\" = 'Spark'")
    assert ok.ok
    warn = v.validate('SELECT SUM("Sales_Value_Crs") AS s FROM market_share')
    assert warn.ok and warn.warnings                                                   # missing LEVEL filter


def test_composer_never_sums_a_repeated_base(agent):
    resp = agent.ask("category sales for hair care in jan-26", "t-safety")
    assert "MAX(\"cat_sales\")" in resp["sql"]
    assert "SUM(\"cat_sales\")" not in resp["sql"]
