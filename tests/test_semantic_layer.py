def test_bootstrap_detects_the_traps(engine):
    model = engine.loaded.model
    assert model.columns["market_share.cat_sales"].is_repeated_base
    assert model.columns["market_share.cat_sales"].grain == ["key"]
    assert not model.columns["market_share.Sales_Value_Crs"].is_repeated_base
    assert not model.columns["finance_pnl.Total Acquisition Turnover"].is_repeated_base
    assert "\"LEVEL\" = 'brand'" in model.tables["market_share"].default_filters


def test_generic_vocabulary_stays_on_the_plain_column(engine):
    model = engine.loaded.model
    assert "revenue" in model.columns["finance_pnl.Total Turnover"].synonyms
    assert "revenue" not in model.columns["finance_pnl.Total Acquisition Turnover"].synonyms


def test_joins_are_measured(engine):
    joins = {(j.left_table, j.left_column, j.right_table, j.right_column): j for j in engine.loaded.model.joins}
    big_c = [j for k, j in joins.items() if "Big C" in k or "Big_C" in k][0]
    assert not big_c.enabled  # HC/PC/FD codes never match "Home Care"
    brand = [j for j in joins.values() if j.entity == "brand" and {j.left_table, j.right_table} == {"finance_pnl", "market_share"}][0]
    assert brand.enabled and brand.alignment >= 0.5


def test_value_index_matching(engine):
    vi = engine.loaded.values
    assert vi.canonicalize("finance_pnl.Main Brand", "brightwav")[0] == "Brightwave"
    assert vi.canonicalize("bmi_spends.Brand", "Clearo")[0] == "Clearo Surface"
    assert vi.canonicalize("finance_pnl.Main Brand", "zzz")[0] is None


def test_resolver_reads_shape_words_and_exclusions(engine):
    ctx = engine.loaded.resolver.resolve("2x2 matrix of brands by turnover and gross margin")
    assert not ctx.value_matches and ctx.shape_values
    ctx = engine.loaded.resolver.resolve("turnover for matrix brand")
    assert any(m.value == "Matrix" for m in ctx.value_matches)
    ctx = engine.loaded.resolver.resolve("top 3 brands by turnover excluding Lumen")
    assert all(m.negated for m in ctx.value_matches if m.value.lower() == "lumen")


def test_enrichment_synonym_is_flagged(engine):
    ctx = engine.loaded.resolver.resolve("manufacturing cost by region")
    assert ctx.unverified_matches
