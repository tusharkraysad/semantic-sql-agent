"""
Reference answers for the golden set, written directly against the raw files.

This file deliberately does NOT import the semantic layer. Each answer encodes
the business rule by hand (LEVEL = 'brand', de-duplicate cat_sales per key,
Actual spend only, the small-base growth rule), so a bug in the semantic layer
shows up as a disagreement instead of being reproduced on both sides.
"""

from __future__ import annotations

import os
from datetime import date

import polars as pl


def _load(data_dir: str):
    fin = pl.read_csv(os.path.join(data_dir, "finance_pnl.csv"), try_parse_dates=True)
    ms = pl.read_csv(os.path.join(data_dir, "market_share.csv"))
    bmi = pl.read_csv(os.path.join(data_dir, "bmi_spends.csv"), schema_overrides={"time_key": pl.Utf8})
    return fin, ms, bmi


def _year(fin: pl.DataFrame, y: int) -> pl.DataFrame:
    return fin.filter(pl.col("Monthly Period").dt.year() == y)


def _pairs(df: pl.DataFrame, k: str, v: str) -> list[tuple]:
    return [(str(a), float(b)) for a, b in zip(df[k].to_list(), df[v].to_list())]


def _share(ms: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    b = ms.filter(pl.col("LEVEL") == "brand")
    num = b.group_by(by).agg(pl.col("Sales_Value_Crs").sum().alias("n"))
    den = b.group_by(by + ["key"]).agg(pl.col("cat_sales").max().alias("c")).group_by(by).agg(pl.col("c").sum().alias("d"))
    return num.join(den, on=by).with_columns((100.0 * pl.col("n") / pl.col("d")).alias("v"))


def _growth(cur: pl.DataFrame, prev: pl.DataFrame, k: str, v: str, ratio: bool = False, small: float = 0.05):
    j = cur.join(prev, on=k, suffix="_p")
    if ratio:
        j = j.with_columns((pl.col(v) - pl.col(v + "_p")).alias("g"))
        return [str(x) for x in j.sort("g", descending=True)[k].to_list()]
    j = j.with_columns((100.0 * (pl.col(v) - pl.col(v + "_p")) / pl.col(v + "_p").abs()).alias("g"))
    med = j[v + "_p"].abs().median()
    j = j.with_columns((pl.col(v + "_p").abs() < small * med).alias("small"))
    return [str(x) for x in j.sort(["small", "g"], descending=[False, True])[k].to_list()]


def expected(data_dir: str) -> dict:
    fin, ms, bmi = _load(data_dir)
    f25, f24 = _year(fin, 2025), _year(fin, 2024)
    out: dict = {}
    out["G01"] = float(f25["Total Turnover"].sum())
    t = f25.group_by("Main Brand").agg(pl.col("Total Turnover").sum()).sort("Total Turnover", descending=True)
    out["G02"] = _pairs(t.head(5), "Main Brand", "Total Turnover")
    last3 = fin.filter(pl.col("Monthly Period") >= date(2025, 10, 1))
    out["G03"] = _pairs(last3.group_by("Big C").agg(pl.col("Total Turnover").sum()), "Big C", "Total Turnover")
    gm = f25.group_by("Main Brand").agg((100 * pl.col("Total Gross Profit").sum() / pl.col("Total Turnover").sum()).alias("v"))
    out["G04"] = _pairs(gm, "Main Brand", "v")
    d = fin.filter(pl.col("Monthly Period") == date(2025, 12, 1)).group_by("Region").agg(pl.col("Total Gross Profit").sum())
    out["G05"] = _pairs(d.sort("Total Gross Profit").head(3), "Region", "Total Gross Profit")
    tr = f25.filter(pl.col("Main Brand") == "Spark").group_by("Monthly Period").agg(pl.col("Total Turnover").sum()).sort("Monthly Period")
    out["G06"] = [(p.strftime("%Y-%m-%d"), float(v)) for p, v in zip(tr["Monthly Period"], tr["Total Turnover"])]
    s = _share(ms.filter((pl.col("category") == "skin care") & (pl.col("BRAND") == "velvet") & (pl.col("month") == "apr-26")), ["BRAND"])
    out["G07"] = float(s["v"][0])
    s = _share(ms.filter((pl.col("category") == "fabric wash") & (pl.col("month") == "apr-26")), ["BRAND"]).sort("v", descending=True)
    out["G08"] = _pairs(s.head(3), "BRAND", "v")
    r = ms.filter((pl.col("LEVEL") == "brand") & (pl.col("month") == "dec-25")).group_by("manufacturer").agg(pl.col("Sales_Value_Crs").sum())
    out["G09"] = _pairs(r, "manufacturer", "Sales_Value_Crs")
    hc = ms.filter((pl.col("category") == "hair care") & (pl.col("month") == "jan-26"))
    out["G10"] = float(hc.group_by("key").agg(pl.col("cat_sales").max())["cat_sales"].sum())
    b25 = bmi.filter((pl.col("Version") == "Actual") & pl.col("time_key").str.starts_with("2025"))
    out["G11"] = _pairs(b25.group_by("Media Type").agg(pl.col("Value").sum()), "Media Type", "Value")
    out["G12"] = float(b25.filter(pl.col("Brand") == "Clearo Surface")["Value"].sum())
    t = f25.filter(pl.col("Main Brand") != "Lumen").group_by("Main Brand").agg(pl.col("Total Turnover").sum()).sort("Total Turnover", descending=True)
    out["G13"] = _pairs(t.head(3), "Main Brand", "Total Turnover")
    out["G14"] = float(fin.filter(pl.col("Monthly Period") != date(2025, 12, 1))["Total Turnover"].sum())
    p = f25.group_by("SKU").agg(pl.col("Total Turnover").sum()).sort(["Total Turnover", "SKU"], descending=[True, False])
    out["G15"] = _pairs(p, "SKU", "Total Turnover")
    cur = f25.group_by("Main Brand").agg(pl.col("Total Turnover").sum())
    prev = f24.group_by("Main Brand").agg(pl.col("Total Turnover").sum())
    out["G16"] = _growth(cur, prev, "Main Brand", "Total Turnover")[:5]
    out["G17"] = float(f25.filter(pl.col("Main Brand") == "Brightwave")["Total Turnover"].sum())
    out["G18"] = float(f25.filter(pl.col("Main Brand") == "Matrix")["Total Turnover"].sum())
    out["G19"] = _pairs(f25.group_by("Region").agg(pl.col("Volume Tonnes").sum()), "Region", "Volume Tonnes")
    q3 = fin.filter((pl.col("Monthly Period") >= date(2025, 7, 1)) & (pl.col("Monthly Period") <= date(2025, 9, 1)))
    out["G20"] = _pairs(q3.group_by("Small C").agg(pl.col("Total Turnover").sum()), "Small C", "Total Turnover")
    out["G21"] = float(f25["Total Acquisition Turnover"].sum())
    crosswalk = {"glint dishwash": "glint", "clearo surface": "clearo", "silkara hair": "silkara", "nova skin": "nova"}
    sp = b25.with_columns(pl.col("Brand").str.to_lowercase().replace(crosswalk).alias("k")).group_by("k").agg(
        pl.col("Value").sum(), pl.col("Brand").max().alias("label"))
    to = f25.with_columns(pl.col("Main Brand").str.to_lowercase().alias("k")).group_by("k").agg(pl.col("Total Turnover").sum())
    mi = sp.join(to, on="k").with_columns((100 * pl.col("Value") / pl.col("Total Turnover")).alias("v"))
    out["G22"] = _pairs(mi, "label", "v")
    order = ["nov-25", "dec-25", "jan-26", "feb-26", "mar-26", "apr-26"]
    st = _share(ms.filter(pl.col("BRAND") == "spark"), ["month"])
    out["G23"] = [(m, float(st.filter(pl.col("month") == m)["v"][0])) for m in order]
    q = gm.join(cur, on="Main Brand")
    mg, mt = q["v"].median(), q["Total Turnover"].median()
    labels = {}
    for b, g, tt in zip(q["Main Brand"], q["v"], q["Total Turnover"]):
        x = "High Gross margin %" if g >= mg else "Low Gross margin %"
        y = "high Total Turnover" if tt >= mt else "low Total Turnover"
        labels[b] = f"{x}, {y}"
    out["G24"] = labels
    out["G25"] = float(f25.filter(pl.col("Main Brand") == "Glint")["Total Turnover"].sum())
    dc = fin.filter(pl.col("Monthly Period") == date(2025, 12, 1)).group_by("Big C").agg(pl.col("Total Turnover").sum())
    dp = fin.filter(pl.col("Monthly Period") == date(2024, 12, 1)).group_by("Big C").agg(pl.col("Total Turnover").sum())
    out["G28"] = _growth(dc, dp, "Big C", "Total Turnover")
    g25 = f25.group_by("Big C").agg((100 * pl.col("Total Gross Profit").sum() / pl.col("Total Turnover").sum()).alias("v"))
    g24 = f24.group_by("Big C").agg((100 * pl.col("Total Gross Profit").sum() / pl.col("Total Turnover").sum()).alias("v"))
    out["G29"] = _growth(g25, g24, "Big C", "v", ratio=True)
    lr = f25.filter(pl.col("Main Brand") == "Lumen").group_by("Region").agg(pl.col("Total Turnover").sum()).sort("Total Turnover", descending=True)
    out["G30"] = _pairs(lr.head(1), "Region", "Total Turnover")
    return out
