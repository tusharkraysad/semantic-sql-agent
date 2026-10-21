"""
Synthetic data generator
========================
Builds three files that look like a consumer-goods company's commercial data,
with invented brands and companies. No real client data is used anywhere.

The files deliberately carry the traps that make plain text-to-SQL fail:

  finance_pnl.csv
    - "Monthly Period" is an ISO date, Jan-2024 .. Dec-2025
    - "Total Acquisition Turnover" is zero on ~99% of rows and nets negative:
      a column that *sounds* like revenue but must not be used for it
    - brand "Nova" launched in 2025 with a near-zero 2024 base, so naive growth
      rankings put it at the top with an absurd percentage

  market_share.csv
    - "month" is text like 'dec-25' (does not sort chronologically) and the
      file runs to Apr-26, four months past the P&L
    - "LEVEL" mixes brand, manufacturer and category rows: summing without
      LEVEL = 'brand' triple counts
    - "cat_sales" is a category total repeated on every row of its
      market x category x month group: SUM() inflates it ~10x
    - brand names are stored lower case, the P&L stores them in Title Case

  bmi_spends.csv
    - "time_key" is YYYYMMDD text; forecast months carry a '_F' suffix
    - "Big_C" uses codes (HC, PC, FD) that never match the P&L's names
    - "Brand" only partly matches the P&L (some names carry a suffix)

Usage:  python -m sqlagent.synthetic --out data --seed 7
"""

from __future__ import annotations

import argparse
import calendar
import os
from datetime import date

import numpy as np
import polars as pl

PORTFOLIO = {
    # Big C -> Small C -> Main Brand -> [Sub Brands]
    "Home Care": {
        "Fabric Wash": {"Spark": ["Spark Ultra", "Spark Classic"], "Brightwave": ["Brightwave Matic", "Brightwave Bar"]},
        "Dishwash": {"Glint": ["Glint Gel", "Glint Bar"]},
        "Surface Cleaners": {"Clearo": ["Clearo Floor", "Clearo Glass"]},
    },
    "Personal Care": {
        "Skin Care": {"Velvet": ["Velvet Glow", "Velvet Care"], "Lumen": ["Lumen Bright", "Lumen Night"], "Nova": ["Nova Fresh"]},
        "Hair Care": {"Matrix": ["Matrix Strong", "Matrix Smooth"], "Silkara": ["Silkara Herbal"]},
        "Oral Care": {"Pearla": ["Pearla White", "Pearla Sensitive"]},
    },
    "Foods": {
        "Tea": {"Red Leaf": ["Red Leaf Gold", "Red Leaf Dust"]},
        "Spreads": {"Goldspread": ["Goldspread Classic"]},
    },
}
REGIONS = ["North", "South", "East", "West"]
BIG_C_CODE = {"Home Care": "HC", "Personal Care": "PC", "Foods": "FD"}

# Market-share categories: ours plus competitors, all stored lower case.
MS_CATEGORIES = {
    "fabric wash": {"acme consumer goods": ["spark", "brightwave"], "orion brands": ["ultrafresh", "washwell"], "zenith home": ["purely"]},
    "dishwash": {"acme consumer goods": ["glint"], "orion brands": ["dazzle"], "zenith home": ["scrubby"]},
    "skin care": {"acme consumer goods": ["velvet", "lumen", "nova"], "orion brands": ["dermis"], "zenith home": ["herbalia"]},
    "hair care": {"acme consumer goods": ["matrix", "silkara"], "orion brands": ["mane"], "zenith home": ["curlz"]},
    "tea": {"acme consumer goods": ["red leaf"], "orion brands": ["chai king"], "zenith home": ["leafy"]},
}
MS_MARKETS = ["north", "south", "east", "west", "metro", "rural"]
MS_MONTHS = [(2025, 11), (2025, 12), (2026, 1), (2026, 2), (2026, 3), (2026, 4)]


def mon_yy(year: int, month: int) -> str:
    return f"{calendar.month_abbr[month].lower()}-{year % 100:02d}"


def build_finance(rng: np.random.Generator) -> pl.DataFrame:
    months = [date(y, m, 1) for y in (2024, 2025) for m in range(1, 13)]
    rows = []
    brand_scale = {}
    brand_growth = {}
    brand_margin = {}
    for big_c, smalls in PORTFOLIO.items():
        for small_c, brands in smalls.items():
            for brand, subs in brands.items():
                brand_scale[brand] = float(rng.uniform(0.6, 3.0))
                brand_growth[brand] = float(rng.uniform(-0.08, 0.22))
                brand_margin[brand] = float(rng.uniform(0.32, 0.56))
                for sub in subs:
                    for k in range(1, 4):
                        sku = f"{sub.upper().replace(' ', '-')[:12]}-{k}"
                        for region in REGIONS:
                            region_factor = {"North": 1.1, "South": 1.0, "East": 0.8, "West": 1.2}[region]
                            for i, period in enumerate(months):
                                seasonal = 1 + 0.12 * np.sin(2 * np.pi * (period.month - 3) / 12)
                                trend = (1 + brand_growth[brand]) ** (i / 12)
                                base = brand_scale[brand] * region_factor * seasonal * trend / 3.0
                                if brand == "Nova" and period.year == 2024:
                                    base = 1e-6  # tiny pre-launch base: the small-base growth trap
                                turnover = max(base * float(rng.normal(1.0, 0.06)), 0.0)
                                margin = brand_margin[brand] + float(rng.normal(0, 0.015))
                                gp = turnover * margin
                                acq = 0.0
                                if rng.random() < 0.01:
                                    acq = float(rng.normal(-0.3, 0.2))  # rare adjustments, net negative
                                rows.append(
                                    {
                                        "Monthly Period": period,
                                        "Big C": big_c,
                                        "Small C": small_c,
                                        "Main Brand": brand,
                                        "Sub Brand": sub,
                                        "SKU": sku,
                                        "Region": region,
                                        "Total Turnover": round(turnover, 6),
                                        "Total Acquisition Turnover": round(acq, 6),
                                        "Total Gross Profit": round(gp, 6),
                                        "Total COP": round(turnover - gp, 6),
                                        "Volume Tonnes": round(turnover * float(rng.uniform(8, 14)), 3),
                                    }
                                )
    return pl.DataFrame(rows)


def build_market_share(rng: np.random.Generator) -> pl.DataFrame:
    rows = []
    for category, makers in MS_CATEGORIES.items():
        brand_strength = {b: float(rng.uniform(0.5, 3.0)) for m in makers.values() for b in m}
        for market in MS_MARKETS:
            market_factor = float(rng.uniform(0.6, 1.4))
            for i, (y, m) in enumerate(MS_MONTHS):
                month = mon_yy(y, m)
                key = f"{market}_{category.replace(' ', '')}_{y}{m:02d}"
                brand_values = {}
                for maker, brands in makers.items():
                    for b in brands:
                        drift = 1 + 0.03 * i * (1 if maker == "acme consumer goods" else -0.5)
                        brand_values[(maker, b)] = brand_strength[b] * market_factor * drift * float(rng.normal(1, 0.05))
                others = sum(brand_values.values()) * float(rng.uniform(0.25, 0.4))  # long tail outside these brands
                cat_total = sum(brand_values.values()) + others
                cat_vol = cat_total * 11.0
                for (maker, b), value in brand_values.items():
                    rows.append((month, market, category, "brand", b, maker, key, value, value * 11.0, cat_total, cat_vol))
                for maker in makers:
                    mv = sum(v for (mk, _), v in brand_values.items() if mk == maker)
                    rows.append((month, market, category, "manufacturer", maker, maker, key, mv, mv * 11.0, cat_total, cat_vol))
                rows.append((month, market, category, "category", category, "all", key, cat_total, cat_vol, cat_total, cat_vol))
    cols = ["month", "market", "category", "LEVEL", "BRAND", "manufacturer", "key",
            "Sales_Value_Crs", "Sales_Vol_Tonnes", "cat_sales", "cat_vol"]
    df = pl.DataFrame(rows, schema=cols, orient="row")
    return df.with_columns([pl.col(c).round(6) for c in ["Sales_Value_Crs", "Sales_Vol_Tonnes", "cat_sales", "cat_vol"]])


def build_bmi(rng: np.random.Generator) -> pl.DataFrame:
    rename = {"Glint": "Glint Dishwash", "Clearo": "Clearo Surface", "Silkara": "Silkara Hair", "Nova": "Nova Skin"}
    rows = []
    months = [(y, m) for y in (2024, 2025) for m in range(1, 13)] + [(2026, 1), (2026, 2), (2026, 3)]
    for big_c, smalls in PORTFOLIO.items():
        for brands in smalls.values():
            for brand in brands:
                label = rename.get(brand, brand)
                intensity = float(rng.uniform(0.04, 0.12))
                for (y, m) in months:
                    forecast = y == 2026
                    time_key = f"{y}{m:02d}01" + ("_F" if forecast else "")
                    for media, share in (("TV", 0.45), ("Digital", 0.35), ("Print", 0.1), ("OOH", 0.1)):
                        spend = intensity * share * float(rng.uniform(0.7, 1.3))
                        if brand == "Nova" and y == 2024:
                            spend = 0.0
                        rows.append({
                            "time_key": time_key,
                            "Big_C": BIG_C_CODE[big_c],
                            "Brand": label,
                            "Media Type": media,
                            "Version": "Forecast" if forecast else "Actual",
                            "Value": round(spend, 6),
                        })
    return pl.DataFrame(rows)


SOURCES_YAML = """\
# Which files the agent loads. Paths are relative to this file.
sources:
  - name: finance_pnl
    path: finance_pnl.csv
    description: Monthly profit and loss by SKU and region (Jan-2024 to Dec-2025).
  - name: market_share
    path: market_share.csv
    description: Syndicated retail panel by market, category and brand (Nov-25 to Apr-26).
  - name: bmi_spends
    path: bmi_spends.csv
    description: Brand and marketing investment by brand, media type and month, with forecasts.
"""


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    finance = build_finance(rng)
    market = build_market_share(rng)
    bmi = build_bmi(rng)
    finance.write_csv(os.path.join(args.out, "finance_pnl.csv"))
    market.write_csv(os.path.join(args.out, "market_share.csv"))
    bmi.write_csv(os.path.join(args.out, "bmi_spends.csv"))
    with open(os.path.join(args.out, "data_sources.yaml"), "w", encoding="utf-8") as f:
        f.write(SOURCES_YAML)
    print(f"finance_pnl: {finance.height:,} rows | market_share: {market.height:,} rows | bmi_spends: {bmi.height:,} rows -> {args.out}/")


if __name__ == "__main__":
    main()
