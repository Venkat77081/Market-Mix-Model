"""Synthetic insurance datasets with known structure, for tests and the UI.

make_insurance() builds the ins_clean shape from the roadmap: 156 weeks,
9 paid channels, controls, non-media levers and unpaid media. The writers
below turn it into the messy exports Phase A has to read.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CHANNELS = ["tv", "youtube", "search", "aggregator", "meta", "display", "outdoor", "radio", "print"]
REGIONS = {"North": 3.1e7, "South": 2.7e7, "East": 2.2e7, "West": 2.9e7, "Central": 1.6e7}


def make_insurance(weeks: int = 156, regions: bool = False, rf: bool = False, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-02", periods=weeks, freq="W-MON")
    geos = list(REGIONS) if regions else [None]
    frames = []
    for g in geos:
        scale = REGIONS[g] / 2.5e7 if g else 1.0
        df = pd.DataFrame({"week_start": dates})
        if g:
            df["region"] = g
            df["population"] = REGIONS[g]
        for ch in CHANNELS:
            on = rng.random(weeks) < (0.6 if ch in ("tv", "outdoor", "radio", "print") else 0.95)
            df[f"{ch}_spend"] = np.round(on * rng.gamma(4, 2.5e5, weeks) * scale, 0)
        df["tv_grps"] = np.round(df["tv_spend"] / 9e4, 1)
        df["search_clicks"] = np.round(df["search_spend"] / 22, 0)
        df["meta_impressions"] = np.round(df["meta_spend"] * 3.1, 0)
        if rf:
            df["youtube_reach"] = np.round(df["youtube_spend"] * 0.8, 0)
            df["youtube_frequency"] = np.round(1.5 + rng.random(weeks), 2)
        df["premium_index"] = np.round(100 + np.cumsum(rng.normal(0, 0.4, weeks)), 2)
        df["competitor_spend"] = np.round(rng.gamma(5, 3e5, weeks), 0)
        df["vehicle_registrations"] = np.round(rng.normal(2.1e5, 1.5e4, weeks) * scale, 0)
        df["tax_season_flag"] = df["week_start"].dt.month.isin([1, 2, 3]).astype(int)
        df["discount_pct"] = np.round(rng.choice([0, 5, 10], weeks, p=[0.7, 0.2, 0.1]), 0)
        df["email_sends"] = np.round(rng.gamma(3, 4e4, weeks) * scale, 0)
        df["new_policies"] = np.round(rng.normal(4200, 300, weeks) * scale, 0)
        df["avg_premium"] = np.round(rng.normal(14500, 400, weeks), 0)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def indian_number(x: float) -> str:
    """1200000 -> '12,00,000'."""
    s = f"{int(round(x))}"
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return ",".join(groups) + "," + tail


def write_clean(dir: Path) -> Path:
    path = dir / "ins_clean.csv"
    df = make_insurance()
    df["week_start"] = df["week_start"].dt.strftime("%Y-%m-%d")
    df.to_csv(path, index=False)
    return path


def write_excel_with_preamble(dir: Path) -> Path:
    """Two sheets; the data sheet has three preamble lines above the header."""
    path = dir / "ins_export.xlsx"
    df = make_insurance()
    with pd.ExcelWriter(path) as xw:
        pd.DataFrame({"Notes": ["Exported from media dashboard", "Spend in INR"]}).to_excel(xw, sheet_name="Notes", index=False)
        pre = pd.DataFrame([["Weekly media and sales report"], ["Generated 2026-01-05"], [None]])
        pre.to_excel(xw, sheet_name="Weekly data", index=False, header=False)
        df.to_excel(xw, sheet_name="Weekly data", index=False, startrow=3)
    return path


def write_dirty_dates(dir: Path) -> Path:
    """Semicolon-delimited, cp1252-encoded, Indian number formats, text
    nulls, and dates mixing dd/mm/yyyy (first half) with ISO (second half)."""
    path = dir / "ins_dirty.csv"
    df = make_insurance()
    half = len(df) // 2
    df["week_start"] = [
        d.strftime("%d/%m/%Y") if i < half else d.strftime("%Y-%m-%d") for i, d in enumerate(df["week_start"])
    ]
    df["tv_spend"] = df["tv_spend"].map(indian_number)
    df["search_clicks"] = df["search_clicks"].astype(object)
    df.loc[5, "search_clicks"] = "N/A"
    df = df.rename(columns={"discount_pct": "Remise_promo_%"})  # forces a non-UTF-8 byte below
    df.columns = [c.replace("Remise", "Remisé") for c in df.columns]
    df.to_csv(path, index=False, sep=";", encoding="cp1252")
    return path


def write_ambiguous_monthly(dir: Path) -> Path:
    """Monthly data on the 1st: every slash date reads both ways."""
    path = dir / "ins_monthly.csv"
    df = make_insurance(weeks=36)
    df["week_start"] = pd.date_range("2023-01-01", periods=36, freq="MS").strftime("%d/%m/%Y")
    df = df.rename(columns={"week_start": "month"})
    df.to_csv(path, index=False)
    return path


def write_regions(dir: Path) -> Path:
    path = dir / "ins_regions.csv"
    make_insurance(regions=True).to_csv(path, index=False)
    return path


def write_rf(dir: Path) -> Path:
    path = dir / "ins_rf.csv"
    df = make_insurance(rf=True)
    df["print_impressions"] = df.pop("print_spend") * 4  # exposure with no spend
    df["branded_search_queries"] = np.round(df["tv_grps"] * 120 + 5000, 0)
    df["gross_written_premium"] = df["new_policies"] * df["avg_premium"]
    df.to_csv(path, index=False)
    return path


# Phase B test files. MESSY_TRUTH describes exactly what write_messy breaks, so
# tests can check every count.
MESSY_TRUTH = {
    "duplicate_dates": 20,     # weeks 10-29 appear twice, each row holding half the spend and KPI
    "missing_weeks": 4,        # weeks 40, 41, 90, 120 are absent
    "kpi_blank_weeks": 2,      # new_policies blank in weeks 60 and 61
    "radio_starts_week": 30,   # radio_spend blank before week 30
    "negative_cells": 2,       # meta_spend week 70, search_spend week 71
    "kpi_zero_week": 100,
    "index_jump_week": 130,    # premium_index x10 in week 130 only
    "outlier_week": 140,       # new_policies x3
    "tv_unit": "crore",        # tv_spend_cr holds TV spend in crore
}


def write_messy(dir: Path) -> Path:
    """Weekly export with every problem Phase B handles, at known places."""
    path = dir / "ins_messy.csv"
    df = make_insurance()
    df["promo_flag"] = (np.arange(len(df)) % 9 == 0).astype(int)
    df["radio_spend"] = df["radio_spend"].astype(object)
    df.loc[: MESSY_TRUTH["radio_starts_week"] - 1, "radio_spend"] = ""
    df.loc[60:61, "new_policies"] = np.nan
    df.loc[70, "meta_spend"] = -125000
    df.loc[71, "search_spend"] = -40000
    df.loc[MESSY_TRUTH["kpi_zero_week"], "new_policies"] = 0
    df.loc[MESSY_TRUTH["index_jump_week"], "premium_index"] *= 10
    df.loc[MESSY_TRUTH["outlier_week"], "new_policies"] *= 3
    df["tv_spend_cr"] = df.pop("tv_spend") / 1e7

    halves = df.loc[10:29].copy()
    money = [c for c in df.columns if c.endswith(("_spend", "_spend_cr", "_grps", "_clicks", "_impressions", "_sends")) or c == "new_policies"]
    for c in money:
        num = pd.to_numeric(df.loc[10:29, c], errors="coerce")
        df.loc[10:29, c] = num / 2
        halves[c] = num / 2
    df = pd.concat([df, halves]).drop(index=[40, 41, 90, 120]).sort_values("week_start", kind="stable")
    df["week_start"] = df["week_start"].dt.strftime("%Y-%m-%d")
    df.to_csv(path, index=False)
    return path


def write_daily(dir: Path) -> Path:
    """Two years of daily data starting on a Wednesday."""
    path = dir / "ins_daily.csv"
    rng = np.random.default_rng(3)
    days = pd.date_range("2023-01-04", periods=730, freq="D")
    df = pd.DataFrame({"date": days.strftime("%Y-%m-%d")})
    for ch in ("tv", "search", "meta"):
        df[f"{ch}_spend"] = np.round(rng.gamma(4, 3.5e4, len(df)), 0)
    df["festive_flag"] = days.month.isin([10, 11]).astype(int)
    df["premium_index"] = np.round(100 + np.cumsum(rng.normal(0, 0.05, len(df))), 2)
    df["new_policies"] = np.round(rng.normal(600, 40, len(df)), 0)
    df.to_csv(path, index=False)
    return path


# Phase C test files: each plants one problem a profile step must catch.
def _weekly_csv(df: pd.DataFrame, path: Path) -> Path:
    df = df.copy()
    df["week_start"] = df["week_start"].dt.strftime("%Y-%m-%d")
    df.to_csv(path, index=False)
    return path


def write_short(dir: Path) -> Path:
    """90 weeks: refused at step 14."""
    return _weekly_csv(make_insurance(weeks=90), dir / "ins_short.csv")


def write_flat(dir: Path) -> Path:
    """Search and aggregator spend barely vary (CV under 0.2): hard to measure at step 16."""
    df = make_insurance()
    rng = np.random.default_rng(11)
    for ch in ("search", "aggregator"):
        df[f"{ch}_spend"] = np.round(1.2e6 * (1 + rng.normal(0, 0.08, len(df))), 0)
    df["search_clicks"] = np.round(df["search_spend"] / 22, 0)
    return _weekly_csv(df, dir / "ins_flat.csv")


def write_collinear(dir: Path) -> Path:
    """TV and YouTube always on together: combine prompt at step 17."""
    df = make_insurance()
    rng = np.random.default_rng(12)
    df["youtube_spend"] = np.round(df["tv_spend"] * 0.4 * (1 + rng.normal(0, 0.1, len(df))), 0).clip(lower=0)
    return _weekly_csv(df, dir / "ins_collinear.csv")


def write_mediator(dir: Path) -> Path:
    """A search-interest score that follows TV spend a week later: exclusion prompt at step 18."""
    df = make_insurance()
    rng = np.random.default_rng(13)
    df["search_interest_index"] = np.round(40 + df["tv_spend"].shift(1).fillna(0) / 5e4 + rng.normal(0, 1.5, len(df)), 1)
    return _weekly_csv(df, dir / "ins_mediator.csv")


def write_rare(dir: Path) -> Path:
    """Print on in 8 of 156 weeks and radio under 1% of spend: merge prompt at step 15."""
    df = make_insurance()
    on = np.zeros(len(df), dtype=bool)
    on[[10, 30, 50, 70, 90, 110, 130, 150]] = True
    df["print_spend"] = np.where(on, df["print_spend"].clip(lower=5e5), 0.0)
    df["radio_spend"] = np.round(df["radio_spend"] / 200, 0)
    return _weekly_csv(df, dir / "ins_rare.csv")


def write_season(dir: Path) -> Path:
    """Outdoor spends only in tax season: warning at step 19."""
    df = make_insurance()
    df["outdoor_spend"] = np.where(df["tax_season_flag"] == 1, 2.5e6, 0.0)
    return _weekly_csv(df, dir / "ins_season.csv")


SAMPLES = {
    "ins_clean": ("156 weeks, 9 channels, clean CSV", write_clean),
    "ins_export": ("Excel, two sheets, title lines above the header", write_excel_with_preamble),
    "ins_dirty": ("cp1252, semicolons, mixed dd/mm and ISO dates, 1,20,000 numbers", write_dirty_dates),
    "ins_monthly": ("Monthly, every date ambiguous (asks day-first or month-first)", write_ambiguous_monthly),
    "ins_regions": ("5 regions with population", write_regions),
    "ins_rf": ("YouTube reach and frequency, print with no spend, branded search", write_rf),
    "ins_messy": ("Duplicate and missing weeks, negative spend, TV in crore, an outlier", write_messy),
    "ins_daily": ("Two years of daily data starting on a Wednesday", write_daily),
    "ins_short": ("90 weeks: refused at step 14", write_short),
    "ins_flat": ("Search and aggregator spend barely vary: hard to measure", write_flat),
    "ins_collinear": ("TV and YouTube always on together", write_collinear),
    "ins_mediator": ("A search-interest score that follows TV spend", write_mediator),
    "ins_rare": ("Print on in 8 weeks, radio under 1% of spend", write_rare),
    "ins_season": ("Outdoor spends only in tax season", write_season),
}
