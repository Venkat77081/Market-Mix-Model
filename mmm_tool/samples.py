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


SAMPLES = {
    "ins_clean": ("156 weeks, 9 channels, clean CSV", write_clean),
    "ins_export": ("Excel, two sheets, title lines above the header", write_excel_with_preamble),
    "ins_dirty": ("cp1252, semicolons, mixed dd/mm and ISO dates, 1,20,000 numbers", write_dirty_dates),
    "ins_monthly": ("Monthly, every date ambiguous (asks day-first or month-first)", write_ambiguous_monthly),
    "ins_regions": ("5 regions with population", write_regions),
    "ins_rf": ("YouTube reach and frequency, print with no spend, branded search", write_rf),
}
