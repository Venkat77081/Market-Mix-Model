"""MTR_FULL (tests/data/mtr_full/, its README.md): an Indian motor insurer, Phase A to Tracking.

Phases A-D run without Meridian and are checked against expected/expected_steps_1_27.json and
the expected tables cell by cell, with the README's decision script as the answers. The fitted
phases are in test_holdout_knots.py and test_calibrated_channels.py (MMM_FIT_TESTS=1).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mmm_tool.pipeline import run_pipeline

HERE = Path(__file__).parent / "data" / "mtr_full"
RAW = HERE / "raw" / "MTR_FULL_raw.xlsx"
needs = pytest.mark.skipif(not RAW.exists(), reason="MTR_FULL is not in tests/data/mtr_full")
SETTINGS = {"packs": ["india", "insurance"], "calendar": "india_insurance"}
ROLES = {"Week (dd/mm/yyyy)": "date", "New Policies": "kpi", "Premium (₹ lakh)": "revenue_for_kpi", "TV Spend": "media_spend",
         "TV GRPs": "media", "YouTube Spend": "media_spend", "YouTube Impr.": "media", "Search Spend": "media_spend",
         "Search Clicks": "media", "Meta Spend": "media_spend", "PolicyBazaar Commission": "media_spend", "Print Spend": "media_spend",
         "Branded Search Vol": "control", "Car Sales (SIAM)": "control", "Fuel Price Idx": "control", "Competitor Discount": "control",
         "Festive Season": "control", "Comments": "excluded"}
# The README's decision script; every other question takes the tool's default.
ANSWERS = {"roles": ROLES, "calendar_events": ["fy_end"], "knots": 12}
LIFT = {"channel": "search", "start": "2025-06-02", "end": "2025-07-27", "roi": 2.74, "low": 2.34, "high": 3.14}
CLEAN = {"Week (dd/mm/yyyy)": "week_start", "New Policies": "policies", "TV Spend": "tv_spend", "TV GRPs": "tv_grps",
         "YouTube Spend": "yt_spend", "YouTube Impr.": "yt_impressions", "Search Spend": "search_spend", "Search Clicks": "search_clicks",
         "Meta Spend": "meta_spend", "PolicyBazaar Commission": "agg_spend", "Print Spend": "print_spend",
         "Branded Search Vol": "branded_search", "Car Sales (SIAM)": "car_sales", "Fuel Price Idx": "fuel_index",
         "Competitor Discount": "competitor_discount", "Festive Season": "festive", "kpi_imputed": "imputed_flag",
         "event_2024_07_15": "event_flag", "revenue_per_kpi": "revenue_per_policy", "cal_fy_end": "fy_end"}


def run(extra=None):
    return run_pipeline(RAW, {**ANSWERS, **(extra or {})}, accept_defaults=True, settings=SETTINGS)


@pytest.fixture(scope="module")
def mtr():
    return run()


def steps(r):
    return {s.step: s for ph in (r.a, r.b, r.c, r.d) if ph for s in ph.steps}


def expected():
    return json.loads((HERE / "expected" / "expected_steps_1_27.json").read_text(encoding="utf-8"))


@needs
def test_phase_a(mtr):
    st, e = steps(mtr), expected()
    assert "159 rows × 18 columns" in st[1].summary and "sheet 'Weekly'; header on row 3" in st[1].summary
    assert st[2].details["unique_dates"] == e["2_dates"]["unique_dates"] == 156 and st[2].details["exact_copies"] == 3
    assert "97 dayfirst" in st[2].summary and "56 ambiguous" in st[2].summary and "6 same_either_way" in st[2].summary
    assert not any(d.id == "duplicate_dates" for d in st[2].decisions)              # exact copies: step 8's job
    pairs = st[4].decisions[0].answer
    assert {v["spend"]: v.get("exposure") for v in pairs.values() if v.get("exposure") and v["exposure"] != v["spend"]} == {
        k: v for k, v in e["4_pairs"].items() if not k.startswith("_")}


@needs
def test_step3_suggestions_without_answers():
    r = run_pipeline(RAW, {}, accept_defaults=True, settings=SETTINGS)
    sug = {s["column"]: s for s in steps(r)[3].details["suggestions"]}
    assert sug["PolicyBazaar Commission"]["role"] == "media_spend"                 # 'commission' is paid to a partner
    assert sug["Premium (₹ lakh)"]["role"] == "revenue_for_kpi" and "₹0.14" in sug["Premium (₹ lakh)"]["reason"]
    assert all(sug[c]["role"] == "media" for c in ("TV GRPs", "YouTube Impr.", "Search Clicks"))


@needs
def test_phase_b_matches_the_expected_clean_table(mtr):
    st = steps(mtr)
    assert "159 formatted numbers read (e.g. '8,39,000' in Meta Spend)" in st[7].summary
    assert "159 rows → 156" in st[8].summary and "3 exact copy" in st[8].summary
    cols = {c["column"]: c for c in st[10].details["columns"]}
    assert cols["Print Spend"]["ended"] == 79 and cols["Print Spend"]["zeros"] == 48
    assert st[10].details["real_zeros"]["TV Spend"] == 68 and st[10].details["real_zeros"]["YouTube Spend"] == 92
    assert list(st[11].details.get("flagged", {}) or {}) in ([], ["2024-07-15"]) and "1 unusual KPI week" in st[11].summary
    assert "median ₹14,890" in st[13].summary
    t = mtr.b.table.rename(columns=CLEAN)
    t["week_start"] = pd.to_datetime(t["week_start"]).dt.strftime("%Y-%m-%d")
    exp = pd.read_csv(HERE / "expected" / "expected_weekly_clean.csv")
    assert list(t["week_start"]) == list(exp["week_start"])
    for c in exp.columns[1:]:
        if c == "premium_lakh":                                                     # converted into revenue_per_policy
            continue
        assert np.allclose(t[c].astype(float), exp[c].astype(float), rtol=1e-6, atol=1e-6), c


@needs
def test_phase_c(mtr):
    st, e = steps(mtr), expected()
    assert st[14].details["weeks"] == 157 and st[14].details["parameters"]["baseline knots"] == 12   # 157/13 rounds to 12
    assert set(st[16].summary.rstrip(".").split(": ")[-1].split(", ")) == {"policy_bazaar_commission", "search"}
    assert "highest correlation 0.41" in st[17].summary
    assert "Branded Search Vol" in st[18].summary
    assert "₹111.4 Cr" in st[20].summary and e["20_profile"]["total_spend_cr"] == 111.39


@needs
def test_phase_d_matches_the_expected_model_table(mtr):
    st, e = steps(mtr), expected()
    assert "cal_fy_end" in st[21].summary and "tax" not in st[21].summary              # the preset that doesn't apply, declined
    if st[24].details["locations"] is not None:                                     # None without Meridian installed
        assert st[24].details["locations"] == e["24_knots"]["locations"]
    assert st[24].details["knots"] == e["24_knots"]["knots"]
    assert mtr.d.config["holdout"]["train_weeks"] == e["27_holdout"]["train_weeks"] == 144
    t = mtr.d.table.rename(columns=CLEAN)
    exp = pd.read_csv(HERE / "expected" / "model_input.csv")
    for c in exp.columns[1:]:
        assert np.allclose(t[c].astype(float), exp[c].astype(float), rtol=1e-6, atol=1e-6), c


@needs
def test_the_knots_default_is_12_for_157_weeks():
    r = run_pipeline(RAW, {"roles": ROLES}, accept_defaults=True, settings=SETTINGS)        # no knots answer: the default
    assert r.d.config["knots"] == 12


@needs
def test_run2_lift_test_calibrates_search():
    r = run({"lift_tests": [LIFT]})
    assert "search" in r.d.config["calibration"]


@needs
def test_refit_with_actuals_reads_day_first_dates_as_the_run_does():
    # pandas' default reads 13/01/2026 as NaT (month-first) and the merge kept 69 of 169 rows
    from mmm_tool.api.future_routes import _parsed_dates
    from mmm_tool.ingest.load import load_file
    from mmm_tool.ingest.types import Answers

    old, _ = load_file(RAW, Answers({}, True))
    new = pd.read_csv(HERE / "raw" / "MTR_actuals_2026Q1.csv", dtype=str)
    both = pd.concat([old, new[[c for c in old.columns if c in new.columns]]], ignore_index=True)
    d = _parsed_dates(both, "Week (dd/mm/yyyy)", {})
    assert d.notna().all() and d.iloc[-1] == pd.Timestamp("2026-03-30") and d.iloc[len(old)] == pd.Timestamp("2026-01-05")
    assert both.assign(_d=d).drop_duplicates("_d").shape[0] == 156 + 13                 # the 3 exact copies, then 13 new weeks


def test_a_short_event_is_not_the_same_as_a_mostly_zero_column():
    from mmm_tool.configure.calendar import same_event

    n = 170
    imputed, shortage = np.zeros(n), np.zeros(n)
    imputed[89] = 1
    shortage[165:] = 1
    assert not same_event(imputed, shortage)                                          # agree in 164/170 weeks, share none
    festive, flag = np.zeros(n), np.zeros(n)
    festive[40:56], flag[41:56] = 1, 1                                                # one boundary week differs
    assert same_event(festive, flag)
