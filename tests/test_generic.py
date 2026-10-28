"""The tool with nothing domain-specific switched on: no keyword packs, no
project calendar. Files from other industries and countries (a US online
shop in dollars, a file with no currency at all) must go through with
their own currency, no calendar flags unless one is picked, and no Indian
or insurance words anywhere in what the user sees.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from mmm_tool import currency as cur
from mmm_tool.ingest.keywords import KeywordError, load_keywords
from mmm_tool.pipeline import run_pipeline

INDIA_OR_INSURANCE = ["₹", "rupee", "lakh", "crore", "diwali", "festive", "insurance", "policy", "policies", "premium"]


@pytest.fixture(autouse=True)
def generic(monkeypatch, tmp_path):
    monkeypatch.setenv("MMM_KEYWORD_PACKS", "")
    monkeypatch.setenv("MMM_CALENDARS", str(tmp_path / "no_calendars.yaml"))


def shop(weeks=156, seed=3, money_sign="$"):
    """A US online shop: orders, a revenue column, spend in dollars, a price index and a holiday flag."""
    rng = np.random.default_rng(seed)
    t = np.arange(weeks)
    week = pd.date_range("2022-01-03", periods=weeks, freq="7D")
    df = pd.DataFrame({"week": week.strftime("%Y-%m-%d")})
    orders = 5000 + 400 * np.sin(2 * np.pi * t / 52) + rng.normal(0, 150, weeks)
    for ch, base in (("tv", 9e4), ("search", 6e4), ("social", 4e4)):
        df[f"{ch}_spend"] = np.round(np.clip(rng.normal(base, base * 0.2, weeks), 0, None), -2)
        orders += 0.004 * df[f"{ch}_spend"]
    df["search_clicks"] = np.round(df["search_spend"] / 1.8)
    df["price_index"] = np.round(1 + rng.normal(0, 0.02, weeks), 3)
    df["holiday_flag"] = ((week.month == 12) & (week.day <= 24)).astype(int)
    df["orders"] = np.round(orders)
    df["revenue"] = np.round(df["orders"] * rng.normal(62, 2, weeks), 2)
    if money_sign:
        df = df.rename(columns={c: f"{c} ({money_sign})" for c in df.columns if c.endswith("_spend") or c == "revenue"})
    return df


def write(tmp_path, df, name="shop.csv"):
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


def text_of(r) -> str:
    out = {"a": r.a.to_dict() if r.a else None, "b": r.b.to_dict() if r.b else None,
           "c": r.c.to_dict() if r.c else None, "d": r.d.to_dict() if r.d else None}
    # The step 21 picker lists every available preset by name; that is a choice, not a default.
    return json.dumps(out, default=str).lower().replace("india_insurance", "").replace("india insurance", "")


def step(phase, n):
    return next(s for s in phase.steps if s.step == n)


# A US shop in dollars -----------------------------------------------------------------------------

def test_us_shop_runs_in_dollars_with_no_indian_or_insurance_words(tmp_path):
    r = run_pipeline(write(tmp_path, shop()), {}, accept_defaults=True)
    assert r.d is not None and r.d.status == "complete", r.last.steps[-1].summary
    roles = r.a.roles
    assert roles["orders"] == "kpi" and roles["revenue ($)"] == "revenue_for_kpi"
    assert roles["tv_spend ($)"] == "media_spend" and roles["price_index"] == "control"
    assert r.a.kpi["currency"]["code"] == "USD" and r.a.kpi["roi_units"] == "$ per $"
    assert "$" in step(r.b, 13).summary                                   # revenue per KPI median in dollars
    assert not [c for c in r.d.table.columns if c.startswith("cal_")]     # no calendar unless one is picked
    found = [w for w in INDIA_OR_INSURANCE if w in text_of(r)]
    print("\nIndian or insurance words in a US run:", found or "none")
    assert found == []


def test_no_currency_shows_plain_numbers(tmp_path):
    r = run_pipeline(write(tmp_path, shop(money_sign="")), {}, accept_defaults=True)
    assert r.d is not None and r.d.status == "complete"
    assert r.a.kpi["currency"]["code"] is None
    assert r.a.kpi["roi_units"] == "return per unit of spend"
    assert "₹" not in text_of(r) and "rupee" not in text_of(r)


def test_currency_can_be_changed(tmp_path):
    r = run_pipeline(write(tmp_path, shop()), {"currency": "EUR"}, accept_defaults=True)
    assert r.a.kpi["currency"]["symbol"] == "€" and r.a.kpi["roi_units"] == "€ per €"
    r = run_pipeline(write(tmp_path, shop()), {"currency": "XYZ"}, accept_defaults=True)
    assert r.status == "needs_input" and "Unknown currency" in r.a.steps[-1].summary


@pytest.mark.parametrize("cells, code", [(["$1,200", "$950"], "USD"), (["€1,200", "€950"], "EUR"),
                                         (["£1,200", "£950"], "GBP"), (["₹1,20,000", "₹95,000"], "INR"),
                                         (["1200", "950"], None)])
def test_currency_detected_from_cells(cells, code):
    df = pd.DataFrame({"spend": cells * 10})
    assert cur.detect(df, ["spend"]) == code


def test_number_grouping_follows_the_currency():
    inr, usd, none = cur.get("INR"), cur.get("USD"), cur.get(None)
    assert cur.money(1234567, inr) == "₹12,34,567" and cur.money(1234567, usd) == "$1,234,567"
    assert cur.money(1234567, none) == "1,234,567"
    assert cur.money_short(23_000_000, inr) == "₹2.3 Cr" and cur.money_short(2_300_000, usd) == "$2.3M"
    assert cur.per(none) == "return per unit of spend" and cur.per(usd) == "$ per $"


# Calendar presets ----------------------------------------------------------------------------------

def test_calendar_preset_is_none_by_default_and_can_be_picked(tmp_path):
    p = write(tmp_path, shop())
    r = run_pipeline(p, {}, accept_defaults=True)
    d = next(x for x in step(r.d, 21).decisions if x.id == "calendar_preset")
    assert d.default == "none" and {"us_retail", "india_insurance"} <= set(d.options)
    r = run_pipeline(p, {"calendar_preset": "us_retail"}, accept_defaults=True)
    added = r.d.config["calendar"]["added"]
    print("\nus_retail adds:", added)
    assert "cal_black_friday" in added
    s21 = step(r.d, 21)
    assert any("christmas" in e["event"] for e in s21.details["events"])   # already in the data as holiday_flag, or added


def test_project_calendar_file_adds_presets_and_sets_the_default(tmp_path, monkeypatch):
    f = tmp_path / "calendars.yaml"
    f.write_text("default_preset: summer_sale\npresets:\n  summer_sale:\n    label: Summer sale\n    events:\n"
                 "      - {name: summer_sale, label: Summer sale, recurring: {from: '07-01', to: '07-14'}}\n", encoding="utf-8")
    monkeypatch.setenv("MMM_CALENDARS", str(f))
    r = run_pipeline(write(tmp_path, shop()), {}, accept_defaults=True)
    assert r.d.config["calendar"]["preset"] == "summer_sale" and "cal_summer_sale" in r.d.config["calendar"]["added"]


# Keyword packs -------------------------------------------------------------------------------------

def test_insurance_words_only_with_the_pack(tmp_path, monkeypatch):
    df = shop(money_sign="").rename(columns={"orders": "new_policies"})
    r = run_pipeline(write(tmp_path, df), {}, accept_defaults=True)
    assert r.a.roles.get("new_policies") != "kpi" or r.status != "complete" and "new_policies" in r.last.steps[-1].summary
    monkeypatch.setenv("MMM_KEYWORD_PACKS", "insurance")
    r = run_pipeline(write(tmp_path, df), {}, accept_defaults=True)
    assert r.a.roles["new_policies"] == "kpi"


def test_india_pack_reads_lakh_and_suggests_rupees(tmp_path, monkeypatch):
    df = shop(money_sign="")
    df["tv_spend"] = [f"{v / 1e5:.2f} L" for v in df["tv_spend"]]
    monkeypatch.setenv("MMM_KEYWORD_PACKS", "india")
    r = run_pipeline(write(tmp_path, df), {}, accept_defaults=True)
    assert r.a.kpi["currency"]["code"] == "INR"                       # the pack's default, nothing in the file
    assert r.b.table["tv_spend"].median() > 1e4                       # "0.91 L" read as 91,000


def test_direction_hints_are_neutral_without_packs(tmp_path, monkeypatch):
    df = shop()
    df["premium_index"] = np.round(1 + np.random.default_rng(9).normal(0, 0.02, len(df)), 3)
    r = run_pipeline(write(tmp_path, df), {}, accept_defaults=True)
    sug = step(r.d, 21).details["suggest_direction"]
    assert sug.get("price_index") == "lowers" and "premium_index" not in sug
    monkeypatch.setenv("MMM_KEYWORD_PACKS", "insurance")
    r = run_pipeline(write(tmp_path, df), {}, accept_defaults=True)
    assert step(r.d, 21).details["suggest_direction"].get("premium_index") == "lowers"


def test_unknown_pack_is_an_error_and_a_pack_can_be_a_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MMM_KEYWORD_PACKS", "nonesuch")
    with pytest.raises(KeywordError, match="Unknown keyword pack"):
        load_keywords()
    mine = tmp_path / "retail.yaml"
    mine.write_text("rules:\n  - role: kpi\n    words: [baskets]\n    confidence: high\n", encoding="utf-8")
    monkeypatch.setenv("MMM_KEYWORD_PACKS", str(mine))
    assert load_keywords().match("baskets_sold").role == "kpi"
