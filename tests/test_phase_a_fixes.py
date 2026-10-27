"""Phase A fixes: month-only dates, wide regions, outcome columns, repeats.

General rules are tested on synthetic frames; the acceptance criteria are
tested on the files in tests/data/phase_a_tests (answer key in
expected_phase_a.json).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import gen
from mmm_tool.ingest import Answers, run_phase_a
from mmm_tool.ingest.dates import classify
from mmm_tool.ingest.wide import detect_wide_regions

DATA = Path(__file__).parent / "data" / "phase_a_tests"


def run(name_or_path, answers=None, defaults=True):
    path = name_or_path if isinstance(name_or_path, Path) else DATA / name_or_path
    return run_phase_a(path, Answers(answers or {}, accept_defaults=defaults))


def step(r, n):
    return next(s for s in r.steps if s.step == n)


def decisions(r):
    return {d.id: d for s in r.steps for d in s.decisions}


def sugg(r):
    return {x["column"]: x for x in step(r, 3).details["suggestions"]}


def write(tmp_path, df, name="t.csv"):
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


# Fix 1: month-only dates ---------------------------------------------------

@pytest.mark.parametrize("text", ["2023-01", "2023/01", "2023/1", "01/2023", "1-2023", "Jan-2023", "Jan 2023", "January 2023", "jan 2023"])
def test_month_only_dates_are_the_first_of_the_month(text):
    assert classify(text) == ("month_only", pd.Timestamp("2023-01-01"))


def test_month_year_is_never_day_first():
    assert classify("03/2024") == ("month_only", pd.Timestamp("2024-03-01"))  # not 3 Feb or 2 Mar
    assert classify("13/2024")[0] == "unparseable"


def test_text_dates_never_borrow_todays_day():
    assert classify("05-Jan-23") == ("text_month", pd.Timestamp("2023-01-05"))
    assert classify("Jan 5")[0] == "unparseable"  # no year


def test_a12_monthly():
    r = run("A12_monthly.csv")
    s2, s6 = step(r, 2).details, step(r, 6)
    assert (s2["rows"], s2["unique_dates"], s2["first"], s2["last"]) == (36, 36, "2023-01-01", "2025-12-01")
    assert s6.summary.lower().startswith("monthly, 36 periods")
    assert any("under 104 weeks" in w for w in s6.warnings)


# Fix 2: wide regions -------------------------------------------------------

def test_wide_detection_rules():
    assert detect_wide_regions(["week", "policies_TN", "tv_spend_TN", "policies_KA", "tv_spend_KA"])["tags"] == ["TN", "KA"]
    assert detect_wide_regions(["week", "North sales", "North tv spend", "South sales", "South tv spend"])["mode"] == "prefix"
    assert detect_wide_regions(["week", "policies-east", "tv-spend-east", "policies-west", "tv-spend-west"])["tags"] == ["east", "west"]
    # a tag on only one stem is not a region
    assert detect_wide_regions(["week", "policies_TN", "policies_KA", "tv_spend"]) is None
    # metric words, channel prefixes and years are not regions
    assert detect_wide_regions(["week", "tv_spend", "tv_grps", "search_spend", "search_grps"]) is None
    assert detect_wide_regions(["week", "tv_spend", "tv_impressions", "meta_spend", "meta_impressions"]) is None
    assert detect_wide_regions(["week", "sales_2023", "sales_2024", "spend_2023", "spend_2024"]) is None


def test_a14_reshape():
    r = run("A14_regions_wide.csv")
    d = decisions(r)
    assert d["wide_regions"].default == "reshape" and d["wide_regions"].options == ["reshape", "sum", "keep"]
    assert r.status == "complete" and len(r.table) == 468
    assert r.kpi["kpi"] == "policies"
    assert sorted(c.spend for c in r.channels) == ["search_spend", "tv_spend"]
    assert step(r, 6).details["regions"] == ["KA", "MH", "TN"]
    assert any(x["decision"] == "wide_regions" for x in r.decision_log)


def test_a14_waits_for_the_choice():
    r = run("A14_regions_wide.csv", defaults=False)
    assert r.status == "needs_input" and r.pending[0]["id"] == "wide_regions"
    assert "roles" not in decisions(r)  # asked before role mapping


def test_a14_region_column_name():
    r = run("A14_regions_wide.csv", {"region_column": "state"})
    assert "state" in r.table.columns and r.roles["state"] == "geo"


def test_a14_sum_to_national():
    r = run("A14_regions_wide.csv", {"wide_regions": "sum"})
    raw = pd.read_csv(DATA / "A14_regions_wide.csv")
    assert len(r.table) == 156 and step(r, 6).details["regions"] == []
    assert np.allclose(r.table["policies"], raw[["policies_TN", "policies_KA", "policies_MH"]].sum(axis=1))


def test_sum_asks_how_to_combine_non_additive_columns(tmp_path):
    df = pd.read_csv(DATA / "A14_regions_wide.csv")
    for t in ("TN", "KA", "MH"):
        df[f"price_index_{t}"] = 100.0 + np.arange(len(df)) % 7
    p = write(tmp_path, df)
    r = run(p, {"wide_regions": "sum"}, defaults=False)
    assert r.pending[0]["id"] == "wide_regions_combine" and r.pending[0]["default"] == {"price_index": "mean"}
    r = run(p, {"wide_regions": "sum", "wide_regions_combine": {"price_index": "mean"}, "roles": {}}, defaults=False)
    assert np.allclose(r.table["price_index"], df["price_index_TN"])


def test_keep_never_crowns_one_regions_outcome():
    r = run("A14_regions_wide.csv", {"wide_regions": "keep"})
    s = sugg(r)
    assert all(s[f"policies_{t}"]["role"] == "needs_role" for t in ("TN", "KA", "MH"))
    assert not any(v["role"] == "excluded" for v in s.values())
    assert any("other regions' sales" in w for w in step(r, 3).warnings)
    assert r.status == "needs_input"


def test_a13_long_regions_unchanged():
    r = run("A13_regions_long.csv")
    s = sugg(r)
    assert "wide_regions" not in decisions(r)
    assert (s["state"]["role"], s["adult_population"]["role"]) == ("geo", "population")
    assert len(step(r, 6).details["regions"]) == 5 and "geo_mode" in decisions(r)


# Fix 3: outcome columns ----------------------------------------------------

def test_a01_premium_is_revenue_for_kpi():
    r = run("A01_clean_weekly.csv")
    s = sugg(r)["new_premium_inr"]
    assert (s["role"], s["confidence"]) == ("revenue_for_kpi", "high")
    assert "₹9,310" in s["reason"] and "₹11,096" in s["reason"]
    assert decisions(r)["kpi_choice"].options == ["count", "revenue"]
    assert r.kpi["kpi_type"] == "non_revenue" and r.kpi["revenue_per_kpi_column"] == "revenue_per_kpi"
    assert 9300 < r.table["revenue_per_kpi"].median() < 11100  # per policy, never the total
    assert "new_premium_inr" not in r.table.columns
    assert "new_premium_inr" not in r.coord_to_columns()["coord_to_columns"].get("controls", [])


def test_a01_avg_premium_index_is_a_control_not_revenue_per_policy():
    r = run("A01_clean_weekly.csv")
    assert sugg(r)["avg_premium_index"]["role"] == "control"


def test_kpi_choice_revenue():
    r = run("A01_clean_weekly.csv", {"kpi_choice": "revenue"})
    assert r.kpi["kpi"] == "new_premium_inr" and r.kpi["kpi_type"] == "revenue"
    assert r.roles["new_policies"] == "excluded" and r.kpi["revenue_per_kpi_column"] is None


def test_unrecognised_number_needs_a_role(tmp_path):
    df = gen.make_insurance()
    df["xyz_metric"] = np.random.default_rng(2).normal(50, 9, len(df))
    r = run(write(tmp_path, df))
    assert sugg(r)["xyz_metric"]["role"] == "needs_role"
    assert r.status == "needs_input" and "Choose a role for: xyz_metric" in step(r, 3).summary
    assert run(write(tmp_path, df), {"roles": {"xyz_metric": "control"}}).status == "complete"


def test_control_that_moves_with_the_kpi_is_a_possible_outcome(tmp_path):
    df = gen.make_insurance()
    # Moves with the KPI (correlation > 0.8) without being a constant multiple of it.
    k = df["new_policies"]
    df["vehicle_registrations"] = (k - k.min()) * 50 + 1000 + np.random.default_rng(3).normal(0, 1500, len(df))
    s = sugg(run(write(tmp_path, df)))["vehicle_registrations"]
    assert s["role"] == "needs_role" and "possible outcome" in s["reason"]


def test_rupee_sign_reads_as_money():
    df = gen.make_insurance()
    df["Premium ₹"] = df["new_policies"] * np.random.default_rng(4).normal(14000, 300, len(df))
    from mmm_tool.ingest.roles import suggest_roles
    s = {x.column: x for x in suggest_roles(df, "week_start")}
    assert s["Premium ₹"].role == "revenue_for_kpi"


# Fix 4: repeated dates and identical columns -------------------------------

def test_unique_dates_reported_a05():
    s2 = step(run("A05_mixed_dates.csv"), 2)
    assert "156 rows, 156 unique dates" in s2.summary
    assert (s2.details["first"], s2.details["last"]) == ("2023-01-02", "2025-12-22")


def test_repeated_dates_stop_unless_explained(tmp_path):
    df = gen.make_insurance()
    p = write(tmp_path, pd.concat([df, df.iloc[[3, 9]]]))
    r = run(p, defaults=False)
    assert r.steps[-1].step == 2 and r.pending[0]["id"] == "duplicate_dates"
    assert "158 rows, 156 unique dates" in r.steps[-1].summary
    r = run(p, {"duplicate_dates": "merge"})
    assert r.status == "complete" and step(r, 2).warnings


def test_regions_explain_repeated_dates():
    s2 = step(run("A13_regions_long.csv"), 2)
    assert s2.details["repeats_explained_by"] == "state" and "780 rows, 156 unique dates" in s2.summary


def test_identical_columns_ask_which_to_keep(tmp_path):
    df = gen.make_insurance()
    df["tv_spend_copy"] = df["tv_spend"]
    p = write(tmp_path, df)
    r = run(p, defaults=False)
    d = decisions(r)["identical:tv_spend"]
    assert d.options == ["tv_spend", "tv_spend_copy"] and d.pending
    r = run(p, {"identical:tv_spend": "tv_spend_copy"})
    assert r.roles["tv_spend"] == "excluded" and r.roles["tv_spend_copy"] == "media_spend"
