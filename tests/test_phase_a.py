"""Phase A tests against synthetic exports with known structure."""
from __future__ import annotations

import pandas as pd
import pytest

import gen
from mmm_tool.ingest import Answers, Verdict, run_phase_a
from mmm_tool.ingest.dates import classify, parse_dates
from mmm_tool.ingest.load import detect_encoding
from mmm_tool.ingest.roles import channel_of, suggest_roles, validate_roles
from mmm_tool.ingest.values import to_number


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    return {
        "clean": gen.write_clean(d),
        "excel": gen.write_excel_with_preamble(d),
        "dirty": gen.write_dirty_dates(d),
        "monthly": gen.write_ambiguous_monthly(d),
        "regions": gen.write_regions(d),
        "rf": gen.write_rf(d),
        "dir": d,
    }


def step(result, n):
    return next(s for s in result.steps if s.step == n)


# Step 1 -----------------------------------------------------------------

def test_clean_csv_completes_with_defaults(data):
    r = run_phase_a(data["clean"], accept_defaults=True)
    assert r.status == "complete"
    assert [s.step for s in r.steps] == [1, 2, 3, 4, 5, 6]
    assert r.table.shape == (156, 21)


def test_excel_picks_data_sheet_and_skips_preamble(data):
    r = run_phase_a(data["excel"], accept_defaults=True)
    s1 = step(r, 1)
    assert s1.details["sheet"] == "Weekly data"
    assert s1.details["header_row"] == 4
    assert s1.details["preamble_rows"] == 3  # two title lines and a blank line above the header
    assert s1.details["preamble_text"][0] == "Weekly media and sales report"
    assert r.status == "complete"


def test_sheet_answer_overrides_default(data):
    r = run_phase_a(data["excel"], {"sheet": "Notes"}, accept_defaults=True)
    assert r.status != "complete"  # the Notes sheet has no dates


def test_cp1252_semicolon_file(data):
    r = run_phase_a(data["dirty"], accept_defaults=True)
    s1 = step(r, 1)
    assert s1.details["encoding"] == "cp1252"
    assert s1.details["delimiter"] == ";"
    assert "Remisé_promo_%" in s1.details["columns"]


def test_detect_encoding_variants():
    assert detect_encoding("a,b\n1,2".encode("utf-8")) == "utf-8"
    assert detect_encoding("﻿a,b".encode("utf-8")) == "utf-8-sig"
    assert detect_encoding("Prämie,Café\n1,2".encode("cp1252")) == "cp1252"


def test_unsupported_file_fails(tmp_path):
    p = tmp_path / "x.json"
    p.write_text("{}")
    r = run_phase_a(p)
    assert r.status == "failed" and r.steps[0].verdict == Verdict.FAIL


# Step 2 -----------------------------------------------------------------

def test_classify_rows():
    assert classify("2024-03-04")[0] == "iso"
    assert classify("25/03/2024") == ("dayfirst", pd.Timestamp("2024-03-25"))
    assert classify("03/25/2024") == ("monthfirst", pd.Timestamp("2024-03-25"))
    assert classify("03/04/2024")[0] == "ambiguous"
    assert classify("04/04/2024")[0] == "same_either_way"
    assert classify("05-Jan-2024") == ("text_month", pd.Timestamp("2024-01-05"))
    assert classify(45292)[0] == "excel_serial"  # 2024-01-01
    assert classify("N/A")[0] == "unparseable"


def test_mixed_ddmm_and_iso_parse_to_true_weeks(data):
    r = run_phase_a(data["dirty"], accept_defaults=True)
    truth = gen.make_insurance()["week_start"]
    assert (r.table["week_start"].reset_index(drop=True) == truth).all()
    assert step(r, 2).details["ambiguous_resolved_as"] == "dayfirst"


def test_all_ambiguous_dates_block_until_answered(data):
    r = run_phase_a(data["monthly"])
    assert r.status == "needs_input"
    assert r.steps[-1].step == 2
    pending = r.pending[0]
    assert pending["id"] == "date_order" and pending["default"] == "dayfirst"

    r2 = run_phase_a(data["monthly"], {"date_order": "dayfirst"}, accept_defaults=True)
    assert r2.status == "complete"
    assert r2.table["month"].iloc[1] == pd.Timestamp("2023-02-01")
    assert r2.granularity["frequency"] == "monthly"


def test_unparseable_dates_block_then_drop(tmp_path):
    df = gen.make_insurance(weeks=120)
    df["week_start"] = df["week_start"].dt.strftime("%Y-%m-%d")
    df.loc[[10, 50], "week_start"] = ["not a date", ""]
    p = tmp_path / "bad_dates.csv"
    df.to_csv(p, index=False)

    r = run_phase_a(p, accept_defaults=True)
    assert r.status == "needs_input" and r.pending[0]["id"] == "unparseable_dates"

    r2 = run_phase_a(p, {"unparseable_dates": "drop_rows"}, accept_defaults=True)
    assert r2.status == "complete" and len(r2.table) == 118


def test_parse_dates_fails_without_date_column():
    df = pd.DataFrame({"a": ["x", "y", "z"], "b": [1, 2, 3]})
    out, _, res = parse_dates(df, Answers())
    assert out is None and res.verdict == Verdict.FAIL


# Step 3 -----------------------------------------------------------------

def test_role_suggestions_for_insurance_columns(data):
    df = gen.make_insurance(rf=True)
    df["branded_search_queries"] = 1.0
    df["gross_written_premium"] = 1.0
    roles = {s.column: s.role for s in suggest_roles(df, "week_start")}
    assert roles["week_start"] == "date"
    assert roles["new_policies"] == "kpi"
    assert roles["avg_premium"] == "revenue_per_kpi"
    assert roles["tv_spend"] == "media_spend"
    assert roles["tv_grps"] == "media"
    assert roles["youtube_reach"] == "reach"
    assert roles["youtube_frequency"] == "frequency"
    assert roles["premium_index"] == "control"
    assert roles["competitor_spend"] == "control"
    assert roles["tax_season_flag"] == "control"
    assert roles["discount_pct"] == "non_media_treatment"
    assert roles["email_sends"] == "organic_media"
    assert roles["branded_search_queries"] == "excluded"
    # A second outcome-like column that doesn't track the KPI waits for the user (never silently dropped).
    assert roles["gross_written_premium"] == "needs_role"


def test_roles_block_until_confirmed(data):
    r = run_phase_a(data["clean"])
    assert r.status == "needs_input"
    assert r.steps[-1].step == 3 and r.pending[0]["id"] == "roles"


def test_partial_role_override(data):
    r = run_phase_a(data["clean"], {"roles": {"vehicle_registrations": "excluded"}}, accept_defaults=True)
    assert r.status == "complete"
    assert "vehicle_registrations" not in r.table.columns
    assert "vehicle_registrations" not in r.coord_to_columns()["coord_to_columns"]["controls"]


def test_roles_without_media_block(data):
    spend = {f"{c}_spend": "excluded" for c in gen.CHANNELS}
    r = run_phase_a(data["clean"], {"roles": spend}, accept_defaults=True)
    assert r.status == "needs_input"
    assert "paid media" in r.steps[-1].summary


def test_validate_roles_rules():
    base = {"d": "date", "k": "kpi", "s": "media_spend"}
    assert validate_roles(base) == []
    assert validate_roles({**base, "k2": "kpi"})
    assert validate_roles({**base, "p": "population"})  # population needs geo
    assert validate_roles({**base, "x": "nonsense"})


def test_channel_names():
    assert channel_of("tv_spend") == "tv"
    assert channel_of("TV Spend (₹)") == "tv"
    assert channel_of("search_clicks") == "search"
    assert channel_of("YT_Impressions") == "youtube"
    assert channel_of("meta_impressions") == "meta"


# Step 4 -----------------------------------------------------------------

def test_pairs_spend_with_exposure(data):
    r = run_phase_a(data["clean"], accept_defaults=True)
    ch = {c.name: c for c in r.channels}
    assert ch["tv"].exposure == "tv_grps"
    assert ch["search"].exposure == "search_clicks"
    assert ch["meta"].exposure == "meta_impressions"
    assert ch["aggregator"].exposure_is_spend and ch["aggregator"].exposure == "aggregator_spend"
    assert len(r.channels) == 9


def test_reach_frequency_and_unpaired_exposure(data):
    r = run_phase_a(data["rf"], accept_defaults=True)
    ch = {c.name: c for c in r.channels}
    assert ch["youtube"].kind == "rf"
    assert (ch["youtube"].reach, ch["youtube"].frequency, ch["youtube"].spend) == ("youtube_reach", "youtube_frequency", "youtube_spend")
    assert r.roles["print_impressions"] == "organic_media"
    m = r.coord_to_columns()
    assert m["coord_to_columns"]["rf_spend"] == ["youtube_spend"]
    assert "youtube_spend" not in m["coord_to_columns"]["media_spend"]


def test_unpaired_exposure_can_be_dropped(data):
    r = run_phase_a(data["rf"], {"unpaired:print_impressions": "excluded"}, accept_defaults=True)
    assert r.roles["print_impressions"] == "excluded"
    assert "print_impressions" not in r.table.columns


def test_second_exposure_metric_is_dropped(tmp_path):
    df = gen.make_insurance()
    df["search_impressions"] = df["search_clicks"] * 40
    p = tmp_path / "two_metrics.csv"
    df.to_csv(p, index=False)
    r = run_phase_a(p, accept_defaults=True)
    ch = {c.name: c for c in r.channels}
    assert ch["search"].exposure == "search_impressions"  # impressions preferred
    assert r.roles["search_clicks"] == "excluded"


# Step 5 -----------------------------------------------------------------

def test_non_revenue_kpi_with_revenue_column(data):
    r = run_phase_a(data["clean"], accept_defaults=True)
    assert r.kpi["kpi_type"] == "non_revenue"
    # Step 5 records where the value per unit comes from; Phase B calculates revenue per KPI after cleaning.
    assert r.kpi["revenue_source"] == "per_unit_column" and r.kpi["revenue_column"] == "avg_premium"
    assert r.kpi["revenue_per_kpi_column"] is None and "revenue_per_kpi" not in r.coord_to_columns()["coord_to_columns"]


def test_non_revenue_kpi_fixed_value_or_kpi_units(data):
    drop = {"roles": {"avg_premium": "excluded"}}
    r = run_phase_a(data["clean"], drop, accept_defaults=True)
    assert r.kpi["roi_units"] == "new_policies per ₹"
    assert any("KPI units" in s.summary for s in r.steps)

    r2 = run_phase_a(data["clean"], {**drop, "revenue_per_kpi_value": 14500}, accept_defaults=True)
    assert r2.kpi["revenue_per_kpi_value"] == 14500.0
    assert r2.coord_to_columns()["revenue_per_kpi_value"] == 14500.0


def test_revenue_kpi(data):
    roles = {"new_policies": "excluded", "gross_written_premium": "kpi", "avg_premium": "excluded"}
    r = run_phase_a(data["rf"], {"roles": roles}, accept_defaults=True)
    assert r.kpi["kpi_type"] == "revenue"
    assert r.kpi["roi_units"] == "₹ per ₹"


# Step 6 -----------------------------------------------------------------

def test_weekly_national(data):
    r = run_phase_a(data["clean"], accept_defaults=True)
    g = r.granularity
    assert (g["frequency"], g["periods"], g["weeks_of_history"], g["week_start"]) == ("weekly", 156, 156, "Monday")
    assert g["geo_mode"] == "national"
    assert "geo" not in r.coord_to_columns()["coord_to_columns"]


def test_regions_default_to_regional_with_population(data):
    r = run_phase_a(data["regions"], accept_defaults=True)
    assert r.granularity["geo_mode"] == "regional"
    assert len(r.granularity["regions"]) == 5
    m = r.coord_to_columns()["coord_to_columns"]
    assert (m["geo"], m["population"]) == ("region", "population")


def test_regional_without_population_blocks(data):
    roles = {"roles": {"population": "excluded"}}
    r = run_phase_a(data["regions"], roles, accept_defaults=True)
    assert r.status == "complete"
    assert r.granularity["geo_mode"] == "national"

    r2 = run_phase_a(data["regions"], {**roles, "geo_mode": "regional"}, accept_defaults=True)
    assert r2.status == "needs_input" and "population" in r2.steps[-1].summary


def test_daily_data_is_flagged_for_resampling(tmp_path):
    df = gen.make_insurance(weeks=800)
    df["week_start"] = pd.date_range("2023-01-01", periods=800, freq="D").strftime("%Y-%m-%d")
    p = tmp_path / "daily.csv"
    df.to_csv(p, index=False)
    r = run_phase_a(p, accept_defaults=True)
    assert r.granularity["frequency"] == "daily"
    assert step(r, 6).verdict == Verdict.WARN


# Output -----------------------------------------------------------------

def test_coord_to_columns_lists_align(data):
    r = run_phase_a(data["clean"], accept_defaults=True)
    out = r.coord_to_columns()
    m = out["coord_to_columns"]
    assert len(m["media"]) == len(m["media_spend"]) == 9
    for exp, spend in zip(m["media"], m["media_spend"]):
        assert out["media_to_channel"][exp] == out["media_spend_to_channel"][spend]
    assert set(m["controls"]) == {"premium_index", "competitor_spend", "vehicle_registrations", "tax_season_flag"}
    assert m["non_media_treatments"] == ["discount_pct"]
    assert m["organic_media"] == ["email_sends"]


def test_decision_log_records_source(data):
    r = run_phase_a(data["clean"], {"kpi_type": "non_revenue"}, accept_defaults=True)
    log = {e["decision"]: e["source"] for e in r.decision_log}
    assert log["kpi_type"] == "user"
    assert log["roles"] == "default"


def test_to_dict_is_json_ready(data):
    import json

    r = run_phase_a(data["clean"], accept_defaults=True)
    json.dumps(r.to_dict(), default=str)


def test_to_number_handles_indian_formats():
    assert to_number("1,20,000") == 120000
    assert to_number("₹ 4,500") == 4500
    assert to_number("(250)") == -250
    assert pd.isna(to_number("N/A"))
