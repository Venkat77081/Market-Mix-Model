"""Phase B tests against exports whose problems are planted at known places."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import gen
from mmm_tool.pipeline import run_pipeline
from mmm_tool.samples import MESSY_TRUTH, write_daily, write_messy


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("b")
    return {
        "clean": gen.write_clean(d),
        "dirty": gen.write_dirty_dates(d),
        "regions": gen.write_regions(d),
        "messy": write_messy(d),
        "daily": write_daily(d),
        "dir": d,
    }


def step(r, n):
    return next(s for s in r.b.steps if s.step == n)


# The messy export repeats 20 weeks; step 2 asks before Phase B merges them.
DUP = {"duplicate_dates": "merge"}


@pytest.fixture(scope="module")
def messy(files):
    return run_pipeline(files["messy"], DUP, accept_defaults=True)


def truth():
    return gen.make_insurance()


# End to end ----------------------------------------------------------------

def test_clean_file_passes_unchanged(files):
    r = run_pipeline(files["clean"], accept_defaults=True)
    assert r.b.status == "complete" and r.phase == "C"  # Phase C runs once B is complete
    t, raw = r.b.table, truth()
    assert len(t) == 156 and not t.isna().any().any()
    assert np.allclose(t["tv_spend"], raw["tv_spend"]) and np.allclose(t["new_policies"], raw["new_policies"])
    assert all(s.verdict.value == "pass" for s in r.b.steps)


def test_phase_b_waits_for_phase_a(files):
    r = run_pipeline(files["clean"])  # roles not confirmed
    assert r.phase == "A" and r.b is None


def test_messy_file_is_complete_and_weekly(messy):
    t = messy.b.table
    assert messy.status == "complete"
    assert len(t) == 156 and t["week_start"].is_unique
    assert (t["week_start"].diff().dropna() == pd.Timedelta(days=7)).all()
    assert not t.isna().any().any()


# Step 7 ------------------------------------------------------------------

def test_coerce_reads_indian_numbers_and_text_nulls(files):
    r = run_pipeline(files["dirty"], accept_defaults=True)
    cols = {c["column"]: c for c in step(r, 7).details["columns"]}
    assert cols["tv_spend"]["formatted"] > 0
    assert cols["search_clicks"]["text_nulls"] == 1
    assert np.allclose(r.b.table["tv_spend"], truth()["tv_spend"])


def test_yes_no_and_unreadable_cells(tmp_path):
    df = gen.make_insurance(weeks=110)
    df["festive_flag"] = np.where(np.arange(110) % 5 == 0, "Yes", "no")
    df["meta_spend"] = df["meta_spend"].astype(object)
    df.loc[3, "meta_spend"] = "approx 5L"
    p = tmp_path / "yn.csv"
    df.to_csv(p, index=False)
    r = run_pipeline(p, accept_defaults=True)
    cols = {c["column"]: c for c in step(r, 7).details["columns"]}
    assert cols["festive_flag"]["yes_no"] == 110
    assert cols["meta_spend"]["unreadable"] == 1 and cols["meta_spend"]["examples"] == ["approx 5L"]
    assert set(r.b.table["festive_flag"].unique()) == {0.0, 1.0}


# Step 8 ------------------------------------------------------------------

def test_duplicates_collapse_with_exact_counts(messy):
    d = step(messy, 8).details
    assert d["duplicate_dates"] == MESSY_TRUTH["duplicate_dates"]
    assert d["duplicate_rows"] == MESSY_TRUTH["duplicate_dates"]


def test_duplicate_spend_sums_back_to_truth(messy):
    t = messy.b.table.set_index("week_start")
    raw = truth().set_index("week_start")
    weeks = raw.index[10:30]
    assert np.allclose(t.loc[weeks, "search_spend"], raw.loc[weeks, "search_spend"])
    assert np.allclose(t.loc[weeks, "tv_spend_cr"], raw.loc[weeks, "tv_spend"])  # also converted from crore


def test_flags_take_the_maximum_not_the_sum(messy):
    t = messy.b.table
    assert t["promo_flag"].max() == 1.0
    assert step(messy, 8).details["flags_max"]


def test_unclassified_column_blocks_only_when_rows_merge(files, tmp_path):
    df = gen.make_insurance()
    df["mystery_metric"] = np.arange(156, dtype=float)
    p = tmp_path / "mystery.csv"
    df.to_csv(p, index=False)
    known = {"roles": {"mystery_metric": "control"}}  # an unrecognised number needs a role (step 3)
    r = run_pipeline(p, known, accept_defaults=True)
    assert r.status == "complete"  # nothing merges, so no rule is needed

    pd.concat([df, df.iloc[[5]]]).to_csv(p, index=False)
    known = {**DUP, "roles": {"mystery_metric": "control"}}
    r = run_pipeline(p, known, accept_defaults=False)
    s8 = step(r, 8)
    assert r.status == "needs_input" and s8.decisions[0].context["unclassified"] == ["mystery_metric"]
    r = run_pipeline(p, {**known, "aggregation_rules": {"mystery_metric": "sum"}}, accept_defaults=False)
    assert r.b.steps[-1].step > 8


def test_regions_add_up_to_national(files):
    r = run_pipeline(files["regions"], {"geo_mode": "national"}, accept_defaults=True)
    t, raw = r.b.table, gen.make_insurance(regions=True)
    assert len(t) == 156 and "region" not in t.columns and "population" not in t.columns
    assert np.isclose(t["new_policies"].sum(), raw["new_policies"].sum())
    assert t["tax_season_flag"].max() == 1.0  # max across regions, not 5
    assert step(r, 8).details["regions_rolled_up"] == 5
    assert "geo" not in r.b.coord_to_columns()["coord_to_columns"]


def test_regional_model_keeps_regions(files):
    r = run_pipeline(files["regions"], accept_defaults=True)
    assert len(r.b.table) == 780
    assert r.b.coord_to_columns()["coord_to_columns"]["geo"] == "region"


# Step 9 ------------------------------------------------------------------

def test_daily_becomes_weeks(files):
    r = run_pipeline(files["daily"], accept_defaults=True)
    t, d = r.b.table, step(r, 9).details
    assert (t["date"].dt.dayofweek == 0).all()  # Monday by default
    assert len(t) == 105 and len(d["partial_weeks_kept"]) == 2  # partial edge weeks kept by default
    raw = pd.read_csv(files["daily"], parse_dates=["date"])
    second = t["date"].iloc[1]
    week = raw[(raw["date"] >= second) & (raw["date"] < second + pd.Timedelta(days=7))]
    assert np.isclose(t["tv_spend"].iloc[1], week["tv_spend"].sum())
    assert np.isclose(t["premium_index"].iloc[1], week["premium_index"].mean())
    assert t["festive_flag"].max() == 1.0
    assert step(r, 9).warnings


def test_partial_edge_weeks_can_be_dropped(files):
    r = run_pipeline(files["daily"], {"partial_weeks": "drop"}, accept_defaults=True)
    assert len(r.b.table) == 103 and len(step(r, 9).details["partial_weeks_dropped"]) == 2


def test_week_start_answer(files):
    r = run_pipeline(files["daily"], {"week_start": "Sunday"}, accept_defaults=True)
    assert (r.b.table["date"].dt.dayofweek == 6).all()


# Step 10 -----------------------------------------------------------------

def test_missing_weeks_and_gaps_are_labelled(messy):
    d = step(messy, 10).details
    assert len(d["missing_weeks"]) == MESSY_TRUTH["missing_weeks"]
    cols = {c["column"]: c for c in d["columns"]}
    assert cols["radio_spend"]["not_started"] == MESSY_TRUTH["radio_starts_week"]
    assert cols["new_policies"]["missing"] == MESSY_TRUTH["kpi_blank_weeks"] + MESSY_TRUTH["missing_weeks"]
    t = messy.b.table
    assert (t["radio_spend"].iloc[: MESSY_TRUTH["radio_starts_week"]] == 0).all()
    assert t["kpi_imputed"].sum() == MESSY_TRUTH["kpi_blank_weeks"] + MESSY_TRUTH["missing_weeks"]
    assert messy.b.roles["kpi_imputed"] == "control"


def test_zero_is_not_missing(messy):
    t, raw = messy.b.table, truth()
    tv_off = raw.index[(raw["tv_spend"] == 0) & ~raw.index.isin([40, 41, 90, 120])]
    assert (t.loc[tv_off, "tv_spend_cr"] == 0).all()


def test_kpi_gaps_block_without_an_answer(files):
    r = run_pipeline(files["messy"], {**DUP, "roles": {}})
    assert r.status == "needs_input" and r.b.pending[0]["id"] == "kpi_gaps"


def test_kpi_gaps_trim(files):
    r = run_pipeline(files["messy"], {**DUP, "kpi_gaps": "trim"}, accept_defaults=True)
    t = r.b.table
    assert "kpi_imputed" not in t.columns
    # KPI gaps sit at weeks 40, 41, 60, 61, 90 and 120; the longest run without one is weeks 0-39.
    assert len(t) == 40 and t["week_start"].iloc[0] == truth()["week_start"].iloc[0]
    assert step(r, 10).details["trimmed_weeks"]


def test_media_holes_can_be_interpolated(tmp_path):
    df = gen.make_insurance()
    df["meta_spend"] = df["meta_spend"].astype(object)
    df.loc[50, "meta_spend"] = ""
    p = tmp_path / "hole.csv"
    df.to_csv(p, index=False)
    zero = run_pipeline(p, accept_defaults=True).b.table
    interp = run_pipeline(p, {"media_gaps": "interpolate"}, accept_defaults=True).b.table
    assert zero.loc[50, "meta_spend"] == 0
    assert np.isclose(interp.loc[50, "meta_spend"], (df.loc[49, "meta_spend"] + df.loc[51, "meta_spend"]) / 2)


# Step 11 -----------------------------------------------------------------

def test_outlier_week_gets_an_event_flag(messy):
    d = step(messy, 11).details
    week = str(truth()["week_start"].iloc[MESSY_TRUTH["outlier_week"]].date())
    assert [o["date"] for o in d["outliers"]] == [week]
    col = "event_" + week.replace("-", "_")
    assert messy.b.table[col].sum() == 1 and messy.b.roles[col] == "control"
    assert col in messy.b.coord_to_columns()["coord_to_columns"]["controls"]
    assert len(messy.b.table) == 156  # never deleted


def test_outlier_can_be_kept(files, messy):
    key = step(messy, 11).details["outliers"][0]["key"]
    r = run_pipeline(files["messy"], {**DUP, "outliers": {key: "keep"}}, accept_defaults=True)
    assert not any(c.startswith("event_") for c in r.b.table.columns)


# Step 12 -----------------------------------------------------------------

def test_negative_spend_blocks_then_is_fixed(files):
    answers = {**DUP, "roles": {}, "kpi_gaps": "interpolate"}
    r = run_pipeline(files["messy"], answers)
    assert r.status == "needs_input" and r.b.pending[0]["id"] == "negative_values"
    cells = {c["key"]: c for c in r.b.pending[0]["context"]["cells"]}
    assert len(cells) == MESSY_TRUTH["negative_cells"]

    meta_key = next(k for k in cells if k.startswith("meta_spend"))
    search_key = next(k for k in cells if k.startswith("search_spend"))
    r = run_pipeline(files["messy"], {**answers, "negative_values": {meta_key: 0, search_key: "keep"}}, accept_defaults=True)
    t = r.b.table.set_index("week_start")
    assert t.loc[cells[meta_key]["date"], "meta_spend"] == 0
    assert t.loc[cells[search_key]["date"], "search_spend"] == -40000


def test_negative_spend_correction_value(files, messy):
    key = next(c["key"] for c in step(messy, 12).details["negative"] if c["column"] == "meta_spend")
    r = run_pipeline(files["messy"], {**DUP, "negative_values": {key: 98000}}, accept_defaults=True)
    date = key.split("|")[1]
    assert r.b.table.set_index("week_start").loc[date, "meta_spend"] == 98000


def test_kpi_zero_and_index_jump(messy):
    d = step(messy, 12).details
    raw = truth()
    assert [z["date"] for z in d["kpi_zero"]] == [str(raw["week_start"].iloc[MESSY_TRUTH["kpi_zero_week"]].date())]
    assert any(j["column"] == "premium_index" and j["date"] == str(raw["week_start"].iloc[MESSY_TRUTH["index_jump_week"]].date()) for j in d["jumps"])
    assert not any(j["column"] == "competitor_spend" for j in d["jumps"])
    assert messy.b.table.loc[MESSY_TRUTH["kpi_zero_week"], "new_policies"] > 0  # interpolated by default


# Step 13 -----------------------------------------------------------------

def test_crore_column_converted(messy):
    d = step(messy, 13).details
    assert d["units_converted"] == [{"column": "tv_spend_cr", "unit": "cr", "multiplied_by": 1e7}]


def test_inflation_and_fixed_value(files):
    answers = {"roles": {"avg_premium": "excluded"}, "revenue_per_kpi_value": 14500,
               "inflation": {"base_year": 2025, "annual_rate_pct": 5}}
    r = run_pipeline(files["clean"], answers, accept_defaults=True)
    t, raw = r.b.table, truth()
    first_2023 = raw.index[raw["week_start"].dt.year == 2023][0]
    assert np.isclose(t.loc[first_2023, "tv_spend"], raw.loc[first_2023, "tv_spend"] * 1.05 ** 2)
    assert np.isclose(t["revenue_per_kpi"].iloc[-1], 14500)
    assert np.isclose(t["revenue_per_kpi"].iloc[0], 14500 * 1.05 ** 2)
    assert r.b.coord_to_columns()["coord_to_columns"]["revenue_per_kpi"] == "revenue_per_kpi"


def test_bad_inflation_answer_blocks(files):
    r = run_pipeline(files["clean"], {"inflation": {"base_year": 2025, "annual_rate_pct": 90}}, accept_defaults=True)
    assert r.status == "needs_input" and "between 0 and 50" in r.b.steps[-1].summary
