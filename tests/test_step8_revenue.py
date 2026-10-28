"""Step 8 (duplicates, split weeks, revisions) and revenue per KPI.

Synthetic files from any domain: the logic may only use step 3 roles, the
values and the region column. INS_FULL tests run when that file is in
tests/data (the expected numbers come from the test set's README).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mmm_tool.clean.duplicates import sig_round
from mmm_tool.pipeline import run_pipeline

WEEKS = 30
ROLES = {"date": "date"}


def weekly(weeks: int = WEEKS, seed: int = 0, regions: list[str] | None = None, revenue: bool = False,
           rf: bool = False) -> pd.DataFrame:
    """A steady weekly export: orders around 1,000 a week, three spend channels, a price index
    and a 0/1 promo flag. Optionally a sales value column (₹ per order varies), regions and an
    rf channel with reach, frequency and impressions."""
    rng = np.random.default_rng(seed)
    parts = []
    for g in regions or [None]:
        n = weeks
        df = pd.DataFrame({"week": pd.date_range("2024-01-01", periods=n, freq="7D")})
        if g is not None:
            df.insert(1, "region", g)
        df["orders"] = np.round(1000 * (1 + rng.normal(0, 0.03, n)))
        df["radio_spend"] = np.round(rng.normal(2e5, 1e4, n), -2)
        df["search_spend"] = np.round(rng.normal(3e5, 1.5e4, n), -2)
        df["search_clicks"] = np.round(rng.normal(9000, 300, n))
        if rf:
            df["video_spend"] = np.round(rng.normal(4e5, 2e4, n), -2)
            df["video_reach"] = np.round(rng.normal(6e4, 2e3, n))
            df["video_frequency"] = np.round(rng.normal(3, 0.1, n), 2)
            df["video_impressions"] = np.round(df["video_reach"] * df["video_frequency"])
        df["price_index"] = np.round(1 + rng.normal(0, 0.01, n), 4)
        df["promo_flag"] = (rng.random(n) < 0.2).astype(int)
        if revenue:
            df["sales_value"] = np.round(df["orders"] * rng.normal(10000, 300, n), -2)
        if g is not None:
            df["population"] = 1_000_000
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


MONEY = ["orders", "radio_spend", "search_spend", "search_clicks", "sales_value", "video_spend", "video_impressions"]


def split(df: pd.DataFrame, i: int, shares=(0.5, 0.5)) -> pd.DataFrame:
    """Row i as several rows each holding a share of the additive columns."""
    rows = []
    for s in shares:
        r = df.iloc[[i]].copy()
        cols = [c for c in MONEY if c in r]
        r[cols] = r[cols] * s
        rows.append(r)
    return pd.concat([df.drop(index=df.index[i])] + rows, ignore_index=True)


def write(tmp_path: Path, df: pd.DataFrame, name="file.csv") -> Path:
    p = tmp_path / name
    df.assign(week=pd.to_datetime(df["week"]).dt.strftime("%Y-%m-%d")).to_csv(p, index=False)
    return p


def run_b(path, answers=None, defaults=True):
    answers = {"duplicate_dates": "merge", **(answers or {})}
    answers.setdefault("roles", {})                                    # confirm step 3's roles
    r = run_pipeline(path, answers, accept_defaults=defaults)
    return r


def s8(r):
    return next(s for s in r.b.steps if s.step == 8)


def rule(r, col):
    return next(x for x in s8(r).details["rules"] if x["column"] == col)


def week(df, date, region=None):
    t = df[pd.to_datetime(df["week"]) == pd.Timestamp(date)]
    return t if region is None else t[t["region"] == region]


# 1. Revenue per unit from split rows ---------------------------------------------------

def test_split_rows_give_total_revenue_over_total_units(tmp_path):
    df = weekly(revenue=True)
    i, date = 12, weekly()["week"][12]
    df = split(df, i)
    rows = week(df, date).index
    df.loc[rows, "orders"] = [100, 900]
    df.loc[rows, "sales_value"] = [100 * 8000, 900 * 12000]
    r = run_b(write(tmp_path, df), {"roles": {"sales_value": "revenue_for_kpi"}})
    t = r.b.table.set_index("week")
    got = float(t.loc[pd.Timestamp(date), "revenue_per_kpi"])
    print(f"\nexpected ₹11,600, got ₹{got:,.1f} (an average of the two ratios would be ₹10,000)")
    assert s8(r).details["groups"][0]["kind"] == "split"
    assert got == pytest.approx(11600)


def test_value_per_unit_column_is_combined_total_over_total(tmp_path):
    df = weekly()
    df["value_per_order"] = 10000.0
    date = df["week"][12]
    df = split(df, 12)
    rows = week(df, date).index
    df.loc[rows, "orders"] = [100, 900]
    df.loc[rows, "value_per_order"] = [8000, 12000]
    r = run_b(write(tmp_path, df), {"roles": {"value_per_order": "revenue_per_kpi"}})
    t = r.b.table.set_index("week")
    assert float(t.loc[pd.Timestamp(date), "revenue_per_kpi"]) == pytest.approx(11600)
    assert "value_per_order" not in t.columns


# 2-3. Columns repeated on split rows -------------------------------------------------------

def test_rounding_noise_is_still_the_same_value(tmp_path):
    df = weekly()
    date = df["week"][8]
    df = split(df, 8)
    df.loc[week(df, date).index, "price_index"] = [1.023, 1.0230000001]
    r = run_b(write(tmp_path, df), defaults=False)
    row = rule(r, "price_index")
    assert row["label"] == "Keep (same on every row)" and not row["choice"]
    assert float(r.b.table.set_index("week").loc[pd.Timestamp(date), "price_index"]) == pytest.approx(1.023)
    assert sig_round(pd.Series([1.023, 1.0230000001]), 6).nunique() == 1


def test_column_identical_in_two_of_three_split_weeks_gets_a_choice(tmp_path):
    df = weekly()
    dates = [str(df["week"][i].date()) for i in (6, 14, 22)]
    for d in dates:
        df = split(df, int(np.flatnonzero(pd.to_datetime(df["week"]) == pd.Timestamp(d))[0]))
    df.loc[week(df, dates[2]).index, "price_index"] = [1.01, 1.03]
    r = run_b(write(tmp_path, df), defaults=False)
    row = rule(r, "price_index")
    assert row["choice"] and row["default"] == "mean" and row["differs"] == [dates[2]]
    assert "Add up if each row covers part" in row["hint"]
    assert not any("doubles" in w for w in s8(r).warnings)                 # no red unless Add up is picked
    r2 = run_b(write(tmp_path, df), {"aggregation_rules": {"price_index": "max"}}, defaults=False)
    assert float(r2.b.table.set_index("week").loc[pd.Timestamp(dates[2]), "price_index"]) == pytest.approx(1.03)


def test_add_up_on_an_identical_column_is_the_only_red(tmp_path):
    df = weekly()
    date = df["week"][8]
    df = split(df, 8)
    r = run_b(write(tmp_path, df), {"aggregation_rules": {"price_index": "sum"}}, defaults=False)
    row = rule(r, "price_index")
    assert row.get("doubles") and row["applied"] == "sum"                 # the rule shown is the rule applied
    assert any("doubles it" in w for w in s8(r).warnings)
    assert float(r.b.table.set_index("week").loc[pd.Timestamp(date), "price_index"]) == pytest.approx(2 * df.loc[week(df, date).index[0], "price_index"])


# 4-5. Split or revision --------------------------------------------------------------------

def test_even_split_is_added_up_not_a_revision(tmp_path):
    raw = weekly()
    date = raw["week"][15]
    r = run_b(write(tmp_path, split(raw, 15)), defaults=False)
    g = s8(r).details["groups"]
    assert [x["kind"] for x in g] == ["split"] and not any(d.id == "multi_row_weeks" for d in s8(r).decisions)
    t = r.b.table.set_index("week")
    assert float(t.loc[pd.Timestamp(date), "orders"]) == pytest.approx(raw.loc[15, "orders"])
    assert float(t.loc[pd.Timestamp(date), "radio_spend"]) == pytest.approx(raw.loc[15, "radio_spend"])
    assert len(t) == WEEKS


def test_revision_is_asked_and_keeps_the_later_row_by_default(tmp_path):
    raw = weekly()
    date = raw["week"][15]
    later = raw.iloc[[15]].copy()
    later[["orders", "radio_spend"]] = later[["orders", "radio_spend"]] * 1.05    # a corrected full week
    df = pd.concat([raw, later], ignore_index=True)
    r = run_b(write(tmp_path, df), defaults=False)
    d = next(d for d in s8(r).decisions if d.id == "multi_row_weeks")
    assert r.status == "needs_input" and d.pending and d.context["weeks"][0]["kind"] == "revision"
    assert list(d.default.values()) == ["later"]
    shown = d.context["weeks"][0]
    assert len(shown["rows"]) == 2 and shown["normal"]["orders"] > 0          # both rows and the normal level
    r = run_b(write(tmp_path, df), defaults=True)
    t = r.b.table.set_index("week")
    assert float(t.loc[pd.Timestamp(date), "orders"]) == pytest.approx(later["orders"].iloc[0])
    assert any("likely revision, keep the later row" in line for e in r.b.decision_log for line in e.get("lines", []))
    r = run_b(write(tmp_path, df), {"multi_row_weeks": "sum"}, defaults=True)    # added up only when asked
    assert float(r.b.table.set_index("week").loc[pd.Timestamp(date), "orders"]) == pytest.approx(raw.loc[15, "orders"] + later["orders"].iloc[0])


def test_kpi_split_with_spend_repeated_on_each_row_is_asked(tmp_path):
    """Sales rows from two sources, each carrying the full week's spend: adding up would double spend."""
    raw = weekly()
    date = raw["week"][15]
    a, b = raw.iloc[[15]].copy(), raw.iloc[[15]].copy()
    a["orders"], b["orders"] = raw.loc[15, "orders"] * 0.3, raw.loc[15, "orders"] * 0.7
    df = pd.concat([raw.drop(index=15), a, b], ignore_index=True)
    r = run_b(write(tmp_path, df), defaults=False)
    g = s8(r).details["groups"][0]
    assert g["kind"] == "unclear" and r.status == "needs_input"


def test_burst_week_split_is_still_a_split(tmp_path):
    """A split week that also has a spend burst: the KPI decides, the spend total is not 'repeated'."""
    raw = weekly()
    raw.loc[15, "radio_spend"] *= 4
    r = run_b(write(tmp_path, split(raw, 15, (0.4, 0.6))), defaults=False)
    assert [x["kind"] for x in s8(r).details["groups"]] == ["split"]


# 6-7. Regions ---------------------------------------------------------------------------------

def test_one_row_per_region_is_never_a_split(tmp_path):
    df = weekly(weeks=10, regions=["A", "B", "C"])
    r = run_b(write(tmp_path, df), {"geo_mode": "regional"})
    d = s8(r).details
    assert d["groups"] == [] and d["exact_copies"] == 0 and d["rows_after"] == 30


def test_split_inside_one_region_touches_only_that_region(tmp_path):
    raw = weekly(weeks=20, regions=["A", "B", "C"])
    i = int(np.flatnonzero((raw["region"] == "B") & (raw["week"] == raw["week"][9]))[0])
    date = raw.loc[i, "week"]
    r = run_b(write(tmp_path, split(raw, i)), {"geo_mode": "regional"}, defaults=False)
    g = s8(r).details["groups"]
    assert [(x["date"], x["geo"], x["kind"]) for x in g] == [(str(date.date()), "B", "split")]
    t = r.b.table.set_index(["week", "region"])
    assert float(t.loc[(date, "B"), "orders"]) == pytest.approx(raw.loc[i, "orders"])
    for reg in ("A", "C"):
        j = int(np.flatnonzero((raw["region"] == reg) & (raw["week"] == date))[0])
        assert float(t.loc[(date, reg), "orders"]) == raw.loc[j, "orders"]


# 8. Reach and frequency -------------------------------------------------------------------------

def test_reach_is_a_choice_and_frequency_is_weighted_by_impressions(tmp_path):
    df = weekly(rf=True)
    date = df["week"][10]
    df = split(df, 10)
    rows = week(df, date).index
    df.loc[rows, "video_reach"] = [60000, 50000]
    df.loc[rows, "video_frequency"] = [2.0, 4.0]
    df.loc[rows, "video_impressions"] = [100000, 300000]
    r = run_b(write(tmp_path, df), defaults=False)
    reach, freq = rule(r, "video_reach"), rule(r, "video_frequency")
    assert reach["choice"] and reach["default"] == "max" and "between the larger row and the sum" in reach["hint"]
    assert not freq["choice"] and "impressions" in freq["label"]
    t = r.b.table.set_index("week")
    print(f"\nreach {t.loc[pd.Timestamp(date), 'video_reach']:,.0f} (expected 60,000); "
          f"frequency {t.loc[pd.Timestamp(date), 'video_frequency']:.2f} (expected 3.50)")
    assert float(t.loc[pd.Timestamp(date), "video_reach"]) == 60000
    assert float(t.loc[pd.Timestamp(date), "video_frequency"]) == pytest.approx(3.5)
    assert "video_impressions" not in t.columns                     # kept only to weight frequency


# 9-11. Revenue per KPI ----------------------------------------------------------------------------

def test_zero_kpi_week_uses_the_overall_value(tmp_path):
    df = weekly(revenue=True)
    date = df["week"][7]
    df.loc[7, ["orders", "sales_value"]] = [0, 0]
    key = f"orders|{date.date()}"
    r = run_b(write(tmp_path, df), {"roles": {"sales_value": "revenue_for_kpi"}, "kpi_zero_weeks": {key: "keep"}})
    t = r.b.table.set_index("week")
    clean = r.b.table
    overall = (df["sales_value"].sum() / df["orders"].sum())
    got = float(t.loc[pd.Timestamp(date), "revenue_per_kpi"])
    s13 = r.b.steps[-1]
    print(f"\nzero-KPI week {date.date()}: ₹{got:,.1f}; whole file ₹{overall:,.1f}")
    assert got == pytest.approx(overall, rel=1e-9)
    assert any(str(date.date()) in w and "is 0" in w for w in s13.warnings)
    assert s13.details["revenue_per_kpi"]["zero_kpi_weeks"] == [str(date.date())]
    assert np.isfinite(clean["revenue_per_kpi"]).all()


def test_interpolated_zero_kpi_week_takes_its_revenue_along(tmp_path):
    df = weekly(revenue=True)
    date = df["week"][7]
    df.loc[7, ["orders", "sales_value"]] = [0, 0]
    r = run_b(write(tmp_path, df), {"roles": {"sales_value": "revenue_for_kpi"}})     # step 12 default: interpolate
    v = float(r.b.table.set_index("week").loc[pd.Timestamp(date), "revenue_per_kpi"])
    assert 0.9 * 10000 < v < 1.1 * 10000                                          # not ₹0


def test_revenue_kpi_has_no_revenue_per_kpi(tmp_path):
    df = weekly(revenue=True).drop(columns="orders")
    r = run_b(write(tmp_path, df), {"roles": {"sales_value": "kpi"}, "kpi_type": "revenue"})
    assert r.b.kpi["kpi_type"] == "revenue"
    assert not any(c.startswith("revenue_per_kpi") for c in r.b.table.columns)
    assert "revenue_per_kpi" not in r.b.coord_to_columns()["coord_to_columns"]


def test_fixed_value_per_unit_every_week(tmp_path):
    r = run_b(write(tmp_path, weekly()), {"revenue_per_kpi_value": 5000})
    t = r.b.table
    assert (t["revenue_per_kpi"] == 5000).all() and len(t) == WEEKS


def test_count_kpi_never_passes_the_revenue_total(tmp_path):
    r = run_b(write(tmp_path, weekly(revenue=True)), {"roles": {"sales_value": "revenue_for_kpi"}})
    assert "sales_value" not in r.b.table.columns
    assert "sales_value" not in str(r.b.coord_to_columns())


# Exact copies ----------------------------------------------------------------------------------------

def test_exact_copies_are_dropped_never_added(tmp_path):
    raw = weekly()
    df = pd.concat([raw, raw.iloc[[3, 9]]], ignore_index=True)
    df.loc[len(raw), "price_index"] += 1e-9                         # rounding noise is still a copy
    r = run_b(write(tmp_path, df), defaults=False)
    d = s8(r).details
    assert d["exact_copies"] == 2 and d["groups"] == []
    assert float(r.b.table["orders"].sum()) == pytest.approx(raw["orders"].sum())


# INS_FULL ------------------------------------------------------------------------------------------

INS_FULL = next((p for p in Path(__file__).parent.joinpath("data").rglob("*") if p.stem.lower() == "ins_full"), None)
needs_ins_full = pytest.mark.skipif(INS_FULL is None, reason="put INS_FULL (ins_full.csv or .xlsx) in tests/data")
INS_SPLITS = ["2023-06-26", "2024-05-06", "2025-07-07"]


@pytest.fixture(scope="module")
def ins():
    return run_pipeline(INS_FULL, {"duplicate_dates": "merge"}, accept_defaults=True)


@needs_ins_full
def test_ins_full_step8_rows_and_splits(ins):
    d = s8(ins).details
    print(f"\nrows {d['rows_before']} → {d['rows_after']}; exact copies {d['exact_copies']}; "
          f"multi-row weeks {[(g['date'], g['kind']) for g in d['groups']]}")
    assert (d["rows_before"], d["rows_after"], d["exact_copies"]) == (161, 154, 4)
    assert [(g["date"], g["kind"]) for g in d["groups"]] == [(w, "split") for w in INS_SPLITS]


@needs_ins_full
@pytest.mark.parametrize("col, expected", [("Avg Premium Idx", [1.023, 1.025, 1.099]),
                                           ("Vahan Registrations", [374300, 371800, 463100]),
                                           ("Brand Search", [63.9, 59.7, 60.1])])
def test_ins_full_repeated_columns_are_kept(ins, col, expected):
    rows = {r["column"]: r for r in s8(ins).details["rules"]}
    name = next(c for c in rows if c.lower().startswith(col.lower()))
    assert rows[name]["label"] == "Keep (same on every row)"
    t = ins.b.table.set_index(ins.b.table.columns[0])
    got = [float(t.loc[pd.Timestamp(w), name]) for w in INS_SPLITS]
    print(f"\n{name}: expected {expected}, got {got}")
    assert got == pytest.approx(expected, rel=1e-6)


@needs_ins_full
def test_ins_full_revenue_per_kpi(ins):
    t = ins.b.table.set_index(ins.b.table.columns[0])
    got = [round(float(t.loc[pd.Timestamp(w), "revenue_per_kpi"]), 1) for w in INS_SPLITS]
    med = float(t["revenue_per_kpi"].median())
    print(f"\nrevenue per KPI in split weeks: expected [10062.2, 10139.7, 10812.4], got {got}; "
          f"median over {len(t)} weeks: expected 10,324, got {med:,.1f}")
    assert got == pytest.approx([10062.2, 10139.7, 10812.4], abs=0.05)
    assert len(t) == 156 and round(med) == 10324
