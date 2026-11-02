"""Future planning (G and H in Future mode) and Tracking.

The period rules, forecasts, plans store and tracking comparisons need no Meridian. build_future_data
is checked on a small hand-made two-region model (no fit). The Meridian calculations run on a real
small fit with MMM_FIT_TESTS=1.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
import pytest
from helpers import project_in, run_path

from mmm_tool.future import forecast, period, store, tracking

WEEKS = [str(d.date()) for d in pd.date_range("2023-01-02", periods=156, freq="7D")]
fit_tests = pytest.mark.skipif(os.environ.get("MMM_FIT_TESTS") != "1", reason="set MMM_FIT_TESTS=1 to run Meridian fits")


# Period ---------------------------------------------------------------------------------------------------

def test_defaults_and_grid():
    d = period.defaults(WEEKS)
    assert d["start"] == "2025-12-29" and d["weeks"] == 13 and d["template"] == {"kind": "last_year"}
    plan, warn = period.plan_weeks("2025-12-29", 13, WEEKS)
    assert plan[0] == "2025-12-29" and plan[-1] == "2026-03-23" and not warn
    with pytest.raises(period.PeriodError, match="weekly grid"):
        period.plan_weeks("2025-12-30", 13, WEEKS)
    with pytest.raises(period.PeriodError, match="after the last data week"):
        period.plan_weeks("2025-12-22", 13, WEEKS)
    _, warn = period.plan_weeks("2026-07-06", 13, WEEKS)               # 28 weeks after the data
    assert warn and "out of date" in warn[0]


def test_templates_and_lead_in():
    plan, _ = period.plan_weeks("2025-12-29", 13, WEEKS)
    ly = period.template_weeks("last_year", plan, WEEKS)
    assert ly[0] == "2024-12-30" and len(ly) == 13 and all(w in WEEKS for w in ly)        # same week of the year
    assert period.template_weeks("recent", plan, WEEKS) == WEEKS[-13:]
    assert period.template_weeks("custom", plan, WEEKS, "2024-03-04")[0] == "2024-03-04"
    with pytest.raises(period.PeriodError):
        period.template_weeks("custom", plan, WEEKS, "2025-12-15")                       # runs past the data
    dates, src = period.lead_in(plan, ly, WEEKS, 8)
    assert len(dates) == 8 and dates[-1] == "2025-12-22" and src == WEEKS[-8:]           # right after: the real weeks
    later, _ = period.plan_weeks("2026-03-02", 13, WEEKS)
    tl = period.template_weeks("last_year", later, WEEKS)
    _, src2 = period.lead_in(later, tl, WEEKS, 8)
    i = WEEKS.index(tl[0])
    assert src2 == WEEKS[i - 8:i]                                                        # later: the template's preceding weeks
    far, _ = period.plan_weeks("2027-01-04", 13, WEEKS)
    with pytest.raises(period.PeriodError, match="same weeks last year"):
        period.template_weeks("last_year", far, WEEKS)


# Forecast -------------------------------------------------------------------------------------------------

KPI = {w: 1000.0 + 10 * i for i, w in enumerate(WEEKS)}


def test_builtin_forecast_is_last_year_times_growth_with_step31_bounds():
    plan, _ = period.plan_weeks("2025-12-29", 13, WEEKS)
    tmpl = period.template_weeks("last_year", plan, WEEKS)
    f = forecast.builtin(plan, tmpl, WEEKS, KPI, None, {"naive_mape": 0.08})
    g = forecast.default_growth(WEEKS, KPI)
    assert g == pytest.approx(sum(KPI[w] for w in WEEKS[-52:]) / sum(KPI[w] for w in WEEKS[-104:-52]) - 1)
    assert f["mean"][0] == pytest.approx(KPI[tmpl[0]] * (1 + g))
    assert f["lower"][0] == pytest.approx(f["mean"][0] * 0.92) and "step 31" in f["note"]
    f2 = forecast.builtin(plan, tmpl, WEEKS, KPI, 0.05, {"accuracy": {"Test": {"MAPE": 0.03}}})
    assert f2["mean"][0] == pytest.approx(KPI[tmpl[0]] * 1.05) and f2["upper"][0] == pytest.approx(f2["mean"][0] * 1.03)


def test_csv_total_and_none():
    plan, _ = period.plan_weeks("2025-12-29", 4, WEEKS)
    text = "week,forecast,lower,upper\n" + "\n".join(f"{w},{100 + i},{90 + i},{110 + i}" for i, w in enumerate(plan))
    f = forecast.from_csv(text, plan)
    assert f["mean"] == [100, 101, 102, 103] and f["lower"][0] == 90 and f["total"] == 406
    with pytest.raises(forecast.ForecastError, match="no row"):
        forecast.from_csv(text.rsplit("\n", 1)[0], plan)
    t = forecast.from_total(1000, plan, WEEKS[-4:], KPI)
    assert sum(t["mean"]) == pytest.approx(1000) and t["mean"][3] > t["mean"][0]       # spread like the template's KPI
    assert forecast.none(plan)["mean"] is None
    assert f["hash"] != forecast.from_csv(text.replace(",100,", ",99,"), plan)["hash"]



def test_lopsided_forecast_bounds_give_a_lopsided_range():
    from mmm_tool.future.engine import forecast_draws

    f = {"mean": [100.0, 200.0], "lower": [90.0, 180.0], "upper": [130.0, 260.0]}      # -10% / +30%
    d = forecast_draws(20000, f)
    lo, hi = np.quantile(d, [0.05, 0.95], axis=0)
    assert lo == pytest.approx([90.0, 180.0], rel=0.01) and hi == pytest.approx([130.0, 260.0], rel=0.01)
    even = forecast_draws(500, {"mean": [100.0], "lower": [90.0], "upper": [110.0]})
    assert even.mean() == pytest.approx(100.0, rel=1e-12)                               # even bounds: centred exactly
    assert np.allclose(forecast_draws(3, {"mean": [5.0, 6.0], "lower": None, "upper": None}), [[5.0, 6.0]] * 3)
    plan = ["2026-01-05", "2026-01-12"]
    bad = "date,forecast,lower,upper\n2026-01-05,100,110,120\n2026-01-12,100,90,110\n"
    with pytest.raises(forecast.ForecastError, match="lower bound must be at or below"):
        forecast.from_csv(bad, plan)

# Plans store ------------------------------------------------------------------------------------------------

def _eval(weeks, spend_per_week, media, total=None):
    """A minimal engine.evaluate-shaped result."""
    n = len(weeks)
    band = lambda v: {"mean": [v] * n, "lo": [v * 0.9] * n, "hi": [v * 1.1] * n}  # noqa: E731
    return {"weeks": weeks, "channels": list(spend_per_week), "spend": {c: [v] * n for c, v in spend_per_week.items()},
            "multipliers": {c: 1.0 for c in spend_per_week},
            "per_week": {"reference": [media] * n, "media": band(media), "change": band(0.0), "total": band(total) if total else None,
                         "forecast": [total - media + media] * n if total else None, "forecast_lower": None, "forecast_upper": None},
            "totals": {"reference": media * n, "media": {"mean": media * n, "lo": 0, "hi": 0}, "change": {"mean": 0.0, "lo": 0, "hi": 0},
                       "total": {"mean": total * n, "lo": 0, "hi": 0} if total else None, "spend": sum(spend_per_week.values()) * n,
                       "template_spend": sum(spend_per_week.values()) * n, "share_better": 0.0},
            "by_channel": {c: {"media": media / len(spend_per_week) * n, "reference": 0} for c in spend_per_week}}


def _setup(pdir, plan):
    return store.save_period(pdir, {"plan": plan, "template": plan, "forecast": {"hash": "f1"}, "run_id": "r", "model_key": "m"})


def test_plans_and_choosing_one_for_tracking(tmp_path):
    plan, _ = period.plan_weeks("2025-12-29", 4, WEEKS)
    s = _setup(tmp_path, plan)
    a = store.add(tmp_path, store.record(_eval(plan, {"tv": 10, "search": 5}, 50, 400), "Same as last year", {"kind": "reference"}, s, "r", "m"))
    b = store.add(tmp_path, store.record(_eval(plan, {"tv": 12, "search": 5}, 55, 405), "TV up", {"kind": "what-if"}, s, "r", "m"))
    assert (a["id"], b["id"]) == ("A", "B")
    r = store.choose(tmp_path, s["id"], "A")
    assert r["from_week"] == plan[0] and store.tracked(tmp_path, s["id"])["id"] == "A"
    t = store.tracking(tmp_path)
    assert set(t["targets"]) == set(plan) and t["targets"][plan[0]]["total"] == 400
    # Two weeks of actuals arrive, then tracking switches to B: those weeks keep A's targets.
    t["actual_weeks"] = plan[:2]
    store.save_tracking(tmp_path, t)
    with pytest.raises(store.PlanStoreError, match="Say why"):
        store.choose(tmp_path, s["id"], "B")
    r = store.choose(tmp_path, s["id"], "B", "Budget raised at the review")
    assert r["from_week"] == plan[2] and r["switched_from"] == "A"
    t = store.tracking(tmp_path)
    assert [t["targets"][w]["plan_id"] for w in plan] == ["A", "A", "B", "B"]
    assert [p["status"] for p in store.plans(tmp_path, s["id"])] == ["draft", "tracked"]          # only one tracked
    log = [e["what"] for e in store.entries(tmp_path)]
    assert log.count("plan chosen for tracking") == 1 and log.count("tracking switched") == 1
    sw = next(e for e in store.entries(tmp_path) if e["what"] == "tracking switched")
    assert sw["details"]["reason"] == "Budget raised at the review" and sw["details"]["model"] == "m" and sw["by"]
    with pytest.raises(store.PlanStoreError, match="can't be archived"):
        store.update(tmp_path, s["id"], "B", status="archived")
    c = store.duplicate(tmp_path, s["id"], "B")
    assert c["id"] == "C" and c["status"] == "draft" and c["name"] == "Copy of TV up"
    assert store.stale(c, s, "m") == [] and store.stale(c, s, "m2") == ["model"]
    assert store.stale(c, {**s, "forecast": {"hash": "f2"}}, "m") == ["forecast"]



def test_setting_up_the_same_dates_again_keeps_plans_and_flags_what_changed(tmp_path):
    plan, _ = period.plan_weeks("2025-12-29", 4, WEEKS)
    R = {"plan": plan, "template": plan, "lead_dates": [], "lead_src": []}
    s1 = store.save_period(tmp_path, {"plan": plan, "template": plan, "resolved": R, "cost": {}, "forecast": {"hash": "f1"}, "run_id": "r"})
    a = store.add(tmp_path, store.record(_eval(plan, {"tv": 10}, 50, 400), "Same as last year", {"kind": "reference"}, s1, "r", "m"))
    store.choose(tmp_path, s1["id"], "A")
    s2 = store.save_period(tmp_path, {"plan": plan, "template": plan, "resolved": R, "cost": {"tv": 0.2}, "forecast": {"hash": "f1"}, "run_id": "r"})
    assert s2["id"] == s1["id"] and store.tracked(tmp_path, s2["id"])["id"] == "A"          # same period, still tracked
    assert store.stale(a, s2, "m") == ["setup"]                                             # the cost change differs
    assert store.snapshot(s2, a)["cost"] == {} and store.snapshot(s2, None)["cost"] == {"tv": 0.2}   # tracked on its own setup
    assert store.reference(tmp_path, s2)["id"] == "A"                                       # the only plan A so far
    a2 = store.add(tmp_path, store.record(_eval(plan, {"tv": 10}, 48, 398), "Same as last year", {"kind": "reference"}, s2, "r", "m"))
    assert store.reference(tmp_path, s2)["id"] == a2["id"] == "B"


def test_tracking_starts_clean_for_a_new_period_and_needs_a_week_left(tmp_path):
    plan, _ = period.plan_weeks("2025-12-29", 4, WEEKS)
    s = _setup(tmp_path, plan)
    store.add(tmp_path, store.record(_eval(plan, {"tv": 10}, 50, 400), "A", {"kind": "reference"}, s, "r", "m"))
    store.add(tmp_path, store.record(_eval(plan, {"tv": 11}, 51, 401), "B", {"kind": "what-if"}, s, "r", "m"))
    t = store.tracking(tmp_path, s["id"])
    t["actual_weeks"], t["renames"] = list(plan), {"tv_inr": "tv_spend"}
    store.save_tracking(tmp_path, t)
    with pytest.raises(store.PlanStoreError, match="no week left"):
        store.choose(tmp_path, s["id"], "A")                                     # every week already has actuals
    later, _ = period.plan_weeks("2026-03-30", 4, WEEKS)
    s2 = _setup(tmp_path, later)
    store.add(tmp_path, store.record(_eval(later, {"tv": 10}, 50, 400), "A", {"kind": "reference"}, s2, "r", "m"))
    r = store.choose(tmp_path, s2["id"], "A")
    t = store.tracking(tmp_path, s2["id"])
    assert r["from_week"] == later[0] and set(t["targets"]) == set(later) and t["actual_weeks"] == []
    assert t["earlier"][0]["period_id"] == s["id"] and t["earlier"][0]["actual_weeks"] == plan
    assert t["renames"] == {"tv_inr": "tv_spend"}                                # kept with the project


def test_actuals_uploaded_a_few_weeks_at_a_time_add_up():
    plan = ["2026-01-05", "2026-01-12", "2026-01-19"]
    first = {"weeks": plan[:2], "kpi": [10.0, 11.0], "spend": {"tv": [1.0, 2.0]}, "units": {"tv": [5.0, 6.0]}}
    second = {"weeks": plan[1:], "kpi": [12.0, 13.0], "spend": {"tv": [3.0, 4.0]}, "units": {}}
    m = tracking.merge(first, second, plan)
    assert m["weeks"] == plan and m["kpi"] == [10.0, 12.0, 13.0] and m["spend"]["tv"] == [1.0, 3.0, 4.0]
    assert m["units"]["tv"] == [5.0, None, None]                                # a later upload without units: the plan's
    assert tracking.merge(None, second, plan) is second

# Tracking ---------------------------------------------------------------------------------------------------

def test_a_renamed_column_is_offered_by_similarity_and_remembered():
    train = ["week", "tv_spend", "youtube_spend", "orders"]
    m = tracking.match_columns(["week", "tv_spend", "youtube_spend_inr", "orders", "whatsapp_spend"], train, {}, 0.75)
    assert m["offered"] == {"youtube_spend_inr": {"column": "youtube_spend", "similarity": m["offered"]["youtube_spend_inr"]["similarity"]}}
    assert m["new"] == ["whatsapp_spend"] and m["missing"] == []
    m2 = tracking.match_columns(["week", "tv_spend", "youtube_spend_inr", "orders"], train, {"youtube_spend_inr": "youtube_spend"}, 0.75)
    assert m2["offered"] == {} and m2["confirmed"] == {"youtube_spend_inr": "youtube_spend"}     # asked once


SET = {"spend_tolerance_pct": 10, "spend_off_weeks": 2, "outside_range_weeks": 3, "cumulative_gap_pct": 5, "execution_share": 0.5, "similarity": 0.75}


def _targets(plan, spend, total):
    return {w: {"week": w, "spend": dict(spend), "total": total, "plan_id": "A"} for w in plan}


def _at_actual(plan, totals, media_by_channel):
    n = len(plan)
    return {"weeks": plan, "per_week": {"total": {"mean": totals, "lo": [t * 0.97 for t in totals], "hi": [t * 1.03 for t in totals]},
                                        "forecast_lower": [0] * n},
            "by_channel": {c: {"media": v} for c, v in media_by_channel.items()}}


def test_actuals_exactly_on_plan_have_no_execution_gap():
    plan, _ = period.plan_weeks("2025-12-29", 6, WEEKS)
    tg = _targets(plan, {"tv": 100, "search": 50}, 1000.0)
    on = {"weeks": plan, "kpi": [1000.0] * 6, "spend": {"tv": [100] * 6, "search": [50] * 6}, "units": {}}
    r = tracking.compare(on, tg, _at_actual(plan, [1000.0] * 6, {"tv": 300, "search": 200}), SET, None, {"tv": {"media": 300}, "search": {"media": 200}})
    assert all(row["execution_gap"] == 0 and row["status"] == "inside" for row in r["rows"])
    assert r["alerts"] == [] and r["tiles"]["inside"] == 6 and r["tiles"]["channels_within"]


def test_a_channel_at_zero_spend_is_named_in_the_execution_alert():
    plan, _ = period.plan_weeks("2025-12-29", 6, WEEKS)
    tg = _targets(plan, {"tv": 100, "search": 50}, 1000.0)
    on = {"weeks": plan, "kpi": [940.0] * 6, "spend": {"tv": [0] * 6, "search": [50] * 6}, "units": {}}
    at = _at_actual(plan, [950.0] * 6, {"tv": 0, "search": 200})          # no TV: the model predicts 50 less a week
    r = tracking.compare(on, tg, at, SET, None, {"tv": {"media": 300}, "search": {"media": 200}})
    assert all(row["execution_gap"] < 0 for row in r["rows"])
    ex = next(a for a in r["alerts"] if a["kind"] == "execution")
    assert [d["channel"] for d in ex["channels"]] == ["tv"] and ex["channels"][0]["effect"] < 0
    sp = next(a for a in r["alerts"] if a["kind"] == "spend")
    assert sp["channels"] == ["tv"]


def test_kpi_below_range_with_spend_on_plan_says_check_outside_causes():
    plan, _ = period.plan_weeks("2025-12-29", 8, WEEKS)
    tg = _targets(plan, {"tv": 100, "search": 50}, 1000.0)
    kpi = [1000.0] * 4 + [920.0] * 4                                      # −8% for the last 4 weeks
    on = {"weeks": plan, "kpi": kpi, "spend": {"tv": [100] * 8, "search": [50] * 8}, "units": {}}
    r = tracking.compare(on, tg, _at_actual(plan, [1000.0] * 8, {"tv": 300, "search": 200}), SET, None, {"tv": {"media": 300}, "search": {"media": 200}})
    red = next(a for a in r["alerts"] if a["level"] == "red")
    assert red["side"] == "below" and red["streak"] == 4
    assert red["text"] == "Spend matched the plan, so the gap is not explained by media. Check outside causes first."
    assert [row["status"] for row in r["rows"]][-4:] == ["below"] * 4 and not any(a["kind"] == "execution" for a in r["alerts"])


def test_without_forecast_bounds_the_range_is_the_holdout_error():
    plan, _ = period.plan_weeks("2025-12-29", 2, WEEKS)
    at = _at_actual(plan, [1000.0, 1000.0], {"tv": 1})
    at["per_week"]["forecast_lower"] = None
    r = tracking.compare({"weeks": plan, "kpi": [1000.0, 1000.0], "spend": {"tv": [1, 1]}, "units": {}}, _targets(plan, {"tv": 1}, 1000.0), at, SET, 0.06)
    assert r["rows"][0]["lo"] == pytest.approx(940) and "holdout" in r["bounds_from"]


# build_future_data on a small hand-made two-region model (no fit) ---------------------------------------------------

class _Model:
    """Just the input data build_future_data reads: two regions, a channel with media units, one spend-only."""

    def __init__(self):
        import xarray as xr

        n, geos, ch = len(WEEKS[:60]), ["north", "south"], ["tv", "search"]
        rng = np.random.default_rng(1)
        spend = rng.uniform(50, 150, (2, n, 2))
        spend[1] *= 3                                                     # the south spends three times the north
        media = spend.copy()
        media[..., 0] = spend[..., 0] * 2.0                               # tv: 2 units per unit of spend
        dims, coords = ("geo", "time", "media_channel"), {"geo": geos, "time": WEEKS[:60], "media_channel": ch}
        self.input_data = type("I", (), {})()
        d = self.input_data
        d.media = xr.DataArray(media, dims=("geo", "media_time", "media_channel"), coords={"geo": geos, "media_time": WEEKS[:60], "media_channel": ch})
        d.media_spend = xr.DataArray(spend, dims=dims, coords=coords)
        d.controls = xr.DataArray(rng.normal(size=(2, n, 1)), dims=("geo", "time", "control_variable"))
        d.revenue_per_kpi = xr.DataArray(np.full((2, n), 40.0), dims=("geo", "time"))
        for k in ("reach", "frequency", "rf_spend", "organic_media", "organic_reach", "organic_frequency", "non_media_treatments"):
            setattr(d, k, None)
        d.time = xr.DataArray(WEEKS[:60], dims="time")
        d.geo = xr.DataArray(geos, dims="geo")


def _resolved(lag=4, n=8):
    ws = WEEKS[:60]
    plan, _ = period.plan_weeks(str((pd.Timestamp(ws[-1]) + pd.Timedelta(days=7)).date()), n, ws)
    return period.resolve({"start": plan[0], "weeks": n, "template": {"kind": "recent"}}, ws, lag)


def test_build_future_data_shapes_scaling_and_cost():
    pytest.importorskip("meridian")
    from mmm_tool.future.data import FutureDataError, build_future_data, spend_only

    mmm, R = _Model(), _resolved()
    assert spend_only(mmm) == {"tv": False, "search": True}
    base, info = build_future_data(mmm, R)
    nd, _ = build_future_data(mmm, R, mult={"tv": 1.5}, cost={"tv": 0.25, "search": 0.5})
    assert np.asarray(base.media).shape == (2, 4 + 8, 2) and list(base.time)[:4] == R["lead_dates"]   # lead-in = max_lag
    b_s, n_s = np.asarray(base.media_spend), np.asarray(nd.media_spend)
    b_m, n_m = np.asarray(base.media), np.asarray(nd.media)
    assert np.allclose(n_s[:, 4:, 0], b_s[:, 4:, 0] * 1.5) and np.allclose(n_s[:, :4], b_s[:, :4])  # plan weeks only
    assert np.allclose(n_s[:, :, 1], b_s[:, :, 1])                                                  # the other channel
    assert np.allclose(n_m[:, 4:, 0], b_m[:, 4:, 0] * 1.5 / 1.25)                                   # units: ×1.5 ÷ (1 + 25%)
    assert np.allclose(n_m[:, :, 1], b_m[:, :, 1])                                                  # spend-only: no cost effect
    c_only, _ = build_future_data(mmm, R, cost={"tv": 0.25})
    assert np.allclose(np.asarray(c_only.media_spend), b_s) and np.allclose(np.asarray(c_only.media)[:, 4:, 0], b_m[:, 4:, 0] / 1.25)
    # Upload: a planned amount per week, split across regions like the template
    want = {"tv": [1000.0] * 8, "search": [400.0] * 8}
    up, info = build_future_data(mmm, R, spend=want)
    u = np.asarray(up.media_spend)
    share_t = b_s[:, 4:, 0] / b_s[:, 4:, 0].sum(axis=0)
    assert np.allclose(u[:, 4:, 0].sum(axis=0), 1000) and np.allclose(u[:, 4:, 0] / 1000, share_t)  # regional shares kept
    assert info["spend"]["tv"] == pytest.approx([1000.0] * 8)
    with pytest.raises(FutureDataError, match="8 weekly amounts"):
        build_future_data(mmm, R, spend={"tv": [1.0] * 7})
    with pytest.raises(FutureDataError, match="Not channels"):
        build_future_data(mmm, R, spend={"radio": [1.0] * 8})
    with pytest.raises(FutureDataError, match="is negative"):
        build_future_data(mmm, R, spend={"tv": [-1.0] + [1.0] * 7})
    # Actual media units (Tracking): known weeks replace the plan's, unknown weeks (None) keep them
    au, _ = build_future_data(mmm, R, cost={"tv": 0.25}, units={"tv": [None] * 7 + [500.0]})
    a_m = np.asarray(au.media)
    assert np.allclose(a_m[:, 4:11, 0], b_m[:, 4:11, 0] / 1.25) and a_m[:, 11, 0].sum() == pytest.approx(500.0)
    r2, _ = build_future_data(mmm, R, rpk={"south": 55.0})
    assert np.allclose(np.asarray(r2.revenue_per_kpi)[1, 4:], 55.0) and np.allclose(np.asarray(r2.revenue_per_kpi)[0, 4:], 40.0)


# API: Past mode untouched; a renamed column asked once (no Meridian) ------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from mmm_tool.api import app as app_mod

    monkeypatch.setattr(app_mod, "RUNS_DIR", tmp_path / "runs")
    c = TestClient(app_mod.app)
    c.runs = tmp_path / "runs"
    return c


def _accepted(client):
    from test_phase_i import fake_fit

    rid = client.post("/api/runs/sample/ins_clean", params={"project": project_in()}).json()["run_id"]
    client.put(f"/api/runs/{rid}/answers", json={"answers": {"roles": {}}})
    fake_fit(run_path(rid))
    return rid


def test_past_mode_is_unchanged(client):
    rid = _accepted(client)
    p = client.get(f"/api/runs/{rid}/plan").json()
    assert p["mode"] == "past" and set(p["periods"]) == {"all", "last_52"} and p["defaults"]["period"] == "all"
    chk = client.post(f"/api/runs/{rid}/plan/check", json={"plan": p["defaults"]}).json()
    assert chk["ok"] and chk["args"]["start_date"] == p["periods"]["all"][0]
    f = client.get(f"/api/runs/{rid}/future").json()
    assert f["period"] is None and f["defaults"]["weeks"] == 13 and f["lag"] >= 1
    assert client.get(f"/api/runs/{rid}/plan?mode=future").status_code == 409            # Future needs its period first


def test_setup_checks_dates(client):
    rid = _accepted(client)
    f = client.get(f"/api/runs/{rid}/future").json()
    bad = {**f["defaults"], "start": str((pd.Timestamp(f["defaults"]["start"]) + pd.Timedelta(days=1)).date())}
    r = client.post(f"/api/runs/{rid}/future/setup", json=bad)
    assert r.status_code == 400 and "weekly grid" in r.json()["detail"]
    r = client.post(f"/api/runs/{rid}/future/setup", json={**f["defaults"], "forecast": {"source": "total"}})
    assert r.status_code == 400 and "total" in r.json()["detail"]


def test_tracking_upload_asks_about_a_renamed_column_once(client):
    rid = _accepted(client)
    d = run_path(rid)
    from mmm_tool import projects

    pdir = projects.folder(client.runs, projects.project_of(d))
    raw = pd.read_csv(d / "ins_clean.csv")
    weeks = sorted(pd.to_datetime(raw.iloc[:, 0]).dt.strftime("%Y-%m-%d"))
    plan, _ = period.plan_weeks(str((pd.Timestamp(weeks[-1]) + pd.Timedelta(days=7)).date()), 4, weeks)
    s = store.save_period(pdir, {"plan": plan, "template": weeks[-4:], "forecast": {"hash": "x", "mean": [1] * 4},
                                 "run_id": rid, "model_key": "m", "resolved": {"plan": plan, "template": weeks[-4:], "lead_dates": [], "lead_src": []}})
    ev = _eval(plan, {"tv": 1.0}, 1.0, 10.0)
    store.add(pdir, store.record(ev, "A", {"kind": "reference"}, s, rid, "m"))
    store.choose(pdir, s["id"], "A")
    new = raw.tail(4).copy()
    new.iloc[:, 0] = plan
    spend_col = next(c for c in new.columns if c.endswith("_spend"))
    new = new.rename(columns={spend_col: spend_col + "_inr"})
    path = client.runs / "new.csv"
    new.to_csv(path, index=False)
    r = client.post(f"/api/runs/{rid}/tracking/upload", files={"file": ("new.csv", path.read_bytes(), "text/csv")}).json()
    assert r["needs_confirm"] and r["match"]["offered"][spend_col + "_inr"]["column"] == spend_col
    r = client.post(f"/api/runs/{rid}/tracking/confirm", json={"file": "new.csv", "renames": {spend_col + "_inr": spend_col}}).json()
    assert r["processed"] and r["weeks"] == plan
    again = client.post(f"/api/runs/{rid}/tracking/upload", files={"file": ("new2.csv", path.read_bytes(), "text/csv")}).json()
    assert again.get("processed") and not again.get("needs_confirm")                      # remembered
    path.write_text(new.tail(2).to_csv(index=False))                                          # only the latest weeks
    part = client.post(f"/api/runs/{rid}/tracking/upload", files={"file": ("new3.csv", path.read_bytes(), "text/csv")}).json()
    assert part["weeks"] == plan                                                              # the earlier weeks stay
    # A refit uses every upload: the training weeks plus all 4 new ones, renamed to the training columns
    rf = client.post(f"/api/runs/{rid}/tracking/refit")
    assert rf.status_code == 200, rf.text
    new_id = rf.json()["run_id"]
    m = json.loads((run_path(new_id) / "meta.json").read_text())
    both = pd.read_csv(run_path(new_id) / m["file"])
    assert len(both) == len(raw) + 4 and spend_col in both.columns and spend_col + "_inr" not in both.columns
    assert both[spend_col].tail(4).notna().all()
    # The bundle carries the project's future plans and tracking
    import io
    import zipfile

    z = zipfile.ZipFile(io.BytesIO(client.get(f"/api/runs/{rid}/bundle.zip").content))
    assert {"future/period.json", "future/plans.json", "future/tracking.json", "future/log.json"} <= set(z.namelist())
    assert "columns matched" in [e["what"] for e in store.entries(pdir)]


# On a real fit (MMM_FIT_TESTS=1) ----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    if os.environ.get("MMM_FIT_TESTS") != "1":
        pytest.skip("set MMM_FIT_TESTS=1 to run Meridian fits")
    from test_phase_g import TINY, shop

    from mmm_tool.fit import model as m
    from mmm_tool.pipeline import run_pipeline

    out = {}
    for rf in (False, True):
        p = tmp_path_factory.mktemp("fut") / f"shop_{rf}.csv"
        shop(rf=rf).to_csv(p, index=False)
        r = run_pipeline(p, {"revenue_per_kpi_value": 40}, accept_defaults=True)
        mmm = m.build(r.d, {**r.d.config, "holdout": None})
        mmm.sample_prior(50, seed=3)
        m.sample_posterior(mmm, TINY, 3)
        out[rf] = mmm
    return out


def _real_resolved(mmm, n=13):
    from mmm_tool.future.data import weeks

    ws = weeks(mmm)
    return period.resolve(period.defaults(ws) | {"weeks": n}, ws, mmm.model_spec.max_lag), ws


@fit_tests
def test_template_of_training_weeks_matches_past_mode(fitted):
    from mmm_tool.fit import model as m
    from mmm_tool.future import engine

    mmm = fitted[False]
    R, ws = _real_resolved(mmm)
    lag = mmm.model_spec.max_lag
    T = ws[-13:]
    Rp = {**R, "template": T, "lead_src": ws[-13 - lag:-13]}          # the lead-in taken from the real preceding weeks
    fut = engine.evaluate(mmm, Rp, None)
    past = np.asarray(m.analyzer(mmm).incremental_outcome(selected_times=T, aggregate_times=False, use_kpi=True,
                                                         include_non_paid_channels=False))
    past = past.reshape(-1, 13, past.shape[-1]).sum(axis=2).mean(axis=0)
    rel = np.abs(np.array(fut["per_week"]["media"]["mean"]) - past) / np.abs(past)
    print(f"\nfuture vs past, largest weekly difference {rel.max():.2e}")
    assert rel.max() < 0.005
    assert sum(fut["per_week"]["media"]["mean"]) == pytest.approx(fut["totals"]["media"]["mean"], rel=1e-6)


@fit_tests
@pytest.mark.parametrize("rf", [False, True])
def test_plan_a_is_the_forecast_and_more_spend_adds(fitted, rf):
    from mmm_tool.future import engine
    from mmm_tool.future.data import build_future_data, channels, weeks

    mmm = fitted[rf]
    R, ws = _real_resolved(mmm)
    kpi = dict(zip(ws, np.asarray(mmm.input_data.kpi.values).sum(axis=0)))
    f = forecast.make({"source": "builtin"}, R["plan"], R["template"], ws, kpi, {"naive_mape": 0.1})
    a = engine.evaluate(mmm, R, f)
    assert a["totals"]["change"] == {"mean": 0.0, "lo": 0.0, "hi": 0.0}
    assert a["totals"]["total"]["mean"] == pytest.approx(f["total"], rel=1e-9)
    c0 = channels(mmm)[0]
    b = engine.evaluate(mmm, R, f, mult={c0: 1.2})
    assert b["totals"]["change"]["lo"] > 0 and b["totals"]["share_better"] == 1.0
    # Money: the same as Meridian's own revenue outcome, and the setup's value per KPI unit applies
    from mmm_tool.fit import model as m

    nd, _ = build_future_data(mmm, R, mult={c0: 1.2})
    mer = np.asarray(m.analyzer(mmm).incremental_outcome(new_data=nd, selected_times=R["plan"], aggregate_times=False,
                                                        use_kpi=False, include_non_paid_channels=False))
    mer_total = float(mer.reshape(-1, len(R["plan"]), mer.shape[-1]).sum(axis=(1, 2)).mean())
    rev = b["revenue"]
    print(f"\nrevenue from media: tool {rev['totals']['media']['mean']:,.0f}, Meridian {mer_total:,.0f}")
    assert rev["totals"]["media"]["mean"] == pytest.approx(mer_total, rel=1e-6)
    assert rev["totals"]["total"]["mean"] == pytest.approx(f["total"] * 40 + rev["totals"]["change"]["mean"], rel=1e-6)
    geo = str(mmm.input_data.geo.values[0])
    b50 = engine.evaluate(mmm, R, f, mult={c0: 1.2}, rpk={geo: 50.0})
    assert b50["revenue"]["totals"]["media"]["mean"] == pytest.approx(rev["totals"]["media"]["mean"] * 50 / 40, rel=1e-6)
    assert b50["totals"]["media"]["mean"] == pytest.approx(b["totals"]["media"]["mean"], rel=1e-9)   # the KPI doesn't move
    if rf:                                                             # reach scales, frequency stays
        rfc = [str(c) for c in mmm.input_data.reach.rf_channel.values][0]
        base, _ = build_future_data(mmm, R)
        up, _ = build_future_data(mmm, R, mult={rfc: 1.5}, cost={rfc: 0.2})
        L = len(R["lead_dates"])
        assert np.allclose(np.asarray(up.reach)[:, L:, 0], np.asarray(base.reach)[:, L:, 0] * 1.5 / 1.2)
        assert np.allclose(np.asarray(up.frequency), np.asarray(base.frequency))


@fit_tests
def test_future_optimizer_keeps_budget_and_locks_and_opens_in_what_if(fitted):
    from mmm_tool.future import engine
    from mmm_tool.future.data import channels, template_spend
    from mmm_tool.plan.optimize import check, defaults

    mmm = fitted[False]
    R, ws = _real_resolved(mmm)
    chans = channels(mmm)
    hist = {c: float(sum(v)) for c, v in template_spend(mmm, R["template"]).items()}
    plan = {**defaults(chans, True), "period": "future", "objective": "kpi", "locks": [chans[-1]]}
    checked = check(plan, hist, True, {"future": (R["plan"][0], R["plan"][-1])})
    res = engine.optimize(mmm, R, None, None, None, checked)
    ev = res["evaluation"]
    print(f"\nbudget {checked['budget']:,.0f}; optimized {ev['totals']['spend']:,.0f}; multipliers {res['multipliers']}")
    assert ev["totals"]["spend"] == pytest.approx(checked["budget"], rel=1e-3)
    assert abs(sum(res["after"]["channels"][chans[-1]]["spend"] for _ in [0]) - hist[chans[-1]]) <= checked["grid_step"] / 2
    # "Open in what-if": the optimized split on the sliders gives the optimizer's own media outcome
    again = engine.evaluate(mmm, R, None, mult=res["multipliers"])
    assert again["totals"]["media"]["mean"] == pytest.approx(res["optimizer_outcome"], rel=0.01)
    # From an uploaded plan: the optimizer starts there, and the optimized plan keeps its weekly shape
    up = {c: [hist[c] / len(R["plan"]) * (0.5 + k % 2) for k in range(len(R["plan"]))] for c in chans}   # alternating weeks
    budget_up = sum(sum(v) for v in up.values())
    checked_up = check({**plan, "locks": []}, {c: sum(v) for c, v in up.items()}, True, {"future": (R["plan"][0], R["plan"][-1])})
    ru = engine.optimize(mmm, R, None, None, None, checked_up, spend=up)
    assert sum(ru["before"]["channels"][c]["spend"] for c in chans) == pytest.approx(budget_up, rel=1e-3)
    assert ru["evaluation"]["totals"]["spend"] == pytest.approx(checked_up["budget"], rel=1e-3)
    for c in chans:
        sp = ru["evaluation"]["spend"][c]
        assert sp[1] / sp[0] == pytest.approx(1.5 / 0.5, rel=1e-3)                    # the upload's week-to-week shape


@fit_tests
def test_tracking_with_actuals_on_plan_has_no_execution_gap(fitted):
    from mmm_tool.future import engine

    mmm = fitted[False]
    R, ws = _real_resolved(mmm)
    kpi = dict(zip(ws, np.asarray(mmm.input_data.kpi.values).sum(axis=0)))
    f = forecast.make({"source": "builtin"}, R["plan"], R["template"], ws, kpi, {"naive_mape": 0.1})
    a = engine.evaluate(mmm, R, f)
    targets = {w: {"week": w, "spend": {c: a["spend"][c][i] for c in a["spend"]}, "total": a["per_week"]["total"]["mean"][i], "plan_id": "A"}
               for i, w in enumerate(R["plan"])}
    at = engine.evaluate(mmm, R, f, spend={c: v for c, v in a["spend"].items()})
    on = {"weeks": R["plan"][:6], "kpi": a["per_week"]["total"]["mean"][:6], "spend": {c: v[:6] for c, v in a["spend"].items()}, "units": {}}
    r = tracking.compare(on, targets, at, SET, None, a["by_channel"])
    assert all(abs(row["execution_gap"]) < 1e-6 * max(1, abs(row["plan_predicted"])) for row in r["rows"])
