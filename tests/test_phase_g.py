"""Phase G: What-If (steps 43-47).

The API rules are tested without a model. The calculations are tested on a
real (small) Meridian fit of a synthetic file, which takes a minute or two,
so those run only with MMM_FIT_TESTS=1.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
import pytest

from mmm_tool import whatif
from mmm_tool.fit import job
from mmm_tool.whatif import scenario
from helpers import project_in, run_path  # noqa: F401

fit_tests = pytest.mark.skipif(os.environ.get("MMM_FIT_TESTS") != "1", reason="set MMM_FIT_TESTS=1 to run Meridian fits")
TINY = {"n_chains": 2, "n_adapt": 100, "n_burnin": 100, "n_keep": 100}


def test_grid_and_settings():
    g = scenario.grid()
    st = scenario.settings()
    assert g[0] == 0 and g[-1] == st["curve_grid"]["max"] and 1.0 in g
    assert 1 + st["slider"]["max"] / 100 <= g[-1]                       # every slider position is on the curves
    assert st["extrapolation"]["low"] < 1 < st["extrapolation"]["high"]


def test_scenario_key_ignores_unchanged_channels():
    a = whatif.scenario_key("all", {"tv": 1.2, "search": 1.0})
    assert a == whatif.scenario_key("all", {"tv": 1.20000001}) and a != whatif.scenario_key("last_52", {"tv": 1.2})


# API ------------------------------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from mmm_tool.api import app as app_mod

    monkeypatch.setattr(app_mod, "RUNS_DIR", tmp_path / "runs")
    c = TestClient(app_mod.app)
    c.runs = tmp_path / "runs"
    return c


def _run(client) -> str:
    run_id = client.post("/api/runs/sample/ins_clean", params={"project": project_in()}).json()["run_id"]
    client.put(f"/api/runs/{run_id}/answers", json={"answers": {"roles": {}}})
    return run_id


def test_whatif_needs_an_accepted_model(client):
    run_id = _run(client)
    r = client.get(f"/api/runs/{run_id}/whatif")
    assert r.status_code == 409 and "accepted model" in r.json()["detail"]


def test_saved_scenarios_are_kept_and_limited(client):
    run_id = _run(client)
    fd = job.fit_dir(run_path(run_id))
    job._write(fd / "status.json", {"state": "done", "status": "accepted", "heartbeat": time.time()})
    job._write(fd / "result.json", {"status": "accepted", "steps": [], "gates": []})
    (fd / "model.pkl").write_bytes(b"not a model")
    one = {"name": "TV down", "period": "all", "multipliers": {"tv": 0.8}, "quick": {"spend": 1, "kpi": 2, "revenue": None}}
    ok = client.put(f"/api/runs/{run_id}/whatif/saved", json={"scenarios": [one, one, one]})
    assert ok.status_code == 200 and len(ok.json()["saved"]) == 3
    too_many = client.put(f"/api/runs/{run_id}/whatif/saved", json={"scenarios": [one] * 4})
    assert too_many.status_code == 400 and "At most 3" in too_many.json()["detail"]
    bad = client.put(f"/api/runs/{run_id}/whatif/saved", json={"scenarios": [{"name": "x"}]})
    assert bad.status_code == 400
    neg = client.post(f"/api/runs/{run_id}/whatif/exact", json={"period": "all", "multipliers": {"tv": -0.5}})
    assert neg.status_code == 400 and "below zero" in neg.json()["detail"]
    stored = json.loads((whatif.folder(run_path(run_id)) / "saved.json").read_text(encoding="utf-8"))
    assert stored["model_key"] == whatif.model_key(run_path(run_id)) and len(stored["scenarios"]) == 3


def test_cached_results_belong_to_one_model(client):
    run_id = _run(client)
    d = run_path(run_id)
    fd = job.fit_dir(d)
    fd.mkdir(parents=True, exist_ok=True)
    (fd / "model.pkl").write_bytes(b"one")
    k1 = whatif.model_key(d)
    time.sleep(0.01)
    (fd / "model.pkl").write_bytes(b"another model")
    assert whatif.model_key(d) != k1                                       # a refit invalidates cached curves


# Calculations on a real fit (opt-in) -------------------------------------------------------------------

def shop(weeks=156, seed=4, rf=False):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"week": pd.date_range("2022-01-03", periods=weeks, freq="7D").strftime("%Y-%m-%d")})
    on = (rng.random(weeks) < 0.45).astype(float)
    df["tv_spend"] = np.round(on * rng.normal(3e5, 3e4, weeks), -2)
    df["search_spend"] = np.round(rng.normal(2e5, 2e4, weeks), -2)
    df["search_clicks"] = np.round(df["search_spend"] / 2.1)
    if rf:
        df["video_spend"] = np.round(rng.normal(1e5, 1.5e4, weeks), -2)
        df["video_reach"] = np.round(df["video_spend"] / 3 * rng.normal(1, 0.05, weeks))
        df["video_frequency"] = np.round(rng.normal(2.5, 0.2, weeks), 2)
    df["price_index"] = np.round(1 + rng.normal(0, 0.02, weeks), 3)
    df["orders"] = np.round(2500 + 0.002 * df["tv_spend"] + 0.003 * df["search_spend"] + rng.normal(0, 50, weeks))
    return df


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    if os.environ.get("MMM_FIT_TESTS") != "1":
        pytest.skip("set MMM_FIT_TESTS=1 to run Meridian fits")
    from mmm_tool.fit import model as m
    from mmm_tool.pipeline import run_pipeline

    out = {}
    for rf in (False, True):
        p = tmp_path_factory.mktemp("g") / f"shop_{rf}.csv"
        shop(rf=rf).to_csv(p, index=False)
        r = run_pipeline(p, {"revenue_per_kpi_value": 40}, accept_defaults=True)
        assert r.d is not None and r.d.status == "complete", r.last.steps[-1].summary
        mmm = m.build(r.d, {**r.d.config, "holdout": None})
        mmm.sample_prior(50, seed=3)
        m.sample_posterior(mmm, TINY, 3)
        out[rf] = (mmm, r)
    return out


@fit_tests
@pytest.mark.parametrize("rf", [False, True])
def test_exact_matches_the_curves_channel_by_channel(fitted, rf):
    mmm, _ = fitted[rf]
    cur = scenario.curves(mmm, "all")
    paid = list(cur["channels"])
    i1 = cur["grid"].index(1.0)
    base = scenario.exact(mmm, {}, "all")
    for c in paid:
        assert base["current"]["kpi_by_channel"][c] == pytest.approx(cur["channels"][c]["kpi"][i1], rel=1e-3)
    m = {paid[0]: 1.5, paid[-1]: 0.5}
    ex = scenario.exact(mmm, m, "all")
    for c, mult in m.items():
        j = cur["grid"].index(mult)
        print(f"\n{c} at {mult}x: exact {ex['scenario']['kpi_by_channel'][c]:,.1f}, curve {cur['channels'][c]['kpi'][j]:,.1f}")
        assert ex["scenario"]["kpi_by_channel"][c] == pytest.approx(cur["channels"][c]["kpi"][j], rel=1e-3)
    s = ex["scenario"]["kpi"]
    assert s["lo"] < s["mean"] < s["hi"] and ex["delta_kpi"]["lo"] <= ex["delta_kpi"]["mean"] <= ex["delta_kpi"]["hi"]
    assert ex["scenario"]["kpi"]["mean"] - ex["current"]["kpi"]["mean"] == pytest.approx(ex["delta_kpi"]["mean"], rel=1e-6)
    assert "revenue" in ex["scenario"] and ex["scenario"]["revenue"]["mean"] > 0


@fit_tests
def test_spend_and_periods(fitted):
    mmm, r = fitted[False]
    cur = scenario.curves(mmm, "all")
    i1 = cur["grid"].index(1.0)
    spend = scenario.spend_in(mmm, None)
    for ch in r.d.channels:
        assert spend[ch.name] == pytest.approx(float(r.d.table[ch.spend].sum()), rel=1e-6)
        assert cur["channels"][ch.name]["spend"][i1] == pytest.approx(spend[ch.name], rel=1e-4)
    last = scenario.curves(mmm, "last_52")
    assert last["weeks"] == 52 and last["last_week"] == cur["last_week"]
    c = next(iter(last["channels"]))
    assert last["channels"][c]["spend"][i1] < cur["channels"][c]["spend"][i1]
    ex = scenario.exact(mmm, {c: 2.0}, "last_52")
    assert ex["scenario"]["spend"] == pytest.approx(ex["current"]["spend"] + last["channels"][c]["spend"][i1], rel=1e-4)


@fit_tests
def test_less_spend_never_more_sales_and_zero_is_zero(fitted):
    mmm, _ = fitted[False]
    cur = scenario.curves(mmm, "all")
    for c, x in cur["channels"].items():
        assert x["kpi"][0] == 0 and all(b >= a - 1e-6 for a, b in zip(x["kpi"], x["kpi"][1:]))
    with pytest.raises(ValueError, match="Not channels"):
        scenario.exact(mmm, {"nonesuch": 1.2}, "all")
    with pytest.raises(ValueError, match="Unknown period"):
        scenario.curves(mmm, "last_9")


# Step 45: the change is per draw (the same draw for scenario and current), then summarised ------------------

@fit_tests
def test_no_change_is_exactly_zero_in_every_draw(fitted):
    mmm, _ = fitted[False]
    paid = [str(c) for c in mmm.input_data.get_all_paid_channels()]
    for mult in ({}, {c: 1.0 for c in paid}):
        d = scenario.change_draws(mmm, mult)
        assert d.size > 1 and (d == 0).all()


@fit_tests
def test_more_of_one_channel_is_better_in_every_draw(fitted):
    mmm, _ = fitted[False]
    paid = [str(c) for c in mmm.input_data.get_all_paid_channels()]
    d = scenario.change_draws(mmm, {paid[0]: 1.2})
    print(f"\n+20% {paid[0]}: change per draw from {d.min():.1f} to {d.max():.1f}")
    assert (d > 0).all()
    x = scenario.exact(mmm, {paid[0]: 1.2})
    assert x["delta_kpi"]["share_better"] == 1.0 and x["delta_kpi"]["favour"] == "very likely better"


@fit_tests
def test_a_resplit_change_is_the_difference_of_the_means(fitted):
    mmm, _ = fitted[False]
    paid = [str(c) for c in mmm.input_data.get_all_paid_channels()]
    x = scenario.exact(mmm, {paid[0]: 1.3, paid[1]: 0.7})
    dk = x["delta_kpi"]
    print(f"\nresplit: change {dk['mean']:.1f} (90%: {dk['lo']:.1f} to {dk['hi']:.1f}); {dk['share_better']:.0%} of draws favour: {dk['favour']}")
    assert dk["mean"] == pytest.approx(x["scenario"]["kpi"]["mean"] - x["current"]["kpi"]["mean"], rel=1e-9, abs=1e-6)
    assert 0 <= dk["share_better"] <= 1 and dk["favour"] == scenario.favour(dk["share_better"])
    sp = x["delta_spend"]
    assert sp["amount"] == pytest.approx(x["scenario"]["spend"] - x["current"]["spend"])
    assert x["weeks"] == len(mmm.input_data.time.values) and x["first_week"] < x["last_week"]


@fit_tests
@pytest.mark.parametrize("period", ["all", "last_52"])
def test_current_plan_is_the_model_summarys_paid_total(fitted, period):
    from mmm_tool.fit import model as m
    from mmm_tool.summary.build import roi_table

    mmm, _ = fitted[False]
    x = scenario.exact(mmm, {}, period)
    sel = scenario.periods(mmm)[period]
    paid_total = next(r for r in roi_table(m.analyzer(mmm), True, sel)["rows"] if r["channel"] == "all paid")
    print(f"\n{period}: current {x['current']['kpi']['mean']:,.1f}, Model Summary {paid_total['kpi_driven']['mean']:,.1f}")
    assert x["current"]["kpi"]["mean"] == pytest.approx(paid_total["kpi_driven"]["mean"], rel=0.005)
    assert x["weeks"] == (52 if period == "last_52" else len(mmm.input_data.time.values))


def test_favour_wording():
    assert [scenario.favour(s) for s in (1.0, 0.9, 0.75, 0.7, 0.5, 0.3, 0.2, 0.05, 0.0)] == [
        "very likely better", "very likely better", "probably better", "probably better", "can't tell", "can't tell",
        "probably worse", "very likely worse", "very likely worse"]
