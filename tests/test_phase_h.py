"""Phase H: Plan, Optimize (steps 48-54).

The plan's mapping to BudgetOptimizer.optimize() arguments and its checks need no
Meridian. A real optimization on a small fit runs only with MMM_FIT_TESTS=1.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pytest

from mmm_tool.fit import job
from mmm_tool.plan.optimize import PlanError, check, defaults

HIST = {"tv": 600.0, "search": 300.0, "radio": 100.0}
PERIODS = {"all": ("2023-01-02", "2025-12-22"), "last_52": ("2024-12-30", "2025-12-22")}


def plan(**kw):
    return {**defaults(list(HIST), money=True), **kw}


def test_defaults_map_to_meridians_fixed_budget_call():
    c = check(plan(), HIST, True, PERIODS)
    a = c["args"]
    assert a["fixed_budget"] and a["budget"] == 1000 and (a["start_date"], a["end_date"]) == PERIODS["all"]
    assert a["spend_constraint_lower"] == [0.3, 0.3, 0.3] and a["spend_constraint_upper"] == [0.3, 0.3, 0.3]
    assert a["use_kpi"] is False and a["confidence_level"] == 0.9
    assert [r["planned"] for r in c["channels"]] == [600, 300, 100]


def test_budget_period_and_objective():
    c = check(plan(budget=2000, period="last_52", objective="kpi"), HIST, True, PERIODS)
    assert c["args"]["budget"] == 2000 and c["args"]["start_date"] == PERIODS["last_52"][0] and c["args"]["use_kpi"]
    assert [r["planned"] for r in c["channels"]] == [1200, 600, 200]          # the budget split as historically
    with pytest.raises(PlanError, match="money value"):
        check(plan(objective="revenue"), HIST, False, PERIODS)
    with pytest.raises(PlanError, match="above zero"):
        check(plan(budget=0), HIST, True, PERIODS)


def test_flexible_budget_uses_a_target():
    c = check(plan(budget_type="flexible", target="mroi", target_value=1.5), HIST, True, PERIODS)
    assert c["args"]["fixed_budget"] is False and c["args"]["target_mroi"] == 1.5 and "budget" not in c["args"]
    with pytest.raises(PlanError, match="target"):
        check(plan(budget_type="flexible", target_value=0), HIST, True, PERIODS)


def test_limits_locks_and_floors():
    p = plan(limits={"tv": {"down": 10, "up": 50}, "search": {"down": 30, "up": 30}, "radio": {"down": 30, "up": 30}},
             locks=["search"], floors={"radio": 90})
    c = check(p, HIST, True, PERIODS)
    lo, hi = c["args"]["spend_constraint_lower"], c["args"]["spend_constraint_upper"]
    assert (lo[0], hi[0]) == (0.1, 0.5)
    assert (lo[1], hi[1]) == (0.0, 0.0)                       # locked: stays at its planned spend
    assert lo[2] == pytest.approx(0.1) and hi[2] == 0.3       # a floor of 90 on 100 planned: may drop 10% at most
    rows = {r["channel"]: r for r in c["channels"]}
    assert rows["radio"]["min"] == pytest.approx(90) and rows["search"]["note"] == "locked"


@pytest.mark.parametrize("bad, msg", [
    ({"floors": {"radio": 150}}, "above its planned spend"),
    ({"locks": ["nonesuch"]}, "not a channel"),
    ({"limits": {"tv": {"down": 120, "up": 30}}}, "may go down 0-100%"),
    ({"limits": {"tv": {"down": 30, "up": 900}}}, "up 0-200%"),
    ({"period": "last_9"}, "Unknown period"),
])
def test_plan_problems(bad, msg):
    with pytest.raises(PlanError, match=msg):
        check(plan(**bad), HIST, True, PERIODS)


def test_a_channel_with_no_spend_in_the_period_stays_at_zero():
    c = check(plan(), {**HIST, "radio": 0.0}, True, PERIODS)
    r = next(x for x in c["channels"] if x["channel"] == "radio")
    assert (r["down"], r["up"], r["note"]) == (0.0, 0.0, "no spend in the period")


# API ----------------------------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from mmm_tool.api import app as app_mod

    monkeypatch.setattr(app_mod, "RUNS_DIR", tmp_path / "runs")
    c = TestClient(app_mod.app)
    c.runs = tmp_path / "runs"
    return c


def _accepted_run(client) -> str:
    run_id = client.post("/api/runs/sample/ins_clean").json()["run_id"]
    client.put(f"/api/runs/{run_id}/answers", json={"answers": {"roles": {}}})
    fd = job.fit_dir(client.runs / run_id)
    job._write(fd / "status.json", {"state": "done", "status": "accepted", "heartbeat": time.time()})
    job._write(fd / "result.json", {"status": "accepted", "steps": [], "gates": []})
    (fd / "model.pkl").write_bytes(b"not a model")
    return run_id


def test_plan_needs_an_accepted_model(client):
    run_id = client.post("/api/runs/sample/ins_clean").json()["run_id"]
    client.put(f"/api/runs/{run_id}/answers", json={"answers": {"roles": {}}})
    assert client.get(f"/api/runs/{run_id}/plan").status_code == 409


def test_plan_screen_and_check(client):
    run_id = _accepted_run(client)
    d = client.get(f"/api/runs/{run_id}/plan").json()
    assert set(d["periods"]) == {"all", "last_52"} and d["defaults"]["budget_type"] == "fixed"
    from mmm_tool.pipeline import run_pipeline

    run = run_pipeline(client.runs / run_id / "ins_clean.csv", {"roles": {}})
    for ch in run.d.channels:
        assert d["hist"]["all"][ch.name] == pytest.approx(float(run.d.table[ch.spend].sum()))
    ok = client.post(f"/api/runs/{run_id}/plan/check", json={"plan": d["defaults"]}).json()
    assert ok["ok"] and ok["args"]["budget"] == pytest.approx(sum(d["hist"]["all"].values()))
    first = d["channels"][0]
    bad = client.post(f"/api/runs/{run_id}/plan/check", json={"plan": {**d["defaults"], "floors": {first: 1e15}}}).json()
    assert bad["ok"] is False and "above its planned spend" in bad["problems"]


# A real optimization (opt-in) ------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    if os.environ.get("MMM_FIT_TESTS") != "1":
        pytest.skip("set MMM_FIT_TESTS=1 to run Meridian fits")
    from test_phase_g import TINY, shop

    from mmm_tool.fit import model as m
    from mmm_tool.pipeline import run_pipeline

    p = tmp_path_factory.mktemp("h") / "shop.csv"
    shop(rf=True).to_csv(p, index=False)                     # three channels, one reach-and-frequency
    r = run_pipeline(p, {"revenue_per_kpi_value": 40}, accept_defaults=True)
    mmm = m.build(r.d, {**r.d.config, "holdout": None})
    mmm.sample_prior(50, seed=3)
    m.sample_posterior(mmm, TINY, 3)
    return mmm, r


@pytest.mark.skipif(os.environ.get("MMM_FIT_TESTS") != "1", reason="set MMM_FIT_TESTS=1 to run Meridian fits")
def test_optimize_keeps_the_budget_limits_and_locks(fitted, tmp_path):
    from mmm_tool.plan.optimize import optimize
    from mmm_tool.whatif.scenario import spend_in

    mmm, r = fitted
    hist = spend_in(mmm, None)
    weeks = [str(t)[:10] for t in mmm.input_data.time.values]
    periods = {"all": (weeks[0], weeks[-1])}
    chans = list(hist)
    p = {**defaults(chans, True), "locks": [chans[-1]], "limits": {c: {"down": 30, "up": 50} for c in chans}}
    checked = check(p, hist, True, periods)
    res = optimize(mmm, checked, tmp_path / "report")
    after = {c: res["after"]["channels"][c]["spend"] for c in chans}
    before = {c: res["before"]["channels"][c]["spend"] for c in chans}
    print(f"\nbefore {before}\nafter  {after}\nchange {res['totals']['delta_kpi']}")
    assert sum(after.values()) == pytest.approx(sum(before.values()), rel=1e-6)          # fixed budget
    for row in checked["channels"]:
        c = row["channel"]
        assert row["min"] * (1 - 1e-3) - 1e3 <= after[c] <= row["max"] * (1 + 1e-3) + 1e3
    assert after[chans[-1]] == pytest.approx(before[chans[-1]], rel=1e-3)                 # locked
    assert any(abs(after[c] / before[c] - 1) > 0.01 for c in chans[:-1])                 # the others moved
    assert res["after"]["attrs"]["total_incremental_outcome"] >= res["before"]["attrs"]["total_incremental_outcome"] - 1e-6
    t = res["totals"]
    assert t["scenario"]["kpi"]["mean"] - t["current"]["kpi"]["mean"] == pytest.approx(t["delta_kpi"]["mean"], rel=1e-6)
    assert res["report"] == "optimization_summary.html" and (tmp_path / "report" / "optimization_summary.html").exists()
