"""Phase E tests.

The gate rules and the job bookkeeping are tested on plain numbers and files.
Real fits take minutes, so the tests that fit with Meridian run only when
MMM_FIT_TESTS=1 is set (and google-meridian is installed).
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import pytest

from mmm_tool import samples
from mmm_tool.fit import gates, job
from mmm_tool.fit.context import FitOptions, sampler_settings, settings
from mmm_tool.pipeline import run_pipeline

fit_tests = pytest.mark.skipif(os.environ.get("MMM_FIT_TESTS") != "1", reason="set MMM_FIT_TESTS=1 to run Meridian fits")
TINY = {"n_chains": 2, "n_adapt": 20, "n_burnin": 20, "n_keep": 20}


# Step 28 -----------------------------------------------------------------

def test_prior_check_covers_and_width():
    assert gates.prior_check(800, 1500, 1000, 10)["ok"]
    miss = gates.prior_check(1200, 1500, 1000, 10)
    assert not miss["ok"] and not miss["covers"]
    wide = gates.prior_check(0, 20000, 1000, 10)
    assert wide["covers"] and wide["width_ratio"] == 20 and not wide["ok"]


# Step 30 -----------------------------------------------------------------

def test_convergence_needs_every_rhat_below_the_limit():
    assert gates.converged({"roi_m": 1.01, "tau_g": 1.05}, 1.1)["ok"]
    res = gates.converged({"roi_m": 1.01, "alpha_m": 1.3}, 1.1)
    assert not res["ok"] and res["bad"] == {"alpha_m": 1.3} and res["worst"]["parameter"] == "alpha_m"
    assert not gates.converged({"roi_m": float("nan")}, 1.1)["ok"]  # an R-hat that can't be computed isn't a pass


# Step 31 -----------------------------------------------------------------

def test_mape_skips_zero_weeks():
    assert math.isclose(gates.mape([100, 200, 0], [110, 180, 5]), 0.1)


def test_seasonal_naive_uses_the_same_week_a_year_earlier():
    y = np.r_[np.full(52, 100.0), np.full(52, 110.0)]
    test = np.zeros(104, bool)
    test[-13:] = True
    assert math.isclose(gates.seasonal_naive_mape(y, test, 52), 10 / 110)


def test_holdout_gate():
    assert gates.holdout_ok(0.06, 0.12, 0.10)["ok"]
    assert not gates.holdout_ok(0.12, 0.20, 0.10)["ok"]          # above 10%
    res = gates.holdout_ok(0.08, 0.05, 0.10)                      # worse than last year's sales
    assert not res["ok"] and res["under_max"] and not res["beats_naive"]
    assert gates.holdout_ok(0.08, float("nan"), 0.10)["ok"]      # no year of history before the test weeks


# Step 32 -----------------------------------------------------------------

LIMITS = dict(roi_min=0.05, roi_max=20, share_range=(0.2, 0.95), tolerance=0.001)


def test_plausibility_gate():
    assert gates.plausible({"tv": 1.2, "search": 3.0}, 0.6, 0.0002, **LIMITS)["ok"]
    res = gates.plausible({"tv": 1.2, "search": 31.0}, 0.6, 0.0, **LIMITS)
    assert not res["ok"] and res["roi_out_of_range"] == {"search": 31.0}
    assert not gates.plausible({"tv": 1.2}, 0.97, 0.0, **LIMITS)["baseline_ok"]
    assert not gates.plausible({"tv": 1.2}, 0.6, 0.02, **LIMITS)["reconciles"]
    # ROI in KPI units per rupee is tiny by nature; the money range doesn't apply.
    assert gates.plausible({"tv": 0.0004}, 0.6, 0.0, roi_in_money=False, **LIMITS)["ok"]


# Steps 33-35 --------------------------------------------------------------

def test_narrowing():
    assert math.isclose(gates.narrowing((0.3, 5.3), (1.0, 3.0)), 0.6)
    assert gates.narrowing((0.3, 5.3), (0.3, 5.0)) < settings()["prior_driven"]["min_narrowing"]


def test_stability_cv():
    assert gates.cv([2.0, 2.1, 1.9]) < 0.3
    assert gates.cv([1.0, 3.0, 0.5]) > 0.3


def test_placebo_credited_only_when_pulled_up_and_clear_of_zero():
    assert not gates.placebo_credited(prior_median=1.2, post_median=0.8, post_lo=0.3, roi_min=0.05)  # pulled down
    assert not gates.placebo_credited(prior_median=1.2, post_median=1.5, post_lo=0.01, roi_min=0.05)  # as good as zero
    assert gates.placebo_credited(prior_median=1.2, post_median=2.5, post_lo=1.4, roi_min=0.05)


# Settings -----------------------------------------------------------------

def test_samplers_from_fit_yaml():
    full = sampler_settings("full")
    assert (full["n_chains"], full["n_adapt"], full["n_burnin"], full["n_keep"]) == (4, 500, 500, 1000)  # the roadmap's
    assert sampler_settings(TINY)["n_keep"] == 20
    with pytest.raises(ValueError):
        sampler_settings("slow")
    with pytest.raises(ValueError):
        sampler_settings({**TINY, "n_chains": 1})  # R-hat needs two chains
    assert FitOptions.from_dict({"sampler": "full", "placebo": 1}).placebo is True


# Job bookkeeping ------------------------------------------------------------

def test_job_status_and_silent_job(tmp_path):
    assert job.status(tmp_path) is None
    fd = job.fit_dir(tmp_path)
    job._write(fd / "status.json", {"state": "running", "heartbeat": time.time(), "started": time.time()})
    assert job.status(tmp_path)["state"] == "running"
    job._write(fd / "status.json", {"state": "running", "heartbeat": time.time() - job.STALE_SECONDS - 5})
    st = job.status(tmp_path)
    assert st["state"] == "failed" and "unexpectedly" in st["message"]


def test_result_attached_once_done(tmp_path):
    fd = job.fit_dir(tmp_path)
    job._write(fd / "status.json", {"state": "done", "status": "accepted", "heartbeat": time.time()})
    job._write(fd / "result.json", {"status": "accepted", "inputs": "abc"})
    assert job.status(tmp_path)["result"]["status"] == "accepted"


def test_inputs_hash_changes_with_settings(tmp_path):
    r = run_pipeline(samples.write_clean(tmp_path), accept_defaults=True)
    r2 = run_pipeline(samples.write_clean(tmp_path), {"knots": 20}, accept_defaults=True)
    assert job.inputs_hash(r.d) == job.inputs_hash(r.d) != job.inputs_hash(r2.d)


def test_cli_explains_open_questions(tmp_path, capsys):
    from mmm_tool.fit.__main__ import main
    from mmm_tool.fit.model import meridian_installed

    code = main([str(samples.write_clean(tmp_path))])  # roles never confirmed
    err = capsys.readouterr().err
    if meridian_installed():
        assert code == 1 and "needs answers first" in err and "roles" in err
    else:
        assert code == 2 and "google-meridian" in err


# API ----------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from mmm_tool.api import app as app_mod

    monkeypatch.setattr(app_mod, "RUNS_DIR", tmp_path / "runs")
    return TestClient(app_mod.app)


def test_fit_api_refuses_until_phase_d_is_complete(client):
    from mmm_tool.fit.model import meridian_installed

    run_id = client.post("/api/runs/sample/ins_clean").json()["run_id"]  # roles not confirmed yet
    r = client.post(f"/api/runs/{run_id}/fit", json={"sampler": "quick"})
    assert r.status_code == (409 if meridian_installed() else 503)
    assert client.get(f"/api/runs/{run_id}/fit").status_code == 404
    assert client.get(f"/api/runs/{run_id}/model.pkl").status_code == 404
    assert client.get("/api/fit-settings").json()["samplers"]["full"]["n_keep"] == 1000
    body = client.get(f"/api/runs/{run_id}").json()
    assert body["fit"] is None and "meridian_installed" in body


def test_fit_api_marks_a_fit_out_of_date(client):
    run_id = client.post("/api/runs/sample/ins_clean").json()["run_id"]
    client.put(f"/api/runs/{run_id}/answers", json={"answers": {"roles": {}}})
    from mmm_tool.api import app as app_mod

    d = app_mod.RUNS_DIR / run_id
    job._write(job.fit_dir(d) / "status.json", {"state": "done", "status": "accepted", "heartbeat": time.time()})
    job._write(job.fit_dir(d) / "result.json", {"status": "accepted", "inputs": "not-these-inputs"})
    assert client.get(f"/api/runs/{run_id}").json()["fit"]["stale"] is True


# Meridian fits (slow) ---------------------------------------------------------

@fit_tests
def test_phase_e_end_to_end(tmp_path):
    pytest.importorskip("meridian")
    from mmm_tool.fit import run_phase_e

    r = run_pipeline(samples.write_clean(tmp_path), accept_defaults=True)
    seen = []
    e = run_phase_e(r.d, {"sampler": TINY, "stability": True, "placebo": True}, out_dir=tmp_path / "fit",
                    progress=lambda step, msg, frac: seen.append((step, frac)))
    assert e.error is None and e.status in ("accepted", "rejected")
    assert [s.step for s in e.steps] == list(range(28, 37))
    assert [s for s, _ in seen][0] == 28 and seen[-1][1] == 1.0
    assert all(b >= a for (_, a), (_, b) in zip(seen, seen[1:]))  # progress never goes back
    names = [c.name for c in r.d.channels]
    assert sorted(x["channel"] for x in e.channels if x["paid"]) == sorted(names)
    s32 = e.steps[4].details
    assert abs(s32["contributions"]["reconcile_error"]) < 1e-3  # baseline + contributions = the fit
    s34 = e.steps[6].details
    assert len(s34["windows"]) == 4 and all(len(c["roi"]) == 4 for c in s34["channels"])
    assert "placebo" in e.steps[7].summary.lower()
    if e.status == "accepted":
        assert (tmp_path / "fit" / "model.pkl").exists()
        meta = json.loads((tmp_path / "fit" / "model_meta.json").read_text())
        assert meta["data_version"] == e.model["data_version"]
    else:
        assert e.steps[8].details["skipped"] and not (tmp_path / "fit" / "model.pkl").exists()


@fit_tests
def test_background_job_through_the_api(client):
    pytest.importorskip("meridian")
    run_id = client.post("/api/runs/sample/ins_clean").json()["run_id"]
    client.put(f"/api/runs/{run_id}/answers", json={"answers": {"roles": {}}})
    r = client.post(f"/api/runs/{run_id}/fit", json={"sampler": "quick"})
    assert r.status_code == 200 and r.json()["state"] in job.ACTIVE
    assert client.post(f"/api/runs/{run_id}/fit", json={"sampler": "quick"}).status_code == 409  # one at a time
    assert client.delete(f"/api/runs/{run_id}/fit").json()["state"] == "cancelled"


@fit_tests
def test_accepted_model_is_saved_and_reloads(tmp_path, monkeypatch):
    """With the gates relaxed (a tiny sampler can't pass them), the final fit runs and saves."""
    pytest.importorskip("meridian")
    from mmm_tool.fit import model as m
    from mmm_tool.fit import run_phase_e

    cfg = settings()
    monkeypatch.setitem(cfg["convergence"], "rhat_max", 1e9)
    monkeypatch.setitem(cfg["holdout"], "test_mape_max", 10.0)
    monkeypatch.setitem(cfg["holdout"], "season_weeks", 10_000)  # no naive forecast to beat
    monkeypatch.setitem(cfg["plausibility"], "roi_max", 1e9)
    monkeypatch.setitem(cfg["plausibility"], "roi_min", 0.0)
    monkeypatch.setitem(cfg["plausibility"], "baseline_share", [0.0, 1.0])
    r = run_pipeline(samples.write_clean(tmp_path), accept_defaults=True)
    e = run_phase_e(r.d, {"sampler": TINY}, out_dir=tmp_path / "fit")
    assert e.status == "accepted", [(s.step, s.summary) for s in e.steps]
    path = tmp_path / "fit" / "model.pkl"
    assert e.model["path"] == str(path) and path.exists()
    meta = json.loads((tmp_path / "fit" / "model_meta.json").read_text())
    assert meta["gates"] == {"30": True, "31": True, "32": True} and meta["config"]["holdout"] is not None
    loaded = m.load(path)
    assert len(m.weeks(loaded)) == 156 and loaded.model_spec.holdout_id is None  # refitted on every week
