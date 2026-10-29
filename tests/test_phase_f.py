"""Phase F: the Model Summary screen (steps 37-42).

tests/data/phase_f/summary.json was produced by the real Phase E path (an
accepted quick fit of the clean sample), so the API and the screen are tested
on real numbers without fitting. Tests that fit with Meridian run only with
MMM_FIT_TESTS=1.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from mmm_tool import samples
from mmm_tool.fit import job
from mmm_tool.pipeline import run_pipeline
from helpers import project_in, run_path  # noqa: F401

FIXTURE = Path(__file__).parent / "data" / "phase_f" / "summary.json"
fit_tests = pytest.mark.skipif(os.environ.get("MMM_FIT_TESTS") != "1", reason="set MMM_FIT_TESTS=1 to run Meridian fits")
TINY = {"n_chains": 2, "n_adapt": 20, "n_burnin": 20, "n_keep": 20}


def accepted(run_dir: Path, inputs: str) -> None:
    """Make a run look like Phase E accepted a model and stored its summary."""
    fd = job.fit_dir(run_dir)
    job._write(fd / "status.json", {"state": "done", "status": "accepted", "heartbeat": time.time()})
    job._write(fd / "result.json", {"status": "accepted", "inputs": inputs, "steps": [], "gates": [],
                                    "model": {"saved_at": "2026-10-28T07:54:23+00:00"}})
    shutil.copy(FIXTURE, fd / "summary.json")


# The stored summary itself ----------------------------------------------------

@pytest.fixture(scope="module")
def summary():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_summary_has_every_panel(summary):
    assert set(summary) >= {"version", "fit", "badges", "roi", "breakdown", "curves", "carryover", "paid"}
    v = summary["version"]
    assert v["last_week"] == summary["fit"]["weeks"][-1] and v["weeks"] == len(summary["fit"]["weeks"])
    assert summary["roi"]["last_52"] is not None                     # 156 weeks: a last-52-weeks view too


def test_breakdown_parts_add_up(summary):
    """Baseline plus every channel equals the predicted total, in total and every week."""
    b = summary["breakdown"]["all"]
    assert np.isclose(b["baseline"]["mean"] + sum(c["mean"] for c in b["channels"].values()), b["total"])
    w = summary["breakdown"]["weekly"]
    weekly = np.asarray(w["baseline"]) + sum(np.asarray(v) for v in w["channels"].values())
    assert np.allclose(weekly, summary["fit"]["expected"], rtol=1e-6, atol=0.05)
    assert np.isclose(sum(summary["fit"]["expected"]), b["total"], rtol=1e-6)


def test_roi_table_matches_curves_and_ranges(summary):
    rows = {r["channel"]: r for r in summary["roi"]["all"]["rows"]}
    for c in summary["paid"]:
        r, curve = rows[c], summary["curves"][c]
        i1 = curve["multiplier"].index(1.0)
        assert np.isclose(curve["kpi"][i1], r["kpi_driven"]["mean"], rtol=1e-6)     # the dot on the curve is today
        assert np.isclose(curve["spend"][i1], r["spend"], rtol=1e-6)
        assert r["roi"]["lo"] <= r["roi"]["mean"] <= r["roi"]["hi"]
    assert "all paid" in rows and np.isclose(rows["all paid"]["spend"], sum(rows[c]["spend"] for c in summary["paid"]))


def test_carryover_starts_at_one_and_fades(summary):
    for c, d in summary["carryover"].items():
        assert d["weeks"][0] == 0 and np.isclose(d["mean"][0], 1.0)
        assert all(b <= a + 1e-9 for a, b in zip(d["mean"], d["mean"][1:])), c
        assert d["half_life"] is None or 0 < d["half_life"] < len(d["weeks"])


def test_badges_carry_reasons(summary):
    tv = {b["badge"]: b for b in summary["badges"]["tv"]}
    assert "calibrated" in tv and "2024-04-01" in tv["calibrated"]["why"]          # the lift test behind it
    assert "lift test" in tv["prior_driven"]["why"]                               # prior-driven because of it


# API -------------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from mmm_tool.api import app as app_mod

    monkeypatch.setattr(app_mod, "RUNS_DIR", tmp_path / "runs")
    return TestClient(app_mod.app), app_mod


def _confirmed_run(client) -> str:
    run_id = client.post("/api/runs/sample/ins_clean", params={"project": project_in()}).json()["run_id"]
    client.put(f"/api/runs/{run_id}/answers", json={"answers": {"roles": {}}})
    return run_id


def test_summary_endpoint(client):
    c, app_mod = client
    run_id = _confirmed_run(c)
    assert c.get(f"/api/runs/{run_id}/summary").status_code == 404
    assert c.post(f"/api/runs/{run_id}/summary").status_code == 409            # no accepted model to rebuild from
    run = run_pipeline(run_path(run_id) / "ins_clean.csv", {"roles": {}})
    accepted(run_path(run_id), job.inputs_hash(run.d))
    s = c.get(f"/api/runs/{run_id}/summary").json()
    assert s["stale"] is False and s["version"]["kpi"] == "new_policies"
    assert s["downloads"] == {"meridian_html": False, "model": False}
    assert c.get(f"/api/runs/{run_id}").json()["fit"]["summary"] is True
    # A change of settings after the fit makes it out of date.
    c.put(f"/api/runs/{run_id}/answers", json={"answers": {"roles": {}, "knots": 20}})
    assert c.get(f"/api/runs/{run_id}/summary").json()["stale"] is True


def _node_playwright() -> bool:
    if not shutil.which("node"):
        return False
    return subprocess.run(["node", "-e", "require('playwright')"], capture_output=True).returncode == 0


@pytest.mark.skipif(not _node_playwright(), reason="needs node with playwright")
def test_summary_screen_in_the_browser(tmp_path):
    """Trust before results; both themes; a 400px screen; charts with hover; the period switch."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    runs = tmp_path / "runs"
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "mmm_tool.api.app:app", "--port", str(port)],
                              env={**os.environ, "MMM_RUNS_DIR": str(runs)}, cwd=Path(__file__).parents[1],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        import httpx

        base = f"http://127.0.0.1:{port}"
        for _ in range(120):
            try:
                httpx.get(base + "/api/samples", timeout=1)
                break
            except httpx.HTTPError:
                time.sleep(0.5)
        slug = httpx.post(base + "/api/projects", json={"name": "Browser", "packs": ["india", "insurance"], "calendar": "india_insurance"}, timeout=30).json()["slug"]
        run_id = httpx.post(base + "/api/runs/sample/ins_clean", params={"project": slug}, timeout=120).json()["run_id"]
        httpx.put(f"{base}/api/runs/{run_id}/answers", json={"answers": {"roles": {}}}, timeout=120)
        d = runs / "projects" / slug / "runs" / run_id
        run = run_pipeline(d / "ins_clean.csv", {"roles": {}})
        accepted(d, job.inputs_hash(run.d))
        out = subprocess.run(["node", str(Path(__file__).parent / "browser" / "phase_f.js"), base, run_id, str(tmp_path)],
                             capture_output=True, text=True, timeout=300)
        assert out.returncode == 0, out.stderr
        m = json.loads(out.stdout.strip().splitlines()[-1])
        assert m["errors"] == []
        assert m["order"] == [37, 40, 39, 38, 41, 42]                # fit health and badges above the ROI table
        assert "model 94544f847494" in m["header"] and "data to 2025-12-22" in m["header"]
        assert "Model Summary" in m["rail_f"]
        assert m["fit_tooltip"].startswith("20") and "predicted" in m["fit_tooltip"]
        assert m["roi_rows"] == 10 and m["curves"] == 9 and m["decays"] == 9 and m["badges"] >= 9
        assert "last 52 weeks" in m["roi_sum_52"] and "last 52 weeks" not in m["roi_sum_all"]
        assert m["bg_light"] != m["bg_dark"] and m["phone_sideways_scroll"] is False
    finally:
        server.terminate()
        server.wait(timeout=20)


# Meridian fits (slow) ------------------------------------------------------------------

@fit_tests
def test_accepted_fit_writes_the_summary_and_it_rebuilds(tmp_path, monkeypatch):
    pytest.importorskip("meridian")
    from mmm_tool.fit import run_phase_e
    from mmm_tool.fit.context import settings

    cfg = settings()   # a tiny sampler can't pass the gates; relax them so the final fit and Phase F run
    monkeypatch.setitem(cfg["convergence"], "rhat_max", 1e9)
    monkeypatch.setitem(cfg["holdout"], "test_mape_max", 10.0)
    monkeypatch.setitem(cfg["holdout"], "season_weeks", 10_000)
    monkeypatch.setitem(cfg["plausibility"], "roi_max", 1e9)
    monkeypatch.setitem(cfg["plausibility"], "roi_min", 0.0)
    monkeypatch.setitem(cfg["plausibility"], "baseline_share", [0.0, 1.0])
    r = run_pipeline(samples.write_clean(tmp_path), accept_defaults=True)
    e = run_phase_e(r.d, {"sampler": TINY}, out_dir=tmp_path / "fit")
    assert e.status == "accepted" and e.model.get("summary") is True, e.model
    s = json.loads((tmp_path / "fit" / "summary.json").read_text())
    b = s["breakdown"]["all"]
    assert np.isclose(b["baseline"]["mean"] + sum(c["mean"] for c in b["channels"].values()), sum(s["fit"]["expected"]), rtol=1e-6)
    assert (tmp_path / "fit" / "model_results_summary.html").stat().st_size > 10_000
    assert s["version"]["data_version"] == e.model["data_version"]
