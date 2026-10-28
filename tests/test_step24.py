"""Step 24: knots preview, on insurance_mmm_sample.csv (156 weeks).

The preview and the knot spacing use Meridian's own knot weights, so most tests
need google-meridian. Run with -s to see the printed values.
"""
from __future__ import annotations

import json
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from mmm_tool.configure import baseline
from mmm_tool.pipeline import run_pipeline

SAMPLE = Path(__file__).parent / "data" / "phase_b_tests" / "insurance_mmm_sample.csv"
needs_meridian = pytest.mark.skipif(not baseline.meridian_installed(), reason="needs google-meridian")


def step24(r):
    return next(s for s in r.d.steps if s.step == 24)


@pytest.fixture(scope="module")
def run():
    return run_pipeline(SAMPLE, {}, accept_defaults=True)


@pytest.fixture(scope="module")
def preview(run):
    cvs = {c["channel"]: c["cv"] for s in run.c.steps if s.step == 16 for c in s.details["channels"]}
    outliers = [o["date"] for s in run.b.steps if s.step == 11 for o in s.details.get("outliers", [])]
    return baseline.preview(run.d, cvs, outliers)


@needs_meridian
def test_knot_locations_and_weights_come_from_meridian():
    info = baseline.knot_info(156, 12)
    assert list(info.knot_locations) == [0, 14, 28, 42, 56, 70, 84, 98, 112, 126, 140, 155]
    w = np.asarray(info.weights)[:, 20]
    assert np.round(w[1], 2) == 0.57 and np.round(w[2], 2) == 0.43 and np.isclose(w.sum(), 1)
    assert baseline.spacing(info.knot_locations) == 14


@needs_meridian
def test_summary_gives_the_real_spacing(run):
    s = step24(run)
    assert s.summary == "12 knots, one about every 14 weeks (the default)."
    assert s.details["locations"][-1] == 155 and s.details["max"] == 39
    assert [q["k"] for q in s.details["quick"]] == [1, 6, 12, 19, 39]


@needs_meridian
@pytest.mark.parametrize("k,level,verdict,text", [
    (19, None, "pass", None),
    (20, "amber", "warn", "more than every 2 months"),
    (27, "red", "warn", "likely to absorb campaign effects"),
    (1, None, "pass", None),
])
def test_spacing_warnings(k, level, verdict, text):
    r = run_pipeline(SAMPLE, {"knots": k}, accept_defaults=True)
    s = step24(r)
    assert s.details["level"] == level and s.verdict.value == verdict
    if text:
        assert text in s.warnings[0]
    if k == 1:
        assert s.summary.startswith("1 knot: a flat baseline")


def test_summary_without_meridian(monkeypatch):
    monkeypatch.setattr(baseline, "meridian_installed", lambda: False)
    s = step24(run_pipeline(SAMPLE, {}, accept_defaults=True))
    assert s.summary == "12 knots over 156 weeks (the default)." and s.details["locations"] is None


def test_decision_log_keeps_the_default():
    r = run_pipeline(SAMPLE, {"knots": 19}, accept_defaults=True)
    entry = next(x for x in r.d.decision_log if x["decision"] == "knots")
    assert (entry["answer"], entry["source"], entry["default"]) == (19, "user", 12)
    assert r.d.config["knots"] == 19


@needs_meridian
def test_preview_covers_every_k(preview):
    assert sorted(preview["fits"]) == list(range(1, 40))
    f12 = preview["fits"][12]
    assert len(f12["baseline"]) == 156 and len(f12["levels"]) == 12
    assert f12["locations"] == [0, 14, 28, 42, 56, 70, 84, 98, 112, 126, 140, 155]
    # The baseline is the knot part only: at a knot week it equals that knot's level.
    assert all(np.isclose(f12["baseline"][t], lv, atol=0.02) for t, lv in zip(f12["locations"], f12["levels"]))
    assert set(preview["strips"]) == {"outdoor", "print", "radio", "tv", "youtube_ott"}  # CV above 0.5 in step 16


@needs_meridian
def test_more_knots_leave_less_for_media(preview):
    s12, s39 = preview["share"][11], preview["share"][38]
    print(f"\nleft for media: K=12 {s12:.1%}, K=39 {s39:.1%}")
    assert s39 < s12


@needs_meridian
def test_more_knots_follow_the_bursts(preview):
    f12, f39 = preview["follow"][11], preview["follow"][38]
    print(f"\nbaseline follows bursts: K=12 {f12:.3f} ({preview['follow_name'][11]}), K=39 {f39:.3f} ({preview['follow_name'][38]})")
    assert f39 > f12
    assert preview["follow"][0] is None  # a flat baseline can't follow anything


@needs_meridian
def test_flat_baseline_misses_the_trend(preview):
    print(f"\ntrend after controls: {preview['trend']['slope']:.2f} a week, t = {preview['trend']['t']:.2f}")
    assert preview["trend"]["warn"] and abs(preview["trend"]["t"]) > 2


# API and browser -----------------------------------------------------------------

ANSWERS = {"roles": {}, "mediator:avg_premium_index": "exclude"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from mmm_tool.api import app as app_mod

    monkeypatch.setattr(app_mod, "RUNS_DIR", tmp_path / "runs")
    return TestClient(app_mod.app)


def _upload(client) -> str:
    with SAMPLE.open("rb") as f:
        run_id = client.post("/api/runs", files={"file": (SAMPLE.name, f, "text/csv")}).json()["run_id"]
    client.put(f"/api/runs/{run_id}/answers", json={"answers": ANSWERS})
    return run_id


@needs_meridian
def test_preview_and_suggestion_endpoints(client):
    run_id = _upload(client)
    p = client.get(f"/api/runs/{run_id}/baseline-preview").json()
    assert p["default"] == 12 and p["max"] == 39 and len(p["fits"]) == 39
    s = client.get(f"/api/runs/{run_id}/knots-suggestion").json()
    print(f"\nMeridian AKS suggests {s['knots']} knots")
    assert s["knots"] == len(s["locations"]) and s["knots"] > 1
    r = client.put(f"/api/runs/{run_id}/answers", json={"answers": {**ANSWERS, "knots": 19}}).json()
    assert r["result_d"]["config"]["knots"] == 19


def test_preview_refuses_before_phase_d(client):
    run_id = client.post("/api/runs/sample/ins_clean").json()["run_id"]  # roles not confirmed
    assert client.get(f"/api/runs/{run_id}/baseline-preview").status_code in (409, 503)


def _node_playwright() -> bool:
    if not shutil.which("node"):
        return False
    return subprocess.run(["node", "-e", "require('playwright')"], capture_output=True).returncode == 0


@needs_meridian
@pytest.mark.skipif(not _node_playwright(), reason="needs node with playwright")
def test_slider_in_the_browser(tmp_path):
    """Moving the slider calls no server and redraws within 200 ms; both themes and a 400px screen render."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    runs = tmp_path / "runs"
    env = {**__import__("os").environ, "MMM_RUNS_DIR": str(runs)}
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "mmm_tool.api.app:app", "--port", str(port)],
                              env=env, cwd=Path(__file__).parents[1], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        import httpx

        base = f"http://127.0.0.1:{port}"
        for _ in range(120):
            try:
                httpx.get(base + "/api/samples", timeout=1)
                break
            except httpx.HTTPError:
                time.sleep(0.5)
        with SAMPLE.open("rb") as f:
            run_id = httpx.post(base + "/api/runs", files={"file": (SAMPLE.name, f, "text/csv")}, timeout=120).json()["run_id"]
        httpx.put(f"{base}/api/runs/{run_id}/answers", json={"answers": ANSWERS}, timeout=120)
        out = subprocess.run(["node", str(Path(__file__).parent / "browser" / "step24.js"), base, run_id, str(tmp_path)],
                             capture_output=True, text=True, timeout=300)
        assert out.returncode == 0, out.stderr
        m = json.loads(out.stdout.strip().splitlines()[-1])
        print(f"\nslowest redraw {m['redraw_ms']:.1f} ms; server calls while sliding: {m['server_calls_while_sliding']}")
        assert m["errors"] == []
        assert m["server_calls_while_sliding"] == 0 and m["redraw_ms"] < 200
        assert m["default_line_label"].endswith("default (12)") and m["knot_dots_at_39"] == 39
        assert m["tooltip"].startswith("Knot 4 of 39") and "level" in m["tooltip"]
        assert "likely to absorb campaign effects" in m["alert_at_39"]
        assert "credited to media" in m["trend_tile"]
        assert "#24 knots = 19 (you; default 12)" in m["log"]
        assert m["summary_after_apply"].startswith("19 knots, one about every 9 weeks")
        assert m["bg_light"] != m["bg_dark"]                      # both themes render
        assert m["phone_sideways_scroll"] is False and m["phone_chart_width"] <= 400
    finally:
        server.terminate()
        server.wait(timeout=20)


@needs_meridian
def test_outlier_weeks_are_passed_to_the_chart(tmp_path):
    from mmm_tool.samples import MESSY_TRUTH, make_insurance, write_messy

    r = run_pipeline(write_messy(tmp_path), {"duplicate_dates": "merge"}, accept_defaults=True)
    cvs = {c["channel"]: c["cv"] for s in r.c.steps if s.step == 16 for c in s.details["channels"]}
    flagged = [o["date"] for s in r.b.steps if s.step == 11 for o in s.details["outliers"]]
    p = baseline.preview(r.d, cvs, flagged)
    week = str(make_insurance()["week_start"].iloc[MESSY_TRUTH["outlier_week"]].date())
    assert [p["weeks"][i] for i in p["outliers"]] == [week]
    # The outlier is three times a normal week; the chart's y range is set without it.
    rest = [v for i, v in enumerate(p["kpi"]) if i not in p["outliers"]]
    assert p["kpi"][p["outliers"][0]] > 1.5 * max(rest)
