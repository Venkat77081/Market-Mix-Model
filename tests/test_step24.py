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

import gen
from mmm_tool.configure import baseline
from mmm_tool.pipeline import run_pipeline
from helpers import project_in, run_path  # noqa: F401

SAMPLE = Path(__file__).parent / "data" / "phase_b_tests" / "insurance_mmm_sample.csv"
needs_meridian = pytest.mark.skipif(not baseline.meridian_installed(), reason="needs google-meridian")


def step24(r):
    return next(s for s in r.d.steps if s.step == 24)


@pytest.fixture(scope="module")
def run():
    return run_pipeline(SAMPLE, {}, accept_defaults=True)


@pytest.fixture(scope="module")
def preview(run):
    outliers = [o["date"] for s in run.b.steps if s.step == 11 for o in s.details.get("outliers", [])]
    return baseline.preview(run.d, outliers)


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
    assert sorted(preview["fits"]) == list(range(1, 40)) and preview["ref_k"] == 2
    f12 = preview["fits"][12]
    assert len(f12["baseline"]) == 156 and len(f12["levels"]) == 12
    assert f12["locations"] == [0, 14, 28, 42, 56, 70, 84, 98, 112, 126, 140, 155]
    # The chart's baseline is the knot part only: at a knot week it equals that knot's level.
    assert all(np.isclose(f12["baseline"][t], lv, atol=0.02) for t, lv in zip(f12["locations"], f12["levels"]))
    assert set(preview["strips"]) == set(preview["group"])


@needs_meridian
def test_insurance_sample_values_for_review(preview, tmp_path):
    """No pass/fail on these numbers: printed for review (run with -s)."""
    print(f"\ninsurance_mmm_sample.csv  burst group: {preview['group']}  reference K = {preview['ref_k']}"
          f"  trend t = {preview['trend']['t']:.2f}")
    print(f"{'K':>4} {'card 1: media share of ups and downs':>38} {'card 2: credit kept':>20}  per channel")
    for k in (1, 2, 6, 12, 18, 24, 39):
        f = preview["fits"][k]
        by = ", ".join(f"{c} {v:.0%}" for c, v in f["by_channel"].items() if v is not None)
        print(f"{k:>4} {f['media_ups_downs']:>38.1%} {f['kept']:>14.0%} {f['tone']:>5}  {by}")


# Property tests on synthetic data where the truth is known (tests/gen.py) --------------

SEEDS = (1, 2, 3, 4, 5)


def _case(tmp_path, kind, seed):
    r = run_pipeline(gen.write_knots_case(tmp_path, kind, seed), {}, accept_defaults=True)
    assert r.d is not None and r.d.status == "complete", r.last.steps[-1].summary
    return baseline.preview(r.d)


@pytest.fixture(scope="module")
def cases(tmp_path_factory):
    if not baseline.meridian_installed():
        pytest.skip("needs google-meridian")
    d = tmp_path_factory.mktemp("knots")
    return {(kind, seed): _case(d, kind, seed) for kind in gen.KNOT_CASES for seed in SEEDS}


def test_baseline_mean_matches_kpi_less_controls(cases):
    """At every K, the chart baseline's mean is within 2% of mean KPI minus the controls' fitted mean."""
    for p in cases.values():
        kpi = np.mean(p["kpi"])
        for f in p["fits"].values():
            assert abs(np.mean(f["baseline"]) - (kpi - f["controls_mean"])) <= 0.02 * kpi


def test_flat_truth_with_tv_bursts(cases):
    """Credit kept is green at the default K and falls below 75% by K = weeks/4. The trend card is
    silent at K = 1 on most files: a |t| > 2 rule false-alarms on about 1 in 20 trend-free files."""
    below, silent = 0, 0
    for seed in SEEDS:
        p = cases[("flat_tv", seed)]
        assert p["group"] == ["tv"]
        assert p["fits"][p["default"]]["kept"] >= 0.90 and p["fits"][p["default"]]["tone"] == "good"
        below += min(p["fits"][k]["kept"] for k in range(1, p["max"] + 1)) < 0.75
        silent += not p["trend"]["warn"]
    assert below >= 4 and silent >= 4, (below, silent)


def test_strong_trend_with_tv_bursts(cases):
    for seed in SEEDS:
        p = cases[("trend_tv", seed)]
        assert p["trend"]["warn"]                                       # the trend card fires at K = 1
        assert p["fits"][p["default"]]["tone"] == "good"                # and the default K is green on card 2


def test_only_always_on_channels_is_not_checked(cases):
    for seed in SEEDS:
        p = cases[("always_on", seed)]
        assert p["group"] == [] and p["strips"] == {}
        assert all(f["kept"] is None and f["tone"] is None for f in p["fits"].values())


def test_tiny_noisy_burst_channel_never_joins_the_group(cases):
    for seed in SEEDS:
        p = cases[("tiny_burst", seed)]
        assert "print" not in p["group"] and p["group"] == ["tv"]


def test_burst_group_rule():
    from types import SimpleNamespace

    import pandas as pd

    chans = [SimpleNamespace(name=c, spend=f"{c}_spend") for c in ("tv", "outdoor", "print", "search")]
    t = pd.DataFrame({"tv_spend": [60.0], "outdoor_spend": [10.0], "print_spend": [4.0], "search_spend": [26.0]})
    r = SimpleNamespace(channels=chans, table=t)
    stats = {"tv": {"cv": 1.1, "share": 0.60}, "outdoor": {"cv": 0.5, "share": 0.10},
             "print": {"cv": 1.8, "share": 0.04}, "search": {"cv": 0.2, "share": 0.26}}
    assert baseline.burst_group(r, stats) == ["tv", "outdoor"]   # print is 4% of spend; search is always on


def test_colours():
    assert [baseline.tone(v) for v in (1.04, 0.90, 0.89, 0.75, 0.74, None)] == ["good", "good", "warn", "warn", "fail", None]


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
        run_id = client.post("/api/runs", files={"file": (SAMPLE.name, f, "text/csv")}, data={"project": project_in()}).json()["run_id"]
    client.put(f"/api/runs/{run_id}/answers", json={"answers": ANSWERS})
    return run_id


@needs_meridian
def test_preview_and_suggestion_endpoints(client):
    run_id = _upload(client)
    p = client.get(f"/api/runs/{run_id}/baseline-preview").json()
    assert p["default"] == 12 and p["max"] == 39 and len(p["fits"]) == 39
    s = client.get(f"/api/runs/{run_id}/knots-suggestion").json()
    print(f"\nMeridian AKS suggests {s['knots']} knots; burst credit kept there: {s['fit']['kept']:.0%} ({s['fit']['tone']})")
    assert s["knots"] == len(s["locations"]) and s["knots"] > 1
    assert len(s["fit"]["baseline"]) == 156 and len(s["fit"]["locations"]) == s["knots"]
    r = client.put(f"/api/runs/{run_id}/answers", json={"answers": {**ANSWERS, "knots": 19}}).json()
    assert r["result_d"]["config"]["knots"] == 19


def test_preview_refuses_before_phase_d(client):
    run_id = client.post("/api/runs/sample/ins_clean", params={"project": project_in()}).json()["run_id"]  # roles not confirmed
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
            slug = httpx.post(base + "/api/projects", json={"name": "Browser", "packs": ["india", "insurance"], "calendar": "india_insurance"}, timeout=30).json()["slug"]
            run_id = httpx.post(base + "/api/runs", files={"file": (SAMPLE.name, f, "text/csv")}, data={"project": slug},
                                timeout=120).json()["run_id"]
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
        print("card colours (card 1, 2, 3):", {k: v["classes"] for k, v in m["cards"].items()})
        print("card 2 at 24:", m["cards"]["24"]["card2"][:160])
        print("suggestion:", m["suggestion"][:200], "| preview:", m["suggested_preview"])
        assert all(v["classes"][0].strip() == "kp-tile" for v in m["cards"].values())     # card 1 has no colour
        assert m["cards"]["12"]["classes"][2].strip() == "kp-tile"                        # card 3 not checked at K >= 2
        assert "warn" in m["cards"]["1"]["classes"][2]                                    # the sample's trend fires at K = 1
        assert "Burst channels (" in m["cards"]["24"]["card2"]
        assert "check card 2 before using it" in m["suggestion"]
        sp = m["suggested_preview"]
        assert sp["apply"] == f"Apply {sp['k']} knots" and (sp["beyond"] == (int(sp["k"]) > 39))
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
    flagged = [o["date"] for s in r.b.steps if s.step == 11 for o in s.details["outliers"]]
    p = baseline.preview(r.d, flagged)
    week = str(make_insurance()["week_start"].iloc[MESSY_TRUTH["outlier_week"]].date())
    assert [p["weeks"][i] for i in p["outliers"]] == [week]
    # The outlier is three times a normal week; the chart's y range is set without it.
    rest = [v for i, v in enumerate(p["kpi"]) if i not in p["outliers"]]
    assert p["kpi"][p["outliers"][0]] > 1.5 * max(rest)
