"""Steps 33-35, the answer key in step 32, and per-channel stats after step 15/17.

Synthetic files and fake fits: the rules are checked without Meridian fits
except where marked. INS_FULL tests run when its files are in tests/data/ins_full/
(the raw file, answers.json from its README decision script, truth_for_tool.json).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mmm_tool.channel_stats import channel_stats
from mmm_tool.fit import gates, truth
from mmm_tool.fit import steps as fsteps
from mmm_tool.fit.context import FitOptions, settings
from mmm_tool.ingest.pairing import Channel
from mmm_tool.ingest.types import Verdict
from mmm_tool.pipeline import run_pipeline

try:
    import meridian  # noqa: F401
    HAVE_MERIDIAN = True
except ImportError:
    HAVE_MERIDIAN = False
needs_meridian = pytest.mark.skipif(not HAVE_MERIDIAN, reason="needs google-meridian")


def mix(weeks=156, seed=5, bursty=True, rare=True):
    """Orders from TV (in flights), video (moves with TV), steady search, and a rarely-on radio."""
    rng = np.random.default_rng(seed)
    week = pd.date_range("2022-01-03", periods=weeks, freq="7D")
    on = np.zeros(weeks)
    i = 0
    while i < weeks:
        n = int(rng.integers(3, 7))
        on[i:i + n] = 1
        i += n + int(rng.integers(4, 9))
    df = pd.DataFrame({"week": week.strftime("%Y-%m-%d")})
    if bursty:
        df["tv_spend"] = np.round(on * rng.normal(4e5, 4e4, weeks), -2)
        df["video_spend"] = np.round(on * rng.normal(1e5, 1e4, weeks) + rng.normal(0, 2e3, weeks).clip(0), -2)
    else:
        df["tv_spend"] = np.round(rng.normal(4e5, 3e4, weeks), -2)
        df["video_spend"] = np.round(rng.normal(1e5, 9e3, weeks), -2)
    df["search_spend"] = np.round(rng.normal(2e5, 1.5e4, weeks), -2)
    if rare:
        df["radio_spend"] = np.where(rng.random(weeks) < 0.05, 5000.0, 0.0)
    df["price_index"] = np.round(1 + rng.normal(0, 0.02, weeks), 3)
    df["orders"] = np.round(3000 + 0.002 * df["tv_spend"] + 0.004 * df["search_spend"] + rng.normal(0, 60, weeks))
    return df


def write(tmp_path, df, name="mix.csv"):
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


def run(path, answers=None):
    r = run_pipeline(path, answers or {}, accept_defaults=True)
    assert r.d is not None and r.d.status == "complete", r.last.steps[-1].summary
    return r


# Per-channel stats after step 15 drops and step 17 combinations ------------------------------------------

def test_stats_follow_the_current_channels(tmp_path):
    raw = mix()
    r = run(write(tmp_path, raw), {"rare_channels": {"radio": "drop"}})
    names = [c.name for c in r.d.channels]
    combined = next(c for c in r.d.channels if len(c.sources()) == 2)
    stats = channel_stats(r.d)
    print(f"\nchannels {names}; combined {combined.name} from {combined.sources()}: "
          f"CV {stats[combined.name]['cv']:.2f}, {stats[combined.name]['share']:.1%} of spend")
    assert sorted(combined.sources()) == ["tv_spend", "video_spend"]
    assert "radio" not in names and not any("radio" in c for ch in r.d.channels for c in ch.sources())
    both = raw["tv_spend"] + raw["video_spend"]
    assert stats[combined.name]["cv"] == pytest.approx(both.std() / both.mean(), rel=1e-6)
    total = raw[["tv_spend", "video_spend", "search_spend"]].to_numpy().sum()
    assert stats[combined.name]["share"] == pytest.approx(both.sum() / total, rel=1e-6)


@needs_meridian
def test_step24_card2_checks_the_combined_channel_and_never_the_dropped_one(tmp_path):
    from mmm_tool.configure import baseline

    r = run(write(tmp_path, mix()), {"rare_channels": {"radio": "drop"}})
    p = baseline.preview(r.d)
    combined = next(c.name for c in r.d.channels if len(c.sources()) == 2)
    print(f"\nburst group {p['group']}; stats {p['channel_stats']}")
    assert p["group"] == [combined]                     # the combined channel qualifies; search is steady
    assert "radio" not in p["channel_stats"] and "tv" not in p["channel_stats"]


@needs_meridian
def test_step24_without_combining_checks_both_parts(tmp_path):
    from mmm_tool.configure import baseline

    df = mix()
    r0 = run(write(tmp_path, df), {"rare_channels": {"radio": "drop"}})
    pair = next(d.id for s in r0.c.steps for d in s.decisions if d.id.startswith("together:"))
    r = run(write(tmp_path, df), {"rare_channels": {"radio": "drop"}, pair: "keep"})
    p = baseline.preview(r.d)
    assert sorted(p["group"]) == ["tv", "video"]
    assert p["channel_stats"]["tv"]["cv"] == pytest.approx(df["tv_spend"].std() / df["tv_spend"].mean(), abs=0.0005)


def test_missing_channel_spend_is_an_error_not_not_checked(tmp_path):
    from mmm_tool.channel_stats import ChannelStatsError

    r = run(write(tmp_path, mix(rare=False)))
    broken = SimpleNamespace(table=r.d.table.drop(columns=["search_spend"]), channels=r.d.channels,
                             coord_to_columns=r.d.coord_to_columns)
    with pytest.raises(ChannelStatsError, match="search"):
        channel_stats(broken)


def test_combined_channel_gets_its_own_variation_badge(tmp_path):
    r = run(write(tmp_path, mix(bursty=False, rare=False)))     # steady spend: CVs under 0.3
    s17 = next(s for s in r.c.steps if s.step == 17)
    combined = next((c.name for c in r.d.channels if len(c.sources()) == 2), None)
    if combined is None:
        pytest.skip("the steady channels didn't move together in this draw")
    assert "hard_to_measure" in r.d.badges.get(combined, []) and any("(combined)" in w for w in s17.warnings)


# Step 33: exact prior range --------------------------------------------------------------------------------

def test_step33_prior_range_is_exact_and_the_same_for_every_channel(tmp_path):
    r = run(write(tmp_path, mix(rare=False)), {"revenue_per_kpi_value": 50})       # ROI priors (money per money)
    assert r.d.config["prior_type"] == "roi"
    names = [c.name for c in r.d.channels]
    noisy = pd.DataFrame({"paid": True, "roi_prior_ci_lo": np.linspace(0.25, 0.31, len(names)),       # draws differ
                          "roi_prior_ci_hi": np.linspace(5.1, 5.6, len(names)),
                          "roi_posterior_ci_lo": 0.8, "roi_posterior_ci_hi": 2.0}, index=names)
    ctx = SimpleNamespace(metrics=noisy, config=r.d.config, cfg=settings(), confidence=0.9, badges={},
                          badge=lambda c, b: None)
    s = fsteps.prior_driven(ctx)
    shown = {(round(c["prior"][0], 2), round(c["prior"][1], 2)) for c in s.details["channels"]}
    print(f"\nprior 90% ranges shown: {shown}")
    assert shown == {(0.28, 5.37)} and all(c["prior_source"] == "exact" for c in s.details["channels"])
    lo, hi = math.exp(0.2 - 1.6448536 * 0.9), math.exp(0.2 + 1.6448536 * 0.9)
    c = s.details["channels"][0]
    assert c["prior"] == pytest.approx([lo, hi], rel=1e-6)
    assert c["narrowing"] == pytest.approx(1 - (2.0 - 0.8) / (hi - lo), rel=1e-6)


def test_exact_prior_range_formulas():
    assert gates.exact_prior_range({"family": "lognormal", "p1": 0.2, "p2": 0.9}, 0.9) == pytest.approx((0.27793, 5.36753), rel=1e-4)
    lo, hi = gates.exact_prior_range({"family": "beta", "p1": 2, "p2": 8}, 0.9)
    from scipy import stats
    assert (lo, hi) == pytest.approx(tuple(stats.beta(2, 8).ppf([0.05, 0.95])))
    assert gates.exact_prior_range({"family": "custom"}, 0.9) is None and gates.exact_prior_range(None, 0.9) is None


# Step 34: stable is not confirmed; skipped ---------------------------------------------------------------------

def _stability_ctx(r, badges):
    names = [c.name for c in r.d.channels]
    metrics = pd.DataFrame({"paid": True, "roi_posterior_mean": 1.5}, index=names)
    return SimpleNamespace(options=FitOptions(stability=True), cfg=settings(), d=r.d, metrics=metrics, config=r.d.config,
                           badges=badges, badge=lambda c, b: badges.setdefault(c, []).append(b),
                           progress=lambda *a: None, seed=1, fit_sampler={}, confidence=0.9, timings={})


def test_step34_stable_but_prior_driven_gets_the_note(tmp_path, monkeypatch):
    df = mix(rare=False)
    pair = next(d.id for s in run(write(tmp_path, df)).c.steps for d in s.decisions if d.id.startswith("together:"))
    r = run(write(tmp_path, df), {pair: "keep"})                                   # three channels
    names = [c.name for c in r.d.channels]
    monkeypatch.setattr(fsteps.m, "window_variant", lambda d, table: (d, []))
    monkeypatch.setattr(fsteps.m, "window_config", lambda cfg, wk: cfg)
    monkeypatch.setattr(fsteps.m, "build", lambda d, cfg: SimpleNamespace(sample_prior=lambda *a, **k: None))
    monkeypatch.setattr(fsteps, "_fit", lambda *a, **k: 1.0)
    monkeypatch.setattr(fsteps.m, "channel_metrics", lambda mmm, conf: pd.DataFrame({"roi_posterior_mean": 1.5}, index=names))
    ctx = _stability_ctx(r, {names[0]: ["prior_driven"], names[1]: ["hard_to_measure"]})
    s = fsteps.stability(ctx)
    notes = {c["channel"]: c.get("note") for c in s.details["channels"]}
    print(f"\nstep 34: {s.summary}")
    near = "Stable because it stays near the starting assumption"
    assert notes[names[0]] == near and notes[names[1]] == near
    assert notes[names[2]] == "Stable, and the data supports it" and "because they stay near the starting assumption" in s.summary


def test_stability_window_leaves_out_a_control_that_is_flat_inside_it(tmp_path):
    from mmm_tool.fit import model as m

    r = run(write(tmp_path, mix(rare=False)))
    t = r.d.table.copy()
    t["event_x"] = 0.0
    t.loc[t.index[-3:], "event_x"] = 1.0                      # on only in the last weeks
    d = m.variant(r.d, table=t, roles={**r.d.roles, "event_x": "control"})
    part, flat = m.window_variant(d, t.iloc[:-26].reset_index(drop=True))
    assert flat == ["event_x"] and "event_x" not in part.table.columns and part.roles["event_x"] == "excluded"
    part, flat = m.window_variant(d, t.iloc[26:].reset_index(drop=True))
    assert flat == [] and "event_x" in part.table.columns


def test_steps34_and_35_not_run_are_skipped_not_pass():
    ctx = SimpleNamespace(options=FitOptions(stability=False, placebo=False))
    s34, s35 = fsteps.stability(ctx), fsteps.placebo(ctx)
    assert s34.verdict == Verdict.SKIPPED and s35.verdict == Verdict.SKIPPED
    assert s34.to_dict()["verdict"] == "skipped" and not s34.blocks


# Step 35: placebo choice and rule ------------------------------------------------------------------------------

def test_placebo_shaped_like_the_biggest_bursty_channel(tmp_path):
    df = mix(rare=False)
    pair = None
    r0 = run(write(tmp_path, df))
    pair = next(d.id for s in r0.c.steps for d in s.decisions if d.id.startswith("together:"))
    r = run(write(tmp_path, df), {pair: "keep"})
    pick = fsteps.placebo_source(r.d, 0.5)
    assert pick["channel"] == "tv" and "biggest spender" in pick["why"]


def test_placebo_with_no_bursty_channel_uses_the_most_variable(tmp_path):
    r = run(write(tmp_path, mix(bursty=False, rare=False)))
    stats = channel_stats(r.d)
    assert all(v["cv"] < 0.5 for v in stats.values())
    pick = fsteps.placebo_source(r.d, 0.5)
    print(f"\nno CV >= 0.5: {pick}")
    assert pick["channel"] == max(stats, key=lambda c: stats[c]["cv"])
    assert pick["why"].startswith("no channel has spend CV 0.5 or more")


@pytest.mark.parametrize("placebo, low, real, ok", [(0.3, 0.1, 1.4, True), (0.8, 0.1, 1.4, False),
                                                     (0.3, 0.35, 1.4, False), (0.69, 0.29, 1.4, True)])
def test_placebo_rule(placebo, low, real, ok):
    assert gates.placebo_passes(placebo, low, real, 0.5, 0.3)["ok"] is ok


# Step 32: the answer key ---------------------------------------------------------------------------------------

def _key(kpi="orders", **extra):
    return {"kpi_column": kpi, "assumes_decisions": "Combine TV and video at step 17.",
            "channels": [{"source_columns": ["tv_spend", "video_spend"], "true_roi": 1.4, "true_share_of_kpi_pct": 12.0,
                          "true_policies_per_lakh": 3.0, "true_marginal_roi": 0.9},
                         {"source_columns": ["search_spend"], "true_roi": 2.5, "true_share_of_kpi_pct": 20.0,
                          "true_policies_per_lakh": 5.0, "true_marginal_roi": 1.8}],
            "true_non_media_share_pct": 68.0, **extra}


def _result(names, money=True):
    chans = [{"channel": n, "paid": True, "roi_posterior_mean": m, "roi_posterior_ci_lo": lo, "roi_posterior_ci_hi": hi,
              "share_of_kpi": sh} for n, (m, lo, hi, sh) in zip(names, [(1.5, 1.1, 1.9, 0.10), (1.2, 0.9, 1.6, 0.25)])]
    return {"channels": chans, "steps": [{"step": 32, "details": {"roi_in_money": money}}]}


def test_truth_matches_by_raw_columns_never_by_name():
    channels = [Channel("whatever_name", "media", "x_spend", exposure="x_spend", exposure_is_spend=True,
                        spend_sources=["tv_spend", "video_spend"]),
                Channel("search", "media", "search_spend", exposure="search_clicks")]
    t = truth.compare(_result(["whatever_name", "search"]), channels, _key(), {"kpi": "orders"})
    rows = {r["channel"]: r for r in t["rows"]}
    assert rows["whatever_name"]["matched"] and rows["whatever_name"]["true"] == 1.4 and rows["whatever_name"]["inside"]
    assert rows["search"]["matched"] and rows["search"]["inside"] is False          # 2.5 is outside 0.9-1.6
    assert (t["inside"], t["compared"]) == (1, 2)
    assert t["non_media"]["model_share"] == pytest.approx(0.65) and t["non_media"]["true_share"] == pytest.approx(0.68)
    assert t["mean_abs_error"] == pytest.approx(((1.5 - 1.4) / 1.4 + (2.5 - 1.2) / 2.5) / 2)
    assert t["true_marginal_roi"] == {"whatever_name": 0.9, "search": 1.8}


def test_unmatched_channels_have_no_truth():
    channels = [Channel("tv", "media", "tv_spend"), Channel("video", "media", "video_spend")]
    t = truth.compare(_result(["tv", "video"]), channels, _key(), {"kpi": "orders"})
    assert [r["matched"] for r in t["rows"]] == [False, False] and t["compared"] == 0


def test_kpi_units_compare_per_lakh():
    channels = [Channel("c", "media", "s", spend_sources=["tv_spend", "video_spend"])]
    res = _result(["c"], money=False)
    res["channels"][0].update(roi_posterior_mean=2.8e-5, roi_posterior_ci_lo=2.0e-5, roi_posterior_ci_hi=3.5e-5)
    t = truth.compare(res, channels, _key(), {"kpi": "orders", "currency": {"code": "INR", "symbol": "₹", "grouping": "indian", "unit": "rupee"}})
    r = t["rows"][0]
    assert t["measure"] == "per_lakh" and t["label"] == "orders per ₹1.0 L"
    assert r["model"] == pytest.approx(2.8) and r["true"] == 3.0 and r["inside"]


def test_answer_key_for_another_kpi_is_refused():
    with pytest.raises(truth.TruthError, match="KPI column 'sales'"):
        truth.validate(_key(kpi="sales"), "orders")


def test_answer_key_never_changes_a_verdict():
    res = _result(["c", "search"])
    before = json.dumps(res, sort_keys=True)
    truth.compare(res, [Channel("c", "media", "s", spend_sources=["tv_spend", "video_spend"])], _key(), {"kpi": "orders"})
    assert json.dumps(res, sort_keys=True) == before


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from mmm_tool.api import app as app_mod

    monkeypatch.setattr(app_mod, "RUNS_DIR", tmp_path / "runs")
    return TestClient(app_mod.app)


def test_answer_key_api(client, tmp_path):
    p = write(tmp_path, mix(rare=False))
    with p.open("rb") as f:
        run_id = client.post("/api/runs", files={"file": ("mix.csv", f, "text/csv")}).json()["run_id"]
    r = client.put(f"/api/runs/{run_id}/answers", json={"answers": {"roles": {}}}).json()
    assert r["answer_key"] == {"loaded": False, "file": None} and "truth" not in (r["fit"] or {})     # nothing changes
    bad = client.post(f"/api/runs/{run_id}/answer-key", files={"file": ("k.json", json.dumps(_key(kpi="sales")), "application/json")})
    assert bad.status_code == 400 and "KPI column 'sales'" in bad.json()["detail"]
    ok = client.post(f"/api/runs/{run_id}/answer-key", files={"file": ("truth.json", json.dumps(_key()), "application/json")})
    assert ok.status_code == 200 and ok.json()["answer_key"] == {"loaded": True, "file": "truth.json"}
    assert client.delete(f"/api/runs/{run_id}/answer-key").json()["answer_key"]["loaded"] is False


# INS_FULL --------------------------------------------------------------------------------------------------------

INS = Path(__file__).parent / "data" / "ins_full"
RAW = next(iter(sorted(INS.glob("*.csv")) + sorted(INS.glob("*.xlsx"))), None) if INS.is_dir() else None
needs_ins = pytest.mark.skipif(RAW is None or not (INS / "truth_for_tool.json").exists() or not (INS / "answers.json").exists(),
                               reason="put INS_FULL's raw file, answers.json (README decision script) and "
                                      "truth_for_tool.json in tests/data/ins_full/")


@pytest.fixture(scope="module")
def ins():
    answers = json.loads((INS / "answers.json").read_text(encoding="utf-8"))
    key = json.loads((INS / "truth_for_tool.json").read_text(encoding="utf-8"))
    return run_pipeline(RAW, answers, accept_defaults=True), key, answers


@needs_ins
@needs_meridian
def test_ins_full_step24_card2(ins):
    from mmm_tool.configure import baseline

    r, key, _ = ins
    p = baseline.preview(r.d)
    combined = next(c.name for c in r.d.channels if sorted(c.sources()) == sorted(["TV Spend (INR)", "YouTube Spend"]))
    st = p["channel_stats"]
    print(f"\ncard 2 group {p['group']}; {combined} CV {st[combined]['cv']:.2f}, {st[combined]['share']:.1%}")
    assert combined in p["group"]
    assert st[combined]["cv"] == pytest.approx(0.94, abs=0.005) and st[combined]["share"] == pytest.approx(0.464, abs=0.0005)
    radio = next(c for c, v in st.items() if any("radio" in s.lower() for s in v["sources"]))
    assert radio not in p["group"] and st[radio]["cv"] == pytest.approx(1.93, abs=0.005)
    assert st[radio]["share"] == pytest.approx(0.031, abs=0.0005)


@needs_ins
def test_ins_full_answer_key_matching(ins):
    r, key, _ = ins
    found = truth.match(r.d.channels, key)
    matched = {c: e for c, e in found.items() if e}
    combined = next(c.name for c in r.d.channels if sorted(c.sources()) == sorted(["TV Spend (INR)", "YouTube Spend"]))
    print(f"\nmatched {len(matched)}: {[(c, e['source_columns']) for c, e in matched.items()]}")
    assert len(matched) == 6 and sorted(matched[combined]["source_columns"]) == ["TV Spend (INR)", "YouTube Spend"]
    truth.validate(key, r.a.kpi["kpi"])
    with pytest.raises(truth.TruthError):
        truth.validate({**key, "kpi_column": "something else"}, r.a.kpi["kpi"])


@needs_ins
@needs_meridian
@pytest.mark.skipif(__import__("os").environ.get("MMM_FIT_TESTS") != "1", reason="set MMM_FIT_TESTS=1 to run Meridian fits")
def test_ins_full_step34_notes(ins, tmp_path):
    from mmm_tool.fit import run_phase_e

    r, _, answers = ins
    e = run_phase_e(r.d, {"sampler": answers.get("sampler", "quick"), "seed": answers.get("seed"), "stability": True},
                    out_dir=tmp_path / "fit")
    rows = {c["channel"]: c for c in next(s for s in e.steps if s.step == 34).details["channels"]}
    for ch, c in rows.items():
        print(f"{ch:16} CV {c['cv']:.2f}  {c['status']:16} {c['note']}")
    assert all(c["note"] for c in rows.values())
    for ch in ("meta", "tv_you_tube"):
        assert rows[ch]["note"] == "Stable, and the data supports it"
    assert rows["radio"]["status"] == "unstable" and rows["radio"]["note"].startswith("Unstable: ROI ranges 0.65–1.53 across windows")
    for ch in ("affiliate", "google_search", "policy_bazaar"):
        assert rows[ch]["note"] == "Stable because it stays near the starting assumption"
