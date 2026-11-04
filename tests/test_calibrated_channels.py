"""Steps 32/33 and the Model Summary for channels with a lift test, and cost per KPI.

Calibrated channels come from the calibration spec Meridian was given, never from names. MTR_FULL
tests run when that file (with raw/lift_tests.csv beside it) is in tests/data and MMM_FIT_TESTS=1.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mmm_tool.fit import model as m
from mmm_tool.fit import steps
from mmm_tool.summary.build import cost_per, with_cost_per


def test_narrowing_text_and_note_share_the_number():
    assert steps.narrowing_text(0.594) == "narrower by 59%" and "by only 59%" in steps.narrowing_note(0.594)
    assert steps.narrowing_text(-0.123) == "wider by 12%" and "12% wider than the prior" in steps.narrowing_note(-0.123)
    assert steps.narrowing_text(-0.004) == "wider by 0%"                     # never clamped to 'narrower by 0%'


def _metrics(rows):
    df = pd.DataFrame(rows).set_index("channel")
    df["paid"] = True
    return df


def _ctx(tmp_path=None, calibrated=("search",)):
    metrics = _metrics([
        # search: lift test; its sampled prior over the whole period, and a posterior 59% narrower
        {"channel": "search", "roi_prior_ci_lo": 2.58, "roi_prior_ci_hi": 4.73, "roi_posterior_ci_lo": 2.46, "roi_posterior_ci_hi": 3.33},
        # tv: on the default prior, well learned
        {"channel": "tv", "roi_prior_ci_lo": 0.3, "roi_prior_ci_hi": 5.0, "roi_posterior_ci_lo": 1.0, "roi_posterior_ci_hi": 1.6},
    ])
    priors = {"search": {"family": "lognormal", "p1": 0.2, "p2": 0.9}, "tv": {"family": "lognormal", "p1": 0.2, "p2": 0.9}}
    ctx = SimpleNamespace(metrics=metrics, config={"prior_type": "roi", "roi_priors": priors}, confidence=0.9,
                          cfg={"prior_driven": {"min_narrowing": 0.30}}, mmm=None, out_dir=tmp_path,
                          badges={"search": ["calibrated", "hard_to_measure"], "tv": ["model_only"]})
    ctx.badge = lambda ch, b: ctx.badges.setdefault(ch, []).append(b) if b not in ctx.badges.get(ch, []) else None
    return ctx, set(calibrated)


def test_step33_measures_a_calibrated_channel_against_its_sampled_prior(monkeypatch):
    ctx, cal = _ctx()
    monkeypatch.setattr(m, "calibrated_channels", lambda mmm: cal)
    s = steps.prior_driven(ctx)
    row = {r["channel"]: r for r in s.details["channels"]}
    assert row["search"]["prior"] == [2.58, 4.73] and row["search"]["prior_source"] == "sampled"
    assert row["search"]["prior_note"] == "(sampled; lift test applies to its own weeks)"
    assert row["search"]["narrowing"] == pytest.approx(1 - (3.33 - 2.46) / (4.73 - 2.58)) and row["search"]["narrowing_text"] == "narrower by 60%"
    assert row["tv"]["prior_source"] == "exact"                               # uncalibrated: the exact lognormal quantiles
    assert "anchored_to_lift_test" in ctx.badges["search"] and "prior_driven" not in ctx.badges["search"]
    assert s.verdict.value == "pass"


def test_a_calibrated_channel_never_counts_as_prior_driven(monkeypatch):
    ctx, cal = _ctx()
    ctx.metrics.loc["search", ["roi_posterior_ci_lo", "roi_posterior_ci_hi"]] = [2.0, 5.0]     # wider than its prior
    monkeypatch.setattr(m, "calibrated_channels", lambda mmm: cal)
    s = steps.prior_driven(ctx)
    row = {r["channel"]: r for r in s.details["channels"]}["search"]
    assert row["narrowing"] < 0 and row["narrowing_text"].startswith("wider by")
    assert s.verdict.value == "pass" and s.details["prior_driven"] == [] and s.warnings == []   # Pass, not Pass with warning
    assert "prior_driven" not in ctx.badges["search"] and "anchored_to_lift_test" in ctx.badges["search"]


def test_an_uncalibrated_low_narrowing_channel_still_warns_with_the_table_number(monkeypatch):
    ctx, _ = _ctx(calibrated=())
    ctx.metrics.loc["tv", ["roi_posterior_ci_lo", "roi_posterior_ci_hi"]] = [0.2, 5.5]
    monkeypatch.setattr(m, "calibrated_channels", lambda mmm: set())
    s = steps.prior_driven(ctx)
    tv = {r["channel"]: r for r in s.details["channels"]}["tv"]
    assert s.verdict.value == "warn" and "tv" in s.details["prior_driven"]
    assert tv["narrowing_text"].startswith("wider by") and tv["narrowing_text"].split()[-1] in s.warnings[[i for i, w in enumerate(s.warnings) if "'tv'" in w][0]]


def _fitted_run(folder: Path, run_id: str, ranges: dict, finished: float):
    d = folder / run_id
    (d / "fit").mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"file": "x.csv", "answers": {}}))
    (d / "fit" / "status.json").write_text(json.dumps({"state": "done", "finished": finished, "heartbeat": finished}))
    (d / "fit" / "result.json").write_text(json.dumps({"channels": [{"channel": c, "paid": True, "roi_posterior_ci_lo": lo,
                                                                     "roi_posterior_ci_hi": hi} for c, (lo, hi) in ranges.items()]}))
    return d


def test_hard_to_measure_is_cleared_when_the_lift_test_narrowed_the_range(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    _fitted_run(runs, "aaaaaaaaaaa1", {"search": (1.0, 6.0)}, 100.0)          # run 1: no lift test, wide
    now = runs / "bbbbbbbbbbb2"
    (now / "fit").mkdir(parents=True)
    (now / "meta.json").write_text(json.dumps({"file": "x.csv", "answers": {}}))
    ctx, cal = _ctx(tmp_path=now / "fit")
    monkeypatch.setattr(m, "calibrated_channels", lambda mmm: cal)
    s = steps.prior_driven(ctx)
    assert "hard_to_measure" not in ctx.badges["search"]
    assert s.details["hard_to_measure_cleared"][0]["previous_run"] == "aaaaaaaaaaa1"
    # a range wider than the previous run's keeps the badge
    ctx2, _ = _ctx(tmp_path=now / "fit")
    ctx2.metrics.loc["search", ["roi_posterior_ci_lo", "roi_posterior_ci_hi"]] = [0.5, 7.0]
    steps.prior_driven(ctx2)
    assert "hard_to_measure" in ctx2.badges["search"]


def test_calibrated_channels_come_from_the_calibration_spec():
    pytest.importorskip("meridian")
    from meridian.model import spec as spec_mod

    data = SimpleNamespace(media_channel=SimpleNamespace(values=np.array(["tv", "search", "meta"])), rf_channel=None)
    one = spec_mod.CalibrationSpec([spec_mod.ChannelCalibrationSpec(channels=["search"], date_ranges=[spec_mod.DateRange("2024-01-01", "2024-02-01")])])
    mmm = SimpleNamespace(model_spec=SimpleNamespace(roi_calibration=one, roi_calibration_period=None, rf_roi_calibration=None,
                                                     rf_roi_calibration_period=None), input_data=data)
    assert m.calibrated_channels(mmm) == {"search"}
    period = np.zeros((10, 3), bool)
    period[2:5, 2] = True
    mmm.model_spec.roi_calibration, mmm.model_spec.roi_calibration_period = None, period
    assert m.calibrated_channels(mmm) == {"meta"}
    mmm.model_spec.roi_calibration_period = None
    assert m.calibrated_channels(mmm) == set()


# Cost per KPI -------------------------------------------------------------------------------------

def test_cost_per_is_spend_over_mean_kpi_and_its_range_flips():
    c = cost_per(1000.0, {"mean": 50.0, "lo": 40.0, "hi": 80.0})
    assert c == {"mean": 20.0, "lo": 12.5, "hi": 25.0}
    assert cost_per(1000.0, {"mean": 0.0, "lo": -1.0, "hi": 2.0})["mean"] is None


def test_every_summary_row_cost_per_times_kpi_driven_is_its_spend():
    rows = [{"channel": "tv", "spend": 8.0e6, "kpi_driven": {"mean": 704.0, "lo": 600.0, "hi": 810.0}, "cost_per_kpi": {"mean": 99.0}},
            {"channel": "search", "spend": 3.1e6, "kpi_driven": {"mean": 601.6, "lo": 500.0, "hi": 700.0}, "cost_per_kpi": {"mean": 1.0}},
            {"channel": "all paid", "spend": 11.1e6, "kpi_driven": {"mean": 1305.6, "lo": 1150.0, "hi": 1460.0}, "cost_per_kpi": {"mean": 3.0}}]
    S = with_cost_per({"roi": {"all": {"rows": rows}, "last_52": None}})
    for r in S["roi"]["all"]["rows"]:
        assert r["cost_per_kpi"]["mean"] * r["kpi_driven"]["mean"] == pytest.approx(r["spend"], rel=0.005)
        assert r["cost_per_kpi"]["lo"] <= r["cost_per_kpi"]["mean"] <= r["cost_per_kpi"]["hi"]


@pytest.mark.skipif(os.environ.get("MMM_FIT_TESTS") != "1", reason="set MMM_FIT_TESTS=1 to run Meridian fits")
def test_a_real_summary_cost_per_matches_its_own_row(tmp_path):
    from test_phase_g import TINY, shop

    from mmm_tool.pipeline import run_pipeline
    from mmm_tool.summary.build import build_summary  # noqa: F401
    from mmm_tool.summary.build import roi_table

    p = tmp_path / "shop.csv"
    shop().to_csv(p, index=False)
    r = run_pipeline(p, {"revenue_per_kpi_value": 40}, accept_defaults=True)
    mmm = m.build(r.d, {**r.d.config, "holdout": None})
    mmm.sample_prior(50, seed=3)
    m.sample_posterior(mmm, TINY, 3)
    for row in roi_table(m.analyzer(mmm), True, None)["rows"]:
        assert row["cost_per_kpi"]["mean"] * row["kpi_driven"]["mean"] == pytest.approx(row["spend"], rel=0.005), row["channel"]


# MTR_FULL, run 2 (a lift test on search) ------------------------------------------------------------

from test_mtr_full import LIFT  # noqa: E402
from test_mtr_full import RAW as MTR_FULL  # noqa: E402

needs_mtr = pytest.mark.skipif(not MTR_FULL.exists() or os.environ.get("MMM_FIT_TESTS") != "1",
                               reason="MTR_FULL is in tests/data/mtr_full; set MMM_FIT_TESTS=1 to fit it")


@needs_mtr
def test_mtr_full_run2_search_is_anchored_to_its_lift_test():
    from test_mtr_full import run

    from mmm_tool.fit.pipeline import run_phase_e

    r = run({"lift_tests": [LIFT]})
    e = run_phase_e(r.d, {"sampler": {"n_chains": 4, "n_adapt": 500, "n_burnin": 500, "n_keep": 1000}, "seed": 42})
    st = {s.step: s for s in e.steps}
    rows = {x["channel"]: x for x in st[33].details["channels"]}
    print("\n" + "\n".join(f"{c}: prior {x['prior'][0]:.2f}-{x['prior'][1]:.2f} posterior {x['posterior'][0]:.2f}-{x['posterior'][1]:.2f} "
                           f"{x['narrowing_text']}" for c, x in rows.items()))
    s = rows["search"]
    assert s["calibrated"] and 2.3 < s["prior"][0] < 2.9 and 4.3 < s["prior"][1] < 5.2 and 0.5 < s["narrowing"] < 0.68
    assert st[33].verdict.value == "pass"
    badges = next(c for c in e.channels if c["channel"] == "search")["badges"]
    assert "anchored to lift test" in badges and "calibrated" in badges and "prior-driven" not in badges
    assert 0.72 < st[32].details["contributions"]["baseline_share"] < 0.77
