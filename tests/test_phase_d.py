"""Phase D tests: defaults, every setting the user can change, and the Meridian bridge."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from mmm_tool import samples
from mmm_tool.configure.calendar import week_flags
from mmm_tool.configure.context import lognormal_from_range, lognormal_summary
from mmm_tool.pipeline import run_pipeline


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    return samples.write_clean(tmp_path_factory.mktemp("d"))


def run(path, answers=None, defaults=True):
    return run_pipeline(path, answers or {}, accept_defaults=defaults)


def step(r, n):
    return next(s for s in r.d.steps if s.step == n)


@pytest.fixture(scope="module")
def default_run(clean):
    return run(clean)


def test_defaults_complete_without_questions(default_run):
    r = default_run
    assert r.phase == "D" and r.status == "complete"
    cfg = r.d.config
    assert cfg["max_lag"] == 8 and cfg["knots"] == 12
    assert set(cfg["adstock"].values()) == {"geometric"}
    assert all(p["source"] == "default" and p["mu"] == 0.2 and p["sigma"] == 0.9 for p in cfg["roi_priors"].values())
    assert cfg["holdout"] == {"weeks": 13, "start": "2025-09-29", "end": "2025-12-22", "train_weeks": 143}
    assert all(b == ["model_only"] for b in r.d.badges.values())


def test_calendar_flags(default_run):
    r = default_run
    events = {e["event"]: e for e in r.d.config["calendar"]["events"]}
    assert events["tax_season"]["status"] == "already in your data as 'tax_season_flag'"
    assert set(r.d.config["calendar"]["added"]) == {"cal_fy_end", "cal_diwali_festive"}
    t = r.d.table.set_index("week_start")
    assert t.loc["2024-03-25", "cal_fy_end"] == 1 and t.loc["2024-04-08", "cal_fy_end"] == 0
    # Diwali 2024 is on 1 November: that week and the two before are flagged.
    assert t.loc["2024-10-28", "cal_diwali_festive"] == 1 and t.loc["2024-10-14", "cal_diwali_festive"] == 1
    assert t.loc["2024-11-04", "cal_diwali_festive"] == 0
    assert r.d.roles["cal_fy_end"] == "control"
    assert "cal_fy_end" in r.d.coord_to_columns()["coord_to_columns"]["controls"]


def test_week_flags_touching_any_day():
    weeks = pd.date_range("2024-03-11", periods=4, freq="W-MON")  # 11, 18, 25 Mar, 1 Apr
    assert week_flags(weeks, {"recurring": {"from": "03-18", "to": "03-31"}}).tolist() == [0, 1, 1, 0]


def test_events_off_and_custom_events(clean):
    r = run(clean, {"calendar_events": ["fy_end"],
                    "custom_events": [{"name": "Price Change", "ranges": [{"start": "2024-06-03", "end": "2024-06-30"}]}]})
    added = r.d.config["calendar"]["added"]
    assert added == ["cal_fy_end", "cal_price_change"]
    assert r.d.table["cal_price_change"].sum() == 4


def test_bad_custom_event_blocks(clean):
    r = run(clean, {"custom_events": [{"name": "x", "ranges": [{"start": "2024-06-30", "end": "2024-06-01"}]}]})
    assert r.status == "needs_input" and r.d.steps[-1].step == 21


def test_settings_can_change(clean):
    r = run(clean, {"max_lag": 4, "knots": 30, "holdout_weeks": 20, "adstock_shape": {"tv": "binomial"}})
    cfg = r.d.config
    assert cfg["max_lag"] == 4 and cfg["knots"] == 30 and cfg["holdout"]["weeks"] == 20
    assert cfg["adstock"]["tv"] == "binomial" and cfg["adstock"]["search"] == "geometric"
    assert step(r, 24).warnings  # 30 knots is very flexible


@pytest.mark.parametrize("answers,step_no", [({"max_lag": 20}, 22), ({"holdout_weeks": 5}, 27),
                                             ({"adstock_shape": {"tv": "square"}}, 23), ({"knots": 0}, 24)])
def test_out_of_range_settings_block(clean, answers, step_no):
    r = run(clean, answers)
    assert r.status == "needs_input" and r.d.steps[-1].step == step_no


def test_lognormal_from_a_90_percent_range():
    mu, sigma = lognormal_from_range(1.0, 4.0)
    s = lognormal_summary(mu, sigma)
    assert math.isclose(s["low"], 1.0, rel_tol=1e-3) and math.isclose(s["high"], 4.0, rel_tol=1e-3)
    assert math.isclose(s["median"], 2.0, rel_tol=1e-3)


def test_benchmark_ranges(clean):
    r = run(clean, {"roi_ranges": {"search": {"low": 1.5, "high": 4.5}}})
    p = r.d.config["roi_priors"]
    assert p["search"]["source"] == "benchmark" and math.isclose(p["search"]["low"], 1.5, rel_tol=1e-3)
    assert p["tv"]["source"] == "default"


def test_lift_tests_calibrate_and_combine(clean):
    tests = [
        {"channel": "search", "start": "2024-04-01", "end": "2024-05-26", "roi": 2.9, "low": 2.2, "high": 3.8},
        {"channel": "search", "start": "2025-02-03", "end": "2025-03-30", "roi": 2.5, "low": 1.5, "high": 4.2},
    ]
    r = run(clean, {"roi_ranges": {"search": {"low": 1, "high": 2}}, "lift_tests": tests})
    p = r.d.config["roi_priors"]["search"]
    assert p["source"] == "lift_test" and 2.5 < p["median"] < 2.9  # pulled towards the more precise test
    assert r.d.config["calibration"]["search"]["periods"] == [{"start": "2024-04-01", "end": "2024-05-26"},
                                                              {"start": "2025-02-03", "end": "2025-03-30"}]
    assert r.d.badges["search"] == ["calibrated"] and r.d.badges["tv"] == ["model_only"]
    assert any("replaces" in w for w in step(r, 26).warnings)


@pytest.mark.parametrize("test", [
    {"channel": "cinema", "start": "2024-04-01", "end": "2024-05-26", "roi": 2, "low": 1, "high": 3},
    {"channel": "search", "start": "2030-01-01", "end": "2030-02-01", "roi": 2, "low": 1, "high": 3},
    {"channel": "search", "start": "2024-04-01", "end": "2024-05-26", "roi": 5, "low": 1, "high": 3},
    {"channel": "search", "start": "2024-04-01"},
])
def test_bad_lift_tests_block(clean, test):
    r = run(clean, {"lift_tests": [test]})
    assert r.status == "needs_input" and r.d.steps[-1].step == 26


def test_hard_to_measure_channels_are_pointed_to_lift_tests(tmp_path):
    r = run(samples.write_flat(tmp_path))
    assert "lift test" in step(r, 25).warnings[0]


# Meridian bridge: runs only where google-meridian is installed.

def test_builds_meridian_objects(clean):
    pytest.importorskip("meridian")
    from mmm_tool.configure.spec import build_input_data, build_model_spec

    tests = [{"channel": "search", "start": "2024-04-01", "end": "2024-05-26", "roi": 2.9, "low": 2.2, "high": 3.8}]
    r = run(clean, {"lift_tests": tests, "adstock_shape": {"tv": "binomial"}})
    data = build_input_data(r.d)
    assert sorted(map(str, data.media_channel.values)) == sorted(c.name for c in r.d.channels)
    assert data.kpi.shape[-1] == 156
    spec = build_model_spec(r.d.config, data)
    assert spec.max_lag == 8 and spec.knots == 12 and spec.adstock_decay_spec["tv"] == "binomial"
    order = list(map(str, data.media_channel.values))
    mu = np.asarray(spec.prior.roi_m.loc if hasattr(spec.prior.roi_m, "loc") else spec.prior.roi_m.parameters["loc"])
    assert np.isclose(mu[order.index("search")], math.log(2.9), atol=0.05)
    assert spec.roi_calibration is not None and spec.holdout is not None
