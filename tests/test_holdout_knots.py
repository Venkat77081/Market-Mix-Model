"""Step 29: the holdout fit's knots sit inside the training weeks; step 31 says when a one-sided
test error points at a guessed baseline. Any file: weeks, K and the holdout come from the run.

MTR_FULL tests run when that file is in tests/data (its README gives the reference numbers) and
MMM_FIT_TESTS=1.
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mmm_tool.configure.spec import holdout_knots
from mmm_tool.fit import gates

WEEKS157 = [str(d.date()) for d in pd.date_range("2023-01-02", periods=157, freq="W-MON")]


def _cfg(weeks, k=12, held=13, knots_in="training"):
    return {"knots": k, "holdout": {"weeks": held, "start": weeks[-held], "end": weeks[-1], "train_weeks": len(weeks) - held,
                                    "knots_in": knots_in}}


def test_knots_inside_the_training_weeks():
    knots, info = holdout_knots(_cfg(WEEKS157), WEEKS157)
    assert knots == [0, 13, 26, 39, 52, 65, 78, 91, 104, 117, 130, 143]                 # MTR_FULL's shape
    assert info["train_weeks"] == 144 and info["held_weeks"] == 13 and info["last_training_index"] == 143
    assert not info["capped"] and not info["merged"]


def test_meridian_default_no_holdout_and_the_final_fit_are_unchanged():
    assert holdout_knots(_cfg(WEEKS157, knots_in="all"), WEEKS157) == (12, None)
    assert holdout_knots({"knots": 12, "holdout": None}, WEEKS157) == (12, None)          # step 36 / no holdout
    pytest.importorskip("meridian")
    from mmm_tool.configure.baseline import knot_info

    locs = [int(x) for x in knot_info(157, 12, 3).knot_locations]                         # the final fit: Meridian's own
    assert locs == [0, 14, 28, 42, 56, 70, 85, 99, 113, 127, 141, 156]


def test_more_knots_than_training_weeks_are_capped_and_said():
    weeks = WEEKS157[:20]
    knots, info = holdout_knots(_cfg(weeks, k=15, held=8), weeks)
    assert knots == list(range(12)) and info["capped"] and info["knots"] == 12
    from mmm_tool.fit.steps import knot_placement

    p = knot_placement(_cfg(weeks, k=15, held=8), weeks)
    assert "more than the 12 training weeks" in p["cap_warning"]


def test_step29_card_text():
    from mmm_tool.fit.steps import knot_placement

    p = knot_placement(_cfg(WEEKS157), WEEKS157)
    assert p["text"] == ("Knots: 12, placed in the 144 training weeks (weeks 0–143); the baseline is held flat "
                         "over the 13 held-back weeks.")
    pytest.importorskip("meridian")
    q = knot_placement(_cfg(WEEKS157, knots_in="all"), WEEKS157)
    assert q["placement"] == "all" and q["knots_in_held_weeks"] == 1 and "1 of them sit in the 13 held-back weeks" in q["text"]


def test_test_bias():
    a = np.full(13, 100.0)
    one = gates.test_bias(a * 1.28, a)
    assert one["mean"] == pytest.approx(0.28) and one["share_same_sign"] == 1.0 and one["one_sided"]
    both = gates.test_bias(a * np.array([1.2, 0.8] * 6 + [1.2]), a)
    assert not both["one_sided"]
    small = gates.test_bias(a * 1.03, a)
    assert not small["one_sided"]                                                          # under 5%


def _step31(monkeypatch, predicted_factor, knots_in="training"):
    from mmm_tool.fit import model as m
    from mmm_tool.fit import steps

    weeks = WEEKS157
    actual = 1000 + 3 * np.arange(157) + 50 * np.sin(np.arange(157) / 8)       # a trend: last year is a poor guess
    fitted = actual.copy()
    fitted[-13:] *= predicted_factor
    test = slice(144, 157)
    mape = float(np.mean(np.abs(fitted[test] / actual[test] - 1)))
    monkeypatch.setattr(m, "accuracy", lambda mmm: {"Train": {"MAPE": 0.01, "R_Squared": 0.9}, "Test": {"MAPE": mape, "R_Squared": 0.1}})
    monkeypatch.setattr(m, "weeks", lambda mmm: weeks)
    monkeypatch.setattr(m, "actual_kpi", lambda mmm: actual)
    monkeypatch.setattr(m, "fitted_kpi", lambda mmm: fitted)
    ctx = SimpleNamespace(config=_cfg(weeks, knots_in=knots_in), mmm=None, gates={},
                          cfg={"holdout": {"test_mape_max": 0.10, "season_weeks": 52}})
    return steps.holdout_accuracy(ctx)


def test_step31_one_sided_error_points_at_the_baseline(monkeypatch):
    s = _step31(monkeypatch, 1.28, knots_in="all")
    assert s.verdict.value == "fail" and s.details["test_bias"]["mean"] == pytest.approx(0.28)
    assert s.warnings == ["Predictions are consistently above actual in the held-back weeks (+28%). This usually means "
                          "the baseline was guessed for those weeks (knots in the held-back period), not that media is "
                          "wrong. Fix: place knots inside the training weeks (step 29)."]
    assert s.details["flat_baseline_note"] is None                                       # knots spread over all weeks


def test_step31_errors_both_ways_keep_the_old_message(monkeypatch):
    s = _step31(monkeypatch, np.array([1.3, 0.7] * 6 + [1.3]))
    assert s.verdict.value == "fail" and s.details["diagnosis"] == "both_ways"
    assert s.warnings == ["A model that can't predict weeks it hasn't seen can't be trusted to split sales between channels."]


def test_step31_pass_shows_the_flat_baseline_line(monkeypatch):
    s = _step31(monkeypatch, 1.01)
    assert s.verdict.value == "pass" and s.warnings == []
    assert s.details["flat_baseline_note"] == ("Test weeks use a flat baseline from the last training knot, so the test "
                                               "checks media and controls, not a guessed trend.")


def test_step27_offers_the_choice(tmp_path):
    from mmm_tool import samples
    from mmm_tool.pipeline import run_pipeline

    p = samples.write_clean(tmp_path)
    r = run_pipeline(p, {"holdout_knots": "all"}, accept_defaults=True)
    s27 = next(s for s in r.d.steps if s.step == 27)
    assert [d.id for d in s27.decisions] == ["holdout_weeks", "holdout_knots"]
    assert r.d.config["holdout"]["knots_in"] == "all" and "knots spread over all weeks" in s27.summary and s27.warnings
    default = run_pipeline(p, {}, accept_defaults=True)
    assert default.d.config["holdout"]["knots_in"] == "training" and "knots inside the training weeks" in next(
        s for s in default.d.steps if s.step == 27).summary


# MTR_FULL (real fits) ---------------------------------------------------------------------------------

MTR_FULL = next((p for p in Path(__file__).parent.joinpath("data").rglob("*") if p.stem.lower() == "mtr_full"), None)
needs_mtr = pytest.mark.skipif(MTR_FULL is None or os.environ.get("MMM_FIT_TESTS") != "1",
                               reason="put MTR_FULL (mtr_full.csv or .xlsx) in tests/data and set MMM_FIT_TESTS=1")


def _mtr(knots_in):
    from mmm_tool.fit.pipeline import run_phase_e
    from mmm_tool.pipeline import run_pipeline

    r = run_pipeline(MTR_FULL, {"holdout_knots": knots_in, "duplicate_dates": "merge"}, accept_defaults=True)
    e = run_phase_e(r.d, {"sampler": "quick"})
    return r, {s.step: s for s in e.steps}


@needs_mtr
def test_mtr_full_knots_inside_training_weeks():
    r, st = _mtr("training")
    assert r.d.config["knots"] == 12 and len(r.d.weeks) == 157 and r.d.config["holdout"]["weeks"] == 13
    assert st[29].details["knots"]["locations"] == [0, 13, 26, 39, 52, 65, 78, 91, 104, 117, 130, 143]
    mape = st[31].details["accuracy"]["Test"]["MAPE"]
    print(f"\nMTR_FULL test MAPE, knots inside training weeks: {mape:.1%} (reference about 1.3%)")
    assert mape < 0.03


@needs_mtr
def test_mtr_full_knots_spread_over_all_weeks():
    r, st = _mtr("all")
    mape, bias = st[31].details["accuracy"]["Test"]["MAPE"], st[31].details["test_bias"]["mean"]
    print(f"\nMTR_FULL test MAPE, knots over all weeks: {mape:.1%} (reference about 28%); bias {bias:+.0%}")
    assert mape > 0.15 and st[31].details["diagnosis"] == "baseline_guess"
    assert "Predictions are consistently above actual" in st[31].warnings[0]
