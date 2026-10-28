"""Step 34: every channel row gets a status note (stable and supported, stable near the starting
assumption, or unstable with its ROI range and, where it applies, why its evidence is thin)."""
from __future__ import annotations

import itertools

import pandas as pd
import pytest
from test_phase_e_checks import _stability_ctx, mix, run, write

from mmm_tool.channel_stats import channel_stats
from mmm_tool.fit import steps as fsteps
from mmm_tool.fit.steps import stability_note

SUPPORTED = "Stable, and the data supports it"
NEAR = "Stable because it stays near the starting assumption"
ROIS = [1.02, 0.65, 1.53, 1.10]


@pytest.mark.parametrize("cv, badges, active, status, note, extra", [
    (0.1, [], 0.9, "supported", SUPPORTED, []),
    (0.1, ["hard_to_measure"], 0.9, "near_assumption", NEAR, ["hard_to_measure"]),
    (0.1, ["prior_driven", "calibrated"], 0.9, "near_assumption", NEAR, ["prior_driven"]),
    (0.1, ["season_timed"], 0.1, "supported", SUPPORTED, []),                 # stable: the activity reason is for unstable only
    (0.35, [], 0.9, "unstable", "Unstable: ROI ranges 0.65–1.53 across windows", []),
    (0.30, [], 0.9, "unstable", "Unstable: ROI ranges 0.65–1.53 across windows", []),   # at the threshold
    (0.35, ["prior_driven"], 0.9, "unstable", "Unstable: ROI ranges 0.65–1.53 across windows", []),
    (0.35, [], 0.22, "unstable", "Unstable: ROI ranges 0.65–1.53 across windows; few active weeks, "
                                  "so removing 26 weeks removes much of its evidence", ["active in 22% of weeks"]),
    (0.35, ["season_timed"], 0.9, "unstable", "Unstable: ROI ranges 0.65–1.53 across windows; mostly in one season, "
                                              "so removing 26 weeks removes much of its evidence", ["timed with a season"]),
    (0.35, ["season_timed"], 0.2, "unstable", "Unstable: ROI ranges 0.65–1.53 across windows; few active weeks and "
                                              "mostly in one season, so removing 26 weeks removes much of its evidence",
     ["active in 20% of weeks", "timed with a season"]),
    (float("nan"), [], None, "unstable", "Unstable: ROI ranges 0.65–1.53 across windows", []),
])
def test_note_rules(cv, badges, active, status, note, extra):
    n = stability_note(cv, 0.3, ROIS, badges, active, 0.3)
    print(f"\ncv {cv}, badges {badges}, active {active} -> {n['status']}: {n['note']}")
    assert n["status"] == status and n["note"] == note
    assert (n["note_badges"] if status == "near_assumption" else n.get("evidence", [])) == extra


def test_note_is_never_empty():
    for cv, badges, active in itertools.product([0.0, 0.29, 0.3, 5.0, float("nan"), float("inf")],
                                                [[], ["hard_to_measure"], ["season_timed"]], [None, 0.0, 0.5]):
        for rois in (ROIS, [], [float("nan")]):
            assert stability_note(cv, 0.3, rois, badges, active, 0.3)["note"]


def test_active_share_follows_the_current_table(tmp_path):
    r = run(write(tmp_path, mix(rare=False)))
    st = channel_stats(r.d)
    for c in r.d.channels:
        assert st[c.name]["active"] == pytest.approx(float((r.d.table.groupby(r.d.coord_to_columns()["coord_to_columns"]["time"])[c.spend].sum() > 0).mean()))


def test_step34_rows_unstable_with_few_active_weeks(tmp_path, monkeypatch):
    df = mix(rare=False)
    df["display_spend"] = [2e5 if i % 5 == 0 else 0.0 for i in range(len(df))]     # active in 20% of weeks
    pair = next((d.id for s in run(write(tmp_path, df)).c.steps for d in s.decisions if d.id.startswith("together:")), None)
    r = run(write(tmp_path, df), {pair: "keep"} if pair else {})
    names = [c.name for c in r.d.channels]
    promo = next(c.name for c in r.d.channels if "display_spend" in c.sources())
    steady = next(n for n in names if n != promo)
    windows = iter([0.65, 1.53, 1.10])                                              # promo's ROI in the three refits
    monkeypatch.setattr(fsteps.m, "window_variant", lambda d, table: (d, []))
    monkeypatch.setattr(fsteps.m, "window_config", lambda cfg, wk: cfg)
    monkeypatch.setattr(fsteps.m, "build", lambda d, cfg: type("M", (), {"sample_prior": lambda *a, **k: None})())
    monkeypatch.setattr(fsteps, "_fit", lambda *a, **k: 1.0)
    monkeypatch.setattr(fsteps.m, "channel_metrics",
                        lambda mmm, conf: pd.DataFrame({"roi_posterior_mean": [next(windows) if n == promo else 1.5 for n in names]}, index=names))
    ctx = _stability_ctx(r, {steady: ["prior_driven"]})
    ctx.metrics.loc[promo, "roi_posterior_mean"] = 1.02
    s = fsteps.stability(ctx)
    rows = {c["channel"]: c for c in s.details["channels"]}
    for n, c in rows.items():
        print(f"\n{n:14} CV {c['cv']:.2f} {c['status']:16} {c['note']}")
    assert rows[promo]["status"] == "unstable" and rows[promo]["active_share"] == pytest.approx(0.2, abs=0.01)
    assert rows[promo]["note"] == ("Unstable: ROI ranges 0.65–1.53 across windows; few active weeks, "
                                   "so removing 26 weeks removes much of its evidence")
    assert rows[steady]["note"] == NEAR
    assert all(c["note"] for c in rows.values()) and s.details["few_active_share"] == 0.3
    assert {n for n, c in rows.items() if c["status"] == "supported"} == set(names) - {promo, steady}
