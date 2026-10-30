"""Steps 51-52: both limits per channel, Lock, lowest / highest allowed with the active limit,
floors that aren't needed, floors above planned refused, and the past-peak warning."""
from __future__ import annotations

import pytest

from mmm_tool.plan.optimize import PlanError, bounds, check, defaults, display_name

CR = 1e7
TV = 37.18 * CR


def test_floor_above_the_percent_limit_is_the_lowest():
    b = bounds(TV, down=30, up=50, locked=False, floor=30 * CR)
    print(f"\nlowest {b['lowest'] / CR:.2f} Cr ({b['source']}), from -30% {b['from_percent'] / CR:.2f} Cr")
    assert round(b["lowest"] / CR, 2) == 30.00 and b["source"] == "floor" and not b["floor_not_needed"]
    assert round(b["highest"] / CR, 2) == 55.77


def test_a_floor_below_the_percent_limit_isnt_needed():
    b = bounds(TV, down=30, up=50, locked=False, floor=20 * CR)
    assert round(b["lowest"] / CR, 2) == 26.03 and b["source"] == "−30%" and b["floor_not_needed"]


def test_lock_makes_lowest_and_highest_the_planned_spend():
    b = bounds(TV, down=30, up=50, locked=True, floor=30 * CR)
    assert b["lowest"] == b["highest"] == TV and b["source"] == "locked"


HIST = {"tv": TV, "search": 20 * CR, "radio": 5 * CR}
PERIODS = {"all": ("2023-01-02", "2025-12-22")}


def plan(**kw):
    return {**defaults(list(HIST), money=True), **kw}


def test_the_ratios_passed_come_from_the_effective_bounds():
    c = check(plan(floors={"tv": 30 * CR}, locks=["radio"]), HIST, True, PERIODS)
    a, rows = c["args"], {r["channel"]: r for r in c["channels"]}
    print(f"\nlower {a['spend_constraint_lower']}\nupper {a['spend_constraint_upper']}")
    assert a["spend_constraint_lower"] == [pytest.approx(1 - 30 / 37.18), 0.3, 0.0]
    assert a["spend_constraint_upper"] == [0.5, 0.5, 0.0]
    assert rows["tv"]["lowest_source"] == "floor" and rows["radio"]["lowest_source"] == "locked"
    assert rows["search"]["lowest_source"] == "−30%" and rows["search"]["highest"] == pytest.approx(30 * CR)


def test_defaults_are_minus_30_plus_50():
    d = defaults(["tv"], money=True)
    assert d["limits"]["tv"] == {"down": 30, "up": 50}


def test_a_floor_above_planned_is_refused():
    with pytest.raises(PlanError, match="A floor can only limit cuts. To raise a channel, change its planned spend or its 'may go up' limit."):
        check(plan(floors={"tv": 40 * CR}), HIST, True, PERIODS)


def test_a_floor_below_the_limit_is_shown_as_not_needed():
    rows = {r["channel"]: r for r in check(plan(floors={"tv": 20 * CR}), HIST, True, PERIODS)["channels"]}
    assert rows["tv"]["floor_not_needed"] and round(rows["tv"]["lowest"] / CR, 2) == 26.03


def test_above_twice_the_past_peak_is_warned():
    peaks = {"tv": 25 * CR, "search": 20 * CR, "radio": 5 * CR}       # tv's highest allowed, 55.77 Cr, is > 2 x 25 Cr
    c = check(plan(), HIST, True, PERIODS, peaks)
    assert [r["channel"] for r in c["channels"] if r["beyond_seen"]] == ["tv"]
    assert len(c["warnings"]) == 1 and "outside what the model has seen" in c["warnings"][0]


def test_grid_step_matches_meridian():
    get_round_factor = pytest.importorskip("meridian.analysis.optimizer").get_round_factor

    from mmm_tool.plan.optimize import grid_step

    for budget in (5e5, 6.3e7, 2.4e9, 1234.0):
        assert grid_step(budget, 1e-4) == 10 ** -get_round_factor(budget, 1e-4)


def test_display_names():
    assert [display_name(x) for x in ("tv", "ctv", "google_ads", "tv_video", "ooh", "aggregator")] == \
        ["TV", "CTV", "Google Ads", "TV Video", "OOH", "Aggregator"]
