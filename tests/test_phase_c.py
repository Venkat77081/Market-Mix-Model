"""Phase C tests: each sample plants one problem a profile step must catch."""
from __future__ import annotations

import numpy as np
import pytest

from mmm_tool import samples
from mmm_tool.pipeline import run_pipeline
from mmm_tool.profile.context import PARAMS_PER_MEDIA


@pytest.fixture(scope="module")
def d(tmp_path_factory):
    return tmp_path_factory.mktemp("c")


def run(path, answers=None, defaults=True):
    return run_pipeline(path, answers or {}, accept_defaults=defaults)


def step(r, n):
    return next(s for s in r.c.steps if s.step == n)


def decisions(r):
    return {x.id: x for s in r.c.steps for x in s.decisions}


def test_clean_file_passes_with_a_priors_warning(d):
    r = run(samples.write_clean(d))
    assert r.c.status == "complete"
    s14 = step(r, 14)
    assert s14.details["weeks"] == 156 and s14.details["parameters"]["media channels"] == 9 * PARAMS_PER_MEDIA
    assert 3 <= s14.details["obs_per_parameter"] < 6 and "strong priors" in s14.warnings[0]
    assert all(step(r, n).verdict.value == "pass" for n in (15, 16, 17, 18, 19, 20))
    assert r.c.badges == {}


def test_short_history_is_refused(d):
    r = run(samples.write_short(d))
    assert r.status == "failed" and r.c.steps[-1].step == 14
    assert "Refused" in r.c.steps[-1].summary and "90 weeks" in r.c.steps[-1].summary


def test_thin_history_warns(d, tmp_path):
    from mmm_tool.samples import _weekly_csv, make_insurance

    df = make_insurance(weeks=120)[["week_start", "new_policies", "tv_spend", "search_spend", "tax_season_flag"]]
    r = run(_weekly_csv(df, tmp_path / "thin.csv"))
    s14 = step(r, 14)
    assert s14.verdict.value == "warn" and "Thin history: 120 weeks" in s14.warnings[0]


def test_regions_multiply_observations(d):
    r = run(samples.write_regions(d))
    s14 = step(r, 14)
    assert s14.details["observations"] == 780 and s14.details["obs_per_parameter"] > 6


def test_rare_channels_merge_into_other(d):
    p = samples.write_rare(d)
    r = run(p, defaults=False, answers={"roles": {}})
    dec = r.c.pending[0]
    assert dec["id"] == "rare_channels" and set(dec["default"]) == {"print", "radio"}
    r = run(p)
    names = [c.name for c in r.c.channels]
    assert "other" in names and "print" not in names and "radio" not in names
    raw = samples.make_insurance()
    other = next(c for c in r.c.channels if c.name == "other")
    assert other.exposure_is_spend and np.isclose(r.c.table[other.spend].sum(),
                                                   r.b.table["print_spend"].sum() + r.b.table["radio_spend"].sum())
    assert r.c.roles["print_spend"] == "excluded" and other.spend in r.c.coord_to_columns()["coord_to_columns"]["media_spend"]
    assert len(raw) == len(r.c.table)


def test_rare_channels_drop_or_keep(d):
    p = samples.write_rare(d)
    r = run(p, {"rare_channels": {"print": "drop", "radio": "keep"}})
    names = [c.name for c in r.c.channels]
    assert "print" not in names and "radio" in names and "rarely_on" in r.c.badges["radio"]


def test_flat_spend_is_hard_to_measure(d):
    r = run(samples.write_flat(d))
    assert set(step(r, 16).details["hard_to_measure"]) == {"search", "aggregator"}
    assert r.c.badges["search"] == ["hard_to_measure"]
    assert r.c.to_dict()["badges"]["aggregator"] == ["hard to measure"]


def test_collinear_channels_combine_or_keep(d):
    p = samples.write_collinear(d)
    r = run(p, defaults=False, answers={"roles": {}})
    assert r.c.pending[0]["id"] == "together:tv|youtube"
    r = run(p)
    names = [c.name for c in r.c.channels]
    assert "tv_youtube" in names and "tv" not in names and "youtube" not in names
    r = run(p, {"together:tv|youtube": "keep"})
    assert r.c.badges["tv"] == ["moves_with"] and r.c.badges["youtube"] == ["moves_with"]


def test_mediator_is_excluded_after_confirmation(d):
    p = samples.write_mediator(d)
    r = run(p, defaults=False, answers={"roles": {}})
    dec = r.c.pending[0]
    assert dec["id"] == "mediator:search_interest_index" and dec["context"]["channel"] == "tv" and dec["context"]["lag"] == 1
    r = run(p)
    assert r.c.roles["search_interest_index"] == "excluded" and "search_interest_index" not in r.c.table.columns
    r = run(p, {"mediator:search_interest_index": "keep"})
    assert "search_interest_index" in r.c.table.columns and step(r, 18).warnings


def test_season_timed_spend_warns(d):
    r = run(samples.write_season(d))
    s19 = step(r, 19)
    assert [x["channel"] for x in s19.details["pairs"]] == ["outdoor"] and r.c.badges["outdoor"] == ["season_timed"]


def test_profile_report(d):
    r = run(samples.write_flat(d))
    rep = step(r, 20).details
    assert abs(sum(x["share"] for x in rep["spend_share"]) - 1) < 0.01
    assert len(rep["weeks"]) == len(rep["kpi"]) == len(rep["total_spend"]) == 156
    assert set(rep["heatmap"]) == {c.name for c in r.c.channels}
    assert max(max(v) for v in rep["heatmap"].values()) == 1.0
    assert next(x for x in rep["spend_share"] if x["channel"] == "search")["badges"] == ["hard to measure"]


def test_a_lone_rare_channel_is_not_merged_alone(d, tmp_path):
    from mmm_tool.samples import _weekly_csv, make_insurance

    df = make_insurance()
    df["radio_spend"] = (df["radio_spend"] / 200).round()  # under 1% of spend, the only rare channel
    r = run(_weekly_csv(df, tmp_path / "one_rare.csv"), defaults=False, answers={"roles": {}})
    assert r.c.pending[0]["default"] == {"radio": "drop"}
    r = run(_weekly_csv(df, tmp_path / "one_rare.csv"), {"rare_channels": {"radio": "other"}})
    assert any("just renames it" in w for w in step(r, 15).warnings)


def test_season_check_uses_the_calendar_step_21_adds(tmp_path):
    from mmm_tool.samples import _weekly_csv, make_insurance

    df = make_insurance().drop(columns=["tax_season_flag"])  # no season column in the data
    df["outdoor_spend"] = np.where(df["week_start"].dt.month.isin([1, 2, 3]), 2.5e6, 0.0)
    p = _weekly_csv(df, tmp_path / "no_flags.csv")
    r = run(p)
    s19 = step(r, 19)
    assert "cal_tax_season" in s19.details["calendar_flags"]
    pair = next(x for x in s19.details["pairs"] if x["channel"] == "outdoor")
    assert pair["flag"] == "cal_tax_season" and pair["source"].startswith("calendar")
    assert r.c.badges["outdoor"] == ["season_timed"]
    # The flag step 21 then adds is the one step 19 checked.
    assert "cal_tax_season" in r.d.config["calendar"]["added"]

    # Switched off at step 21, it is no longer a season to compete with.
    r = run(p, {"calendar_events": ["fy_end", "diwali_festive"]})
    assert "cal_tax_season" not in step(r, 19).details["calendar_flags"]
    assert "outdoor" not in r.c.badges


def test_calendar_event_already_in_the_data_is_checked_once(d):
    r = run(samples.write_season(d))
    s19 = step(r, 19)
    assert "tax_season_flag" in s19.details["flags"] and "cal_tax_season" not in s19.details["calendar_flags"]
