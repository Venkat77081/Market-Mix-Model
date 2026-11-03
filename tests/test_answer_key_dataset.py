"""The answer-key dataset (examples/answer_key_check/) and the bugs it found.

The committed files must be what make_dataset.py writes, and Phases A-D must read the file as
CHECKLIST.md says: these are the "expected" column of the checklist, checked without Meridian.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mmm_tool.ingest.dates import explaining_region
from mmm_tool.ingest.roles import suggest_role
from mmm_tool.pipeline import run_pipeline

HERE = Path(__file__).resolve().parents[1] / "examples" / "answer_key_check"


def _make():
    import importlib.util

    spec = importlib.util.spec_from_file_location("make_dataset", HERE / "make_dataset.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.make


def test_the_committed_files_are_what_the_generator_writes(tmp_path):
    data, key, details = _make()(11)
    data.to_csv(tmp_path / "d.csv", index=False)
    assert (tmp_path / "d.csv").read_text() == (HERE / "home_goods_mmm.csv").read_text()
    assert key == json.loads((HERE / "truth_for_tool.json").read_text())
    saved = json.loads((HERE / "truth_details.json").read_text())
    assert details["planted_problems"] == saved["planted_problems"] and details["what_if"] == saved["what_if"]


def test_the_actuals_and_the_true_execution_gap():
    import importlib.util

    spec = importlib.util.spec_from_file_location("make_dataset", HERE / "make_dataset.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fut, truth = mod.future_actuals(11)
    assert fut.to_csv(index=False) == (HERE / "home_goods_actuals_next13.csv").read_text()
    assert truth == json.loads((HERE / "truth_details.json").read_text())["future_actuals"]
    assert truth["true_execution_gap_total"] < 0                                  # TV at half costs orders
    # --check-plan agrees with the What-If table
    assert mod.true_change({"search": 1.2})["orders_change"] == json.loads((HERE / "truth_details.json").read_text())["what_if"]["search"]["120%"]["all"]["orders_change"]


def test_phases_a_to_d_read_the_file_as_the_checklist_says():
    r = run_pipeline(HERE / "home_goods_mmm.csv", {}, accept_defaults=True)
    assert r.d is not None and r.d.status == "complete"
    step = {s.step: s for ph in (r.a, r.b, r.c, r.d) for s in ph.steps}
    assert "156 dates × 3 values of 'Region'" in step[2].summary                     # the duplicate row doesn't hide the regions
    roles = step[3].decisions[0].answer
    assert roles["Competitor Promo"] == "control" and roles["Population"] == "population"
    assert roles["Revenue ($)"] == "revenue_for_kpi" and roles["Orders"] == "kpi"
    assert sorted(c.name for c in r.d.channels) == ["display", "search", "social", "tv"]
    assert "469 formatted numbers read (e.g. '$15,928.84' in Search Spend ($))" in step[7].summary
    assert "1 exact copy row(s) dropped; 469 rows → 468" in step[8].summary
    assert "1 empty cell(s) labelled and filled" in step[10].summary
    assert "516 zeros kept as real zeros in media columns" in step[10].summary
    assert "search" in step[16].summary                                                # seasonal but steady: hard to measure
    assert "Last 13 weeks held out" in step[27].summary


def test_exact_duplicate_rows_dont_hide_a_region_column():
    df = pd.DataFrame({"week": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-08", "2024-01-08", "2024-01-08"]),
                       "region": ["N", "S", "N", "S", "S"], "kpi": [1, 2, 3, 4, 4]})
    assert explaining_region(df, "week") == "region"                                 # the last row is an exact copy
    df.loc[4, "kpi"] = 5                                                              # not a copy: two different S rows
    assert explaining_region(df, "week") is None


def test_a_competitors_promotion_is_an_outside_factor():
    s = pd.Series([0, 1, 0, 0, 1] * 10)
    assert suggest_role("Competitor Promo", s, "week").role == "control"
    assert suggest_role("Promo Flag", s, "week").role == "non_media_treatment"


def test_a_flights_off_weeks_dont_reset_the_spend_off_plan_run():
    from mmm_tool.future import tracking

    plan = [f"2026-01-{d:02d}" for d in (5, 12, 19, 26)]
    tg = {w: {"week": w, "spend": {"tv": p, "search": 50.0}, "total": None, "plan_id": "A"} for w, p in zip(plan, (100.0, 100.0, 0.0, 0.0))}
    on = {"weeks": plan, "kpi": [1.0] * 4, "spend": {"tv": [50.0, 50.0, 0.0, 0.0], "search": [50.0] * 4}, "units": {}}
    at = {"weeks": plan, "per_week": {"total": None}, "by_channel": {}}
    sets = {"spend_tolerance_pct": 10, "spend_off_weeks": 2, "outside_range_weeks": 3, "cumulative_gap_pct": 5, "execution_share": 0.5}
    r = tracking.compare(on, tg, at, sets, None)
    assert next(a for a in r["alerts"] if a["kind"] == "spend")["channels"] == ["tv"]       # TV at half, then off as planned
