"""Step 10: real zeros count only media columns and only the export's own zeros; every media
column gets a row; zero runs in an always-on channel ask "real pause or data gap"."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mmm_tool.pipeline import run_pipeline

WEEKS = pd.date_range("2023-01-02", periods=156, freq="7D")
GOOGLE_RUNS = [("2024-04-29", "2024-06-17"), ("2025-03-31", "2025-05-19")]      # 8 + 8 weeks
EXPECTED = {"tv_spend": 66, "ctv_spend": 36, "google_ads_spend": 16, "influencer_spend": 127, "print_spend": 146}


@pytest.fixture(autouse=True)
def generic(monkeypatch, tmp_path):
    monkeypatch.setenv("MMM_KEYWORD_PACKS", "")
    monkeypatch.setenv("MMM_CALENDARS", str(tmp_path / "no_calendars.yaml"))


def _zeros_at(n: int, rng, block: int = 1) -> np.ndarray:
    """A mask of n weeks, in blocks of the given length where it fits."""
    m = np.zeros(len(WEEKS), bool)
    starts = rng.permutation(np.arange(0, len(WEEKS) - block, block))
    for s in starts:
        if m.sum() >= n:
            break
        take = min(block, n - int(m.sum()))
        m[s:s + take] = True
    return m


def shop_with_zeros(seed: int = 7) -> pd.DataFrame:
    """Five paid channels with known real zeros, plus columns whose zeros must not count:
    a 0/1 flag, a discount % and a price index; influencer starts 10 weeks in (blanks before)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"week": WEEKS.strftime("%Y-%m-%d")})
    for col, n, block in (("tv_spend", 66, 6), ("ctv_spend", 36, 3), ("print_spend", 146, 1)):
        on = ~_zeros_at(n, rng, block)
        df[col] = np.where(on, np.round(rng.normal(8e4, 1e4, len(WEEKS)), -2), 0.0)
    g = np.round(rng.normal(5e4, 4e3, len(WEEKS)), -2)
    for a, b in GOOGLE_RUNS:
        g[(WEEKS >= a) & (WEEKS <= b)] = 0.0
    df["google_ads_spend"] = g
    inf = np.round(rng.normal(3e4, 3e3, len(WEEKS)), -2)
    inf[:10] = np.nan                                             # not started: filled, never a real zero
    later = np.zeros(len(WEEKS), bool)                             # 127 zeros after the first 10 weeks
    later[10 + np.random.default_rng(seed + 1).choice(len(WEEKS) - 10, 127, replace=False)] = True
    inf[later] = 0.0
    df["influencer_spend"] = inf
    df["holiday_flag"] = ((WEEKS.month == 12) & (WEEKS.day <= 24)).astype(int)        # mostly 0: not media
    df["discount_pct"] = np.where(rng.random(len(WEEKS)) < 0.6, 0.0, 10.0)            # mostly 0: not media
    df["price_index"] = np.round(1 + rng.normal(0, 0.02, len(WEEKS)), 3)
    df["orders"] = np.round(4000 + 0.01 * df[["tv_spend", "google_ads_spend"]].sum(axis=1) + rng.normal(0, 80, len(WEEKS)))
    return df


def step10(tmp_path, df, answers=None):
    p = tmp_path / "shop.csv"
    df.to_csv(p, index=False)
    r = run_pipeline(p, {"roles": {"discount_pct": "non_media_treatment"}, **(answers or {})}, accept_defaults=True)
    assert r.b is not None, r.last.steps[-1].summary
    return r, next(s for s in r.b.steps if s.step == 10)


def test_real_zeros_count_only_the_exports_zeros_in_media_columns(tmp_path):
    df = shop_with_zeros()
    assert {c: int((df[c] == 0).sum()) for c in EXPECTED} == EXPECTED
    _, s = step10(tmp_path, df)
    real = s.details["real_zeros"]
    print(f"\n{s.summary}\nreal zeros {real}")
    assert real == EXPECTED and sum(real.values()) == 391
    assert "391 zeros kept as real zeros in media columns" in s.summary
    rows = {c["column"]: c for c in s.details["columns"]}
    for col in ("holiday_flag", "discount_pct", "price_index", "orders"):
        assert rows[col]["zeros"] is None and not rows[col]["media"]           # not counted
    assert rows["influencer_spend"]["not_started"] == 10 and rows["influencer_spend"]["zeros"] == 127   # filled cells excluded


def test_every_media_column_gets_a_row(tmp_path):
    _, s = step10(tmp_path, shop_with_zeros())
    media = [c["column"] for c in s.details["columns"] if c["media"]]
    assert sorted(media) == sorted(EXPECTED)                  # including columns with no blanks
    no_blanks = [c for c in s.details["columns"] if c["media"] and not (c["not_started"] or c["ended"] or c["missing"])]
    assert len(no_blanks) == 4 and all(c["fill"] == "none needed" for c in no_blanks)


def test_zero_runs_in_an_always_on_channel_are_flagged_and_kept_by_default(tmp_path):
    r, s = step10(tmp_path, shop_with_zeros())
    note = ("google: 16 zero weeks in 2 runs (2024-04-29 → 06-17, 2025-03-31 → 05-19) "
            "in a channel that otherwise spends every week")
    print(f"\n{s.warnings}")
    assert note in s.warnings
    d = next(x for x in s.decisions if x.id == "zero_runs")
    assert d.default == {"google": "pause"} and d.options == ["pause", "gap"]
    assert list(s.details["zero_runs"]) == ["google"]       # tv, ctv, print, influencer aren't always on
    assert s.details["real_zeros"]["google_ads_spend"] == 16
    assert (r.b.table["google_ads_spend"] == 0).sum() == 16
    log = next(e for e in r.last.decision_log if e["decision"] == "zero_runs")
    assert log["lines"] == ["google: 16 zero weeks kept as a real pause"]


def test_a_data_gap_is_interpolated_and_logged(tmp_path):
    r, s = step10(tmp_path, shop_with_zeros(), {"zero_runs": {"google": "gap"}})
    t = r.b.table
    runs = np.zeros(len(t), bool)
    for a, b in GOOGLE_RUNS:
        runs |= ((t["week"] >= a) & (t["week"] <= b)).to_numpy()
    assert (t.loc[runs, "google_ads_spend"] > 0).all()
    assert s.details["real_zeros"]["google_ads_spend"] == 0 and sum(s.details["real_zeros"].values()) == 375
    row = next(c for c in s.details["columns"] if c["column"] == "google_ads_spend")
    assert "16 zero week(s) read as a data gap, interpolated" in row["fill"]
    log = next(e for e in r.last.decision_log if e["decision"] == "zero_runs")
    assert log["source"] == "user" and log["lines"] == ["google: 16 zero weeks read as a data gap, interpolated"]


def test_a_run_at_the_edge_has_nothing_to_interpolate_from(tmp_path):
    df = shop_with_zeros()
    df.loc[:2, "google_ads_spend"] = 0.0                      # three zero weeks at the very start
    r, s = step10(tmp_path, df, {"zero_runs": {"google": "gap"}})
    assert (r.b.table["google_ads_spend"].iloc[:3] == 0).all() and not r.b.table["google_ads_spend"].isna().any()
    row = next(c for c in s.details["columns"] if c["column"] == "google_ads_spend")
    assert "at the edge kept as 0" in row["fill"]


def test_unknown_answer_asks_again(tmp_path):
    _, s = step10(tmp_path, shop_with_zeros(), {"zero_runs": {"google": "maybe"}})
    assert s.verdict.value == "needs_input" and "pause or gap" in s.summary


# HLT_FULL (when its file is in tests/data/hlt_full/) -----------------------------------------------------

HLT = Path(__file__).parent / "data" / "hlt_full"
HLT_RAW = next(iter(sorted(HLT.glob("*.csv")) + sorted(HLT.glob("*.xlsx"))), None) if HLT.is_dir() else None


@pytest.mark.skipif(HLT_RAW is None, reason="put HLT_FULL's raw file (and answers.json if it needs one) in tests/data/hlt_full/")
def test_hlt_full_real_zeros(monkeypatch):
    import json

    monkeypatch.delenv("MMM_KEYWORD_PACKS")
    answers = json.loads((HLT / "answers.json").read_text()) if (HLT / "answers.json").exists() else {}
    r = run_pipeline(HLT_RAW, answers, accept_defaults=True)
    s = next(x for x in r.b.steps if x.step == 10)
    real = s.details["real_zeros"]
    pick = lambda key, not_=None: next(v for c, v in real.items() if key in c.lower() and not (not_ and not_ in c.lower()))  # noqa: E731
    got = {"TV": pick("tv", not_="ctv"), "CTV": pick("ctv"), "Google": pick("google"), "Influencer": pick("influencer"), "Print": pick("print")}
    print(f"\nHLT_FULL real zeros {got}, total {sum(real.values())}")
    assert got == {"TV": 66, "CTV": 36, "Google": 16, "Influencer": 127, "Print": 146}
    assert sum(real.values()) == pytest.approx(391, abs=5)
    assert any("16 zero weeks in 2 runs (2024-04-29 → 06-17, 2025-03-31 → 05-19)" in w for w in s.warnings)
