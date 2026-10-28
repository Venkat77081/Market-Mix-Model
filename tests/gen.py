"""Test datasets; the generator lives in mmm_tool.samples so the UI can use it too."""
from mmm_tool.samples import *  # noqa: F401,F403
from mmm_tool.samples import CHANNELS, make_insurance  # noqa: F401


# Step 24 knots preview: synthetic weekly exports where the truth is known.
import numpy as _np  # noqa: E402
import pandas as _pd  # noqa: E402

KNOT_CASES = ("flat_tv", "trend_tv", "always_on", "tiny_burst")


def _adstock(x, lag=8, decay=0.5):
    w = decay ** _np.arange(lag + 1)
    w = w / w.sum()
    out = _np.zeros(len(x))
    for i, wi in enumerate(w):
        out[i:] += wi * x[: len(x) - i]
    return out


def make_knots_case(kind: str, seed: int = 1, weeks: int = 156) -> _pd.DataFrame:
    """TV's true effect is modest against the weekly noise (sd 150 policies on about 4,000),
    as in real MMM data: the case where a flexible baseline can take its credit.

    flat_tv     flat true baseline; TV runs in flights of 4-7 weeks; two always-on channels
    trend_tv    as flat_tv, but the true baseline falls 30% over the period
    always_on   flat baseline, only always-on channels (no burst channel)
    tiny_burst  flat_tv plus a small, noisy, on-off channel under 5% of paid spend"""
    if kind not in KNOT_CASES:
        raise ValueError(kind)
    rng = _np.random.default_rng(seed)
    t = _np.arange(weeks)
    df = _pd.DataFrame({"week_start": _pd.date_range("2023-01-02", periods=weeks, freq="7D")})
    level = 4000.0 * (1 - 0.3 * t / (weeks - 1)) if kind == "trend_tv" else _np.full(weeks, 4000.0)
    kpi = level + rng.normal(0, 150, weeks)
    for ch, mean in (("search", 1.2e6), ("aggregator", 1.0e6)):                     # always on: CV about 0.12
        x = _np.round(mean * (1 + rng.normal(0, 0.12, weeks)), -3)
        df[f"{ch}_spend"] = x
        kpi += 120 * _np.log1p(_adstock(x) / x.mean())
    if kind != "always_on":
        tv, i = _np.zeros(weeks), int(rng.integers(0, 6))
        while i < weeks:                                                             # flights: 4-7 weeks on, 6-10 off
            on = int(rng.integers(4, 8))
            tv[i:i + on] = rng.normal(4.0e6, 0.4e6, len(tv[i:i + on]))
            i += on + int(rng.integers(6, 11))
        df["tv_spend"] = _np.round(_np.clip(tv, 0, None), -3)
        kpi += 200 * _np.log1p(_adstock(df["tv_spend"].to_numpy()) / df.loc[df["tv_spend"] > 0, "tv_spend"].mean())
    if kind == "tiny_burst":                                                         # about 3% of paid spend, no effect
        on = rng.random(weeks) < 0.35
        df["print_spend"] = _np.round(_np.where(on, rng.gamma(2.0, 1.1e5, weeks), 0.0), -3)
    df["price_index"] = _np.round(100 + rng.normal(0, 2, weeks), 2)
    kpi -= 15 * (df["price_index"] - 100)
    df["new_policies"] = _np.round(kpi)
    return df


def write_knots_case(dir, kind: str, seed: int = 1):
    from pathlib import Path

    df = make_knots_case(kind, seed)
    path = Path(dir) / f"knots_{kind}_{seed}.csv"
    df.assign(week_start=df["week_start"].dt.strftime("%Y-%m-%d")).to_csv(path, index=False)
    return path
