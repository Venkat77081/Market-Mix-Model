"""Phase G: What-If (steps 43-47), the calculations.

A scenario multiplies each paid channel's spend in a period (all weeks, or the
last 52) by its own factor: 1.2 is "+20%". Media units scale with spend, and
for a reach-and-frequency channel reach scales and frequency stays, exactly as
Meridian's own response curves do (Analyzer.response_curves); weeks before the
period are scaled too, so carryover into it follows the scenario.

  curves  step 44 (quick): each channel's incremental KPI (and revenue, when there
          is a money value) at a grid of multipliers, from the saved model. The
          browser interpolates them as the user moves a slider, with no server call.
          Channels are added up, which is exact for Meridian: a channel's effect
          doesn't depend on the others' spend.
  exact   step 45: the same scenario on every posterior draw at once (incremental_outcome
          with the scaled media), so the total gets its own 90% range.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

SETTINGS = Path(__file__).with_name("whatif.yaml")
CONFIDENCE = 0.9


@lru_cache(maxsize=1)
def settings() -> dict[str, Any]:
    import yaml

    return yaml.safe_load(SETTINGS.read_text(encoding="utf-8"))


def grid() -> list[float]:
    g = settings()["curve_grid"]
    n = int(round(float(g["max"]) / float(g["step"])))
    return [round(i * float(g["step"]), 4) for i in range(n + 1)]


def periods(mmm) -> dict[str, list[str] | None]:
    """'all' (every week) and, with two years or more, 'last_52' (as on the Model Summary)."""
    weeks = [str(t)[:10] for t in mmm.input_data.time.values]
    out: dict[str, list[str] | None] = {"all": None}
    if len(weeks) >= 104:
        out["last_52"] = weeks[-52:]
    return out


def _selected(mmm, period: str) -> list[str] | None:
    p = periods(mmm)
    if period not in p:
        raise ValueError(f"Unknown period '{period}'; use {', '.join(p)}.")
    return p[period]


def curves(mmm, period: str = "all") -> dict[str, Any]:
    """Step 44's curves for one period: {channel: {multiplier, spend, kpi, kpi_lo, kpi_hi, revenue}}."""
    from ..fit import model as m

    an, sel, g = m.analyzer(mmm), _selected(mmm, period), grid()
    money = m.uses_revenue(mmm)
    rc_k = an.response_curves(spend_multipliers=g, selected_times=sel, confidence_level=CONFIDENCE, use_kpi=True)
    rc_r = an.response_curves(spend_multipliers=g, selected_times=sel, confidence_level=CONFIDENCE, use_kpi=False) if money else None
    out = {}
    for c in [str(x) for x in rc_k.channel.values]:
        inc = rc_k["incremental_outcome"].sel(channel=c)
        row = {"multiplier": g, "spend": np.asarray(rc_k["spend"].sel(channel=c), float).round(2).tolist(),
               "kpi": np.asarray(inc.sel(metric="mean"), float).round(3).tolist(),
               "kpi_lo": np.asarray(inc.sel(metric="ci_lo"), float).round(3).tolist(),
               "kpi_hi": np.asarray(inc.sel(metric="ci_hi"), float).round(3).tolist()}
        if rc_r is not None:
            row["revenue"] = np.asarray(rc_r["incremental_outcome"].sel(channel=c, metric="mean"), float).round(2).tolist()
        out[c] = row
    weeks = sel or [str(t)[:10] for t in mmm.input_data.time.values]
    return {"period": period, "weeks": len(weeks), "first_week": weeks[0], "last_week": weeks[-1],
            "money": money, "channels": out, "grid": g}


def _scaled(mmm, mult: dict[str, float]):
    """DataTensors with each paid channel's media (or reach) scaled by its multiplier."""
    from meridian.analysis import tensors

    d = mmm.input_data
    kw = {}
    if d.media is not None:
        names = [str(x) for x in d.media.media_channel.values]
        kw["media"] = np.asarray(d.media.values, float) * np.array([float(mult.get(c, 1.0)) for c in names])
    if d.reach is not None:
        names = [str(x) for x in d.reach.rf_channel.values]
        kw["reach"] = np.asarray(d.reach.values, float) * np.array([float(mult.get(c, 1.0)) for c in names])
        kw["frequency"] = np.asarray(d.frequency.values, float)
    return tensors.DataTensors(**kw)


def _draws(mmm, mult: dict[str, float], sel, use_kpi: bool) -> np.ndarray:
    """Incremental outcome per posterior draw and paid channel, shape (draws, channels)."""
    from ..fit import model as m

    an = m.analyzer(mmm)
    inc = an.incremental_outcome(new_data=_scaled(mmm, mult), selected_times=sel, use_kpi=use_kpi,
                                 include_non_paid_channels=False)
    x = np.asarray(inc, float)
    return x.reshape(-1, x.shape[-1])


def favour(share: float) -> str:
    """Words for the share of posterior draws in which the scenario beats the current plan."""
    for cut, words in ((0.9, "very likely better"), (0.7, "probably better"), (0.3, "can't tell"),
                       (0.1, "probably worse")):
        if share >= cut:
            return words
    return "very likely worse"


def change_draws(mmm, mult: dict[str, float], period: str = "all", use_kpi: bool = True) -> np.ndarray:
    """The scenario minus the current plan, per posterior draw (the same draw for both), summed over channels."""
    sel = _selected(mmm, period)
    return _draws(mmm, mult, sel, use_kpi).sum(axis=1) - _draws(mmm, {}, sel, use_kpi).sum(axis=1)


def _summary(draws: np.ndarray) -> dict[str, float]:
    lo, hi = np.quantile(draws, [(1 - CONFIDENCE) / 2, (1 + CONFIDENCE) / 2])
    return {"mean": float(draws.mean()), "lo": float(lo), "hi": float(hi)}


def exact(mmm, mult: dict[str, float], period: str = "all") -> dict[str, Any]:
    """Step 45: the scenario and the current plan on the saved model, with 90% ranges."""
    from ..fit import model as m

    sel = _selected(mmm, period)
    paid = [str(c) for c in mmm.input_data.get_all_paid_channels()]
    unknown = sorted(set(mult) - set(paid))
    if unknown:
        raise ValueError(f"Not channels of this model: {unknown}.")
    base_spend = spend_in(mmm, sel)
    money = m.uses_revenue(mmm)
    weeks = sel or [str(t)[:10] for t in mmm.input_data.time.values]
    out: dict[str, Any] = {"period": period, "first_week": weeks[0], "last_week": weeks[-1], "weeks": len(weeks),
                           "multipliers": {c: float(mult.get(c, 1.0)) for c in paid}}
    kpi = {"scenario": _draws(mmm, mult, sel, True), "current": _draws(mmm, {}, sel, True)}
    rev = {"scenario": _draws(mmm, mult, sel, False), "current": _draws(mmm, {}, sel, False)} if money else None
    for label, scen in (("scenario", mult), ("current", {})):
        k = kpi[label]
        out[label] = {"kpi": _summary(k.sum(axis=1)), "kpi_by_channel": dict(zip(paid, k.mean(axis=0).round(3).tolist())),
                      "spend": float(sum(base_spend[c] * float(scen.get(c, 1.0)) for c in paid))}
        if rev is not None:
            out[label]["revenue"] = _summary(rev[label].sum(axis=1))
    # The change, draw by draw (the same draws for both), so its range is the range of the difference,
    # and the share of draws in which the scenario comes out ahead.
    dk = kpi["scenario"].sum(axis=1) - kpi["current"].sum(axis=1)
    out["delta_kpi"] = {**_summary(dk), "share_better": float((dk > 0).mean()), "draws": int(dk.size)}
    out["delta_kpi"]["favour"] = favour(out["delta_kpi"]["share_better"])
    if rev is not None:
        dr = rev["scenario"].sum(axis=1) - rev["current"].sum(axis=1)
        out["delta_revenue"] = {**_summary(dr), "share_better": float((dr > 0).mean()), "draws": int(dr.size),
                                "favour": favour(float((dr > 0).mean()))}
    ds = out["scenario"]["spend"] - out["current"]["spend"]
    out["delta_spend"] = {"amount": ds, "pct": ds / out["current"]["spend"] if out["current"]["spend"] else None}
    return out


def spend_in(mmm, sel: list[str] | None) -> dict[str, float]:
    """Each paid channel's total spend in the period (all regions)."""
    d = mmm.input_data
    times = [str(t)[:10] for t in d.time.values]
    keep = np.array([sel is None or t in set(sel) for t in times])
    out = {}
    for arr, dim in ((d.media_spend, "media_channel"), (d.rf_spend, "rf_channel")):
        if arr is None:
            continue
        x = np.asarray(arr.values, float)
        if x.ndim == 2:          # spend given per channel for the whole period, not by week
            x = x[:, None, :].repeat(len(times), axis=1) / len(times)
        for i, c in enumerate(str(v) for v in arr[dim].values):
            out[c] = float(x[:, keep, i].sum())
    return out
