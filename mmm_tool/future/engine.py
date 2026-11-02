"""Meridian calculations for future plans (run in background jobs, never in the web server).

Media outcome per plan week comes from Analyzer.incremental_outcome(new_data=build_future_data(...),
selected_times=plan weeks, aggregate_times=False): one value per posterior draw, week and channel.
The plan and the reference (the template at 100%, with the period's cost change) use the same
draws, so their difference is exact draw by draw. With a forecast:

    total[t] = forecast[t] + (media_plan[t] - media_reference[t])

and its range combines the posterior draws with the forecast's own range (treated as a normal
error, the same z for every week of a draw, so the period total's range is not understated).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .data import build_future_data, channels

Q = (0.05, 0.95)


def _settings():
    from .period import settings

    return settings()


def media_draws(mmm, nd, plan: list[str], use_kpi: bool = True) -> np.ndarray:
    """Incremental outcome per posterior draw, plan week and paid channel: shape (draws, weeks, channels)."""
    from ..fit import model as m

    x = np.asarray(m.analyzer(mmm).incremental_outcome(new_data=nd, selected_times=plan, aggregate_times=False,
                                                       use_kpi=use_kpi, include_non_paid_channels=False), float)
    return x.reshape(-1, x.shape[-2], x.shape[-1])


def _band(x: np.ndarray, axis=0) -> dict[str, Any]:
    lo, hi = np.quantile(x, Q, axis=axis)
    return {"mean": np.mean(x, axis=axis), "lo": lo, "hi": hi}


def _r(v):
    return [round(float(x), 4) for x in np.atleast_1d(v)]


def combine(change: np.ndarray, forecast: dict | None) -> np.ndarray | None:
    """Total draws (draws, weeks) = forecast + its error + the change; None without a forecast."""
    if not forecast or forecast.get("mean") is None:
        return None
    f = np.asarray(forecast["mean"], float)[None, :]
    if forecast.get("lower") is not None:
        sd = (np.asarray(forecast["upper"], float) - np.asarray(forecast["lower"], float)) / (2 * float(_settings()["forecast"]["z90"]))
        z = np.random.default_rng(int(_settings()["seed"])).standard_normal((change.shape[0], 1))
        z = (z - z.mean()) / (z.std() or 1)                 # centred: plan A's total is exactly the forecast
        return f + z * sd[None, :] + change
    return f + change


def evaluate(mmm, resolved: dict, forecast: dict | None, mult: dict | None = None, spend: dict | None = None,
             cost: dict | None = None, rpk: dict | None = None, units: dict | None = None) -> dict[str, Any]:
    """A plan on every posterior draw: per week and in total, its media outcome, the reference's, the
    change, and (with a forecast) the predicted total; plus spend per channel and week."""
    plan = resolved["plan"]
    paid = channels(mmm)
    nd_ref, info_ref = build_future_data(mmm, resolved, cost=cost, rpk=rpk)
    nd, info = build_future_data(mmm, resolved, mult=mult, spend=spend, cost=cost, rpk=rpk, units=units)
    ref = media_draws(mmm, nd_ref, plan)                     # (draws, weeks, channels)
    pln = ref if (not mult or all(abs(v - 1) < 1e-12 for v in mult.values())) and not spend and not units else media_draws(mmm, nd, plan)
    ref_w, pln_w = ref.sum(axis=2), pln.sum(axis=2)          # (draws, weeks)
    change_w = pln_w - ref_w
    total_w = combine(change_w, forecast)
    out: dict[str, Any] = {"weeks": plan, "channels": paid, "draws": int(ref.shape[0]),
                           "spend": {c: _r(v) for c, v in info["spend"].items()},
                           "template_spend": {c: _r(v) for c, v in info_ref["spend"].items()},
                           "multipliers": {c: (sum(info["spend"][c]) / sum(info_ref["spend"][c]) if sum(info_ref["spend"][c]) else 1.0) for c in paid}}
    b = lambda x: {k: _r(v) for k, v in _band(x).items()}  # noqa: E731
    out["per_week"] = {"reference": _r(ref_w.mean(axis=0)), "media": b(pln_w), "change": b(change_w),
                       "total": b(total_w) if total_w is not None else None,
                       "forecast": forecast.get("mean") if forecast else None,
                       "forecast_lower": forecast.get("lower") if forecast else None,
                       "forecast_upper": forecast.get("upper") if forecast else None}
    tb = lambda x: {k: float(v) for k, v in _band(x.sum(axis=1)).items()}  # noqa: E731
    out["totals"] = {"reference": float(ref_w.sum(axis=1).mean()), "media": tb(pln_w), "change": tb(change_w),
                     "total": tb(total_w) if total_w is not None else None,
                     "forecast": forecast.get("total") if forecast else None,
                     "spend": float(sum(sum(v) for v in info["spend"].values())),
                     "template_spend": float(sum(sum(v) for v in info_ref["spend"].values())),
                     "share_better": float((change_w.sum(axis=1) > 0).mean())}
    out["by_channel"] = {c: {"media": float(pln[:, :, i].sum(axis=1).mean()), "reference": float(ref[:, :, i].sum(axis=1).mean())}
                         for i, c in enumerate(paid)}
    return out


def curves(mmm, resolved: dict, cost: dict | None, rpk: dict | None, grid: list[float]) -> dict[str, Any]:
    """Each channel's media outcome over the plan weeks at every multiplier on the grid (mean of the
    draws). Channels add up in Meridian, so one call per grid point serves every channel."""
    plan, paid = resolved["plan"], channels(mmm)
    rows = {c: [] for c in paid}
    tspend = None
    for g in grid:
        nd, info = build_future_data(mmm, resolved, mult={c: g for c in paid}, cost=cost, rpk=rpk)
        x = media_draws(mmm, nd, plan).sum(axis=1).mean(axis=0)          # (channels,)
        for i, c in enumerate(paid):
            rows[c].append(round(float(x[i]), 4))
        if abs(g - 1) < 1e-9:
            tspend = {c: float(sum(v)) for c, v in info["spend"].items()}
    if tspend is None:
        _, info = build_future_data(mmm, resolved, cost=cost, rpk=rpk)
        tspend = {c: float(sum(v)) for c, v in info["spend"].items()}
    return {"grid": grid, "channels": {c: {"kpi": rows[c], "template_spend": tspend[c]} for c in paid}}


def optimize(mmm, resolved: dict, forecast: dict | None, cost: dict | None, rpk: dict | None, checked: dict) -> dict[str, Any]:
    """BudgetOptimizer.optimize on the future weeks (new_data at 100%), then the optimized split
    run as a plan on every draw, for the totals with the forecast."""
    import warnings

    from meridian.analysis import optimizer

    from ..plan.optimize import _table

    nd, info = build_future_data(mmm, resolved, cost=cost, rpk=rpk)
    args = {k: v for k, v in checked["args"].items() if k not in ("start_date", "end_date")}
    args.update(start_date=resolved["plan"][0], end_date=resolved["plan"][-1])
    print("BudgetOptimizer.optimize(new_data=<lead-in + plan weeks>,", flush=True)
    for k, v in args.items():
        print(f"    {k}={v!r},", flush=True)
    print(f")   # channels, in model order: {[r['channel'] for r in checked['channels']]}", flush=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"A `\w+` value was passed in the `new_data` argument")
        res = optimizer.BudgetOptimizer(mmm).optimize(new_data=nd, **args)
    chans = [str(c) for c in res.optimized_data.channel.values]
    before, after = _table(res.nonoptimized_data, chans), _table(res.optimized_data, chans)
    tmpl = {c: float(sum(v)) for c, v in info["spend"].items()}
    mult = {c: (after["channels"][c]["spend"] / tmpl[c] if tmpl.get(c) else 1.0) for c in chans}
    plan_eval = evaluate(mmm, resolved, forecast, mult=mult, cost=cost, rpk=rpk)
    return {"plan": checked, "channels": chans, "before": before, "after": after, "multipliers": mult,
            "evaluation": plan_eval, "optimizer_outcome": float(res.optimized_data.attrs.get("total_incremental_outcome", np.nan))}
