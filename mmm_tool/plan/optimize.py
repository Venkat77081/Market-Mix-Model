"""Phase H: Plan, Optimize (steps 48-54).

Every Plan field maps to one argument of Meridian's BudgetOptimizer.optimize():

  48 Period         start_date, end_date (all weeks, or the last 52)
  49 Budget         budget (fixed budget; defaults to the period's historical spend)
  50 Budget type    fixed_budget; target_roi or target_mroi for a flexible budget;
                    use_kpi (maximise the KPI, or revenue when there is a money value)
  51 Channel limits spend_constraint_lower / _upper, per channel, as a share of its
                    planned spend (the budget split as historically): 0.3 = may move 30%
  52 Locks, floors  a locked channel gets 0 / 0; a floor in money becomes its lower limit
                    (1 - floor / planned spend)
  53 Review         the exact arguments, checked before anything runs
  54 Optimize       the call, then the optimized split run as a What-If scenario on the
                    saved model for 90% ranges of the totals and of the change

Validation needs no Meridian (it runs in the server); optimize() runs in a job.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

SETTINGS = Path(__file__).with_name("plan.yaml")


@lru_cache(maxsize=1)
def settings() -> dict[str, Any]:
    import yaml

    return yaml.safe_load(SETTINGS.read_text(encoding="utf-8"))


class PlanError(ValueError):
    """A plan that can't be optimized as given."""


def defaults(channels: list[str], money: bool) -> dict[str, Any]:
    lim = settings()["limits"]["fixed"]
    return {"period": "all", "budget_type": "fixed", "budget": None, "target": "roi",
            "target_value": float(settings()["flexible"]["target_roi"]), "objective": "revenue" if money else "kpi",
            "limits": {c: {"down": lim["down"], "up": lim["up"]} for c in channels},
            "locks": [], "floors": {}}


def check(plan: dict[str, Any], hist: dict[str, float], money: bool, periods: dict[str, tuple[str, str]]) -> dict[str, Any]:
    """Step 53: the plan turned into BudgetOptimizer.optimize() arguments, with what each
    channel may spend. Raises PlanError with every problem at once.

    hist: each paid channel's spend in the chosen period, in the model's channel order."""
    problems: list[str] = []
    channels = list(hist)
    period = plan.get("period", "all")
    if period not in periods:
        problems.append(f"Unknown period '{period}'; use {', '.join(periods)}.")
    kind = plan.get("budget_type", "fixed")
    if kind not in ("fixed", "flexible"):
        problems.append("Budget type is fixed or flexible.")
    objective = plan.get("objective", "revenue" if money else "kpi")
    if objective not in ("kpi", "revenue") or (objective == "revenue" and not money):
        problems.append("Revenue can only be maximised when the KPI has a money value (step 5); maximise the KPI instead.")
    total = float(sum(hist.values()))
    budget = None
    if kind == "fixed":
        budget = float(plan["budget"]) if plan.get("budget") not in (None, "") else total
        if not budget > 0:
            problems.append("The budget must be above zero.")
    target, tval = plan.get("target", "roi"), plan.get("target_value")
    if kind == "flexible":
        if target not in ("roi", "mroi"):
            problems.append("A flexible budget needs a target ROI or a target marginal ROI.")
        try:
            tval = float(tval)
            if not tval > 0:
                raise ValueError
        except (TypeError, ValueError):
            problems.append("The target must be a number above zero.")
    max_up = float(settings()["limits"]["max_up"])
    share = {c: (hist[c] / total if total else 0.0) for c in channels}
    plan_spend = {c: (budget if budget is not None else total) * share[c] for c in channels}
    lower, upper, rows = [], [], []
    locks, floors = set(plan.get("locks") or []), dict(plan.get("floors") or {})
    for c in sorted(set(locks) | set(floors) | set(plan.get("limits") or {}) - set(channels)):
        if c not in channels:
            problems.append(f"'{c}' is not a channel of this model.")
    for c in channels:
        lim = (plan.get("limits") or {}).get(c, {})
        try:
            down, up = float(lim.get("down", 30)), float(lim.get("up", 30))
        except (TypeError, ValueError):
            problems.append(f"'{c}': limits must be numbers.")
            down, up = 30.0, 30.0
        if not (0 <= down <= 100) or not (0 <= up <= max_up):
            problems.append(f"'{c}': it may go down 0-100% and up 0-{max_up:g}%.")
        lo, hi = down / 100, up / 100
        note = None
        if c in locks:
            lo, hi, note = 0.0, 0.0, "locked"
        elif c in floors and floors[c] not in (None, ""):
            f = float(floors[c])
            ps = plan_spend[c]
            if f < 0:
                problems.append(f"'{c}': a floor can't be negative.")
            elif ps <= 0:
                problems.append(f"'{c}': no planned spend in this period, so a floor can't apply.")
            elif f > ps:
                problems.append(f"'{c}': a floor of {f:,.0f} is above its planned spend ({ps:,.0f}). Meridian keeps each "
                                "channel within limits around its planned spend (the budget split as in the period), so a "
                                "floor can only hold some of it back; lower the floor or raise the budget.")
            else:
                lo = min(lo, 1 - f / ps)          # its minimum, planned x (1 - lo), is at least the floor
                note = "floor"
        if hist[c] <= 0:
            lo, hi, note = 0.0, 0.0, "no spend in the period"
        lower.append(lo)
        upper.append(hi)
        rows.append({"channel": c, "historical": hist[c], "planned": plan_spend[c], "min": plan_spend[c] * (1 - lo),
                     "max": plan_spend[c] * (1 + hi), "down": lo, "up": hi, "note": note})
    if kind == "fixed" and budget and not problems:
        lo_sum, hi_sum = sum(r["min"] for r in rows), sum(r["max"] for r in rows)
        if lo_sum > budget * (1 + 1e-9) or hi_sum < budget * (1 - 1e-9):
            problems.append(f"The channels can spend {lo_sum:,.0f}-{hi_sum:,.0f} between them, which doesn't contain the budget.")
    if problems:
        raise PlanError(" ".join(problems))
    start, end = periods[period]
    args: dict[str, Any] = {"start_date": start, "end_date": end, "fixed_budget": kind == "fixed",
                            "spend_constraint_lower": lower, "spend_constraint_upper": upper,
                            "use_kpi": objective == "kpi", "confidence_level": 0.9, "gtol": float(settings()["gtol"])}
    if kind == "fixed":
        args["budget"] = budget
    else:
        args["target_roi" if target == "roi" else "target_mroi"] = tval
    return {"args": args, "channels": rows, "period": period, "historical_total": total, "budget": budget,
            "budget_type": kind, "objective": objective}


def _table(ds, channels: list[str]) -> dict[str, Any]:
    def metric(var, ch):
        x = ds[var].sel(channel=ch)
        if "metric" in x.dims:
            return {"mean": float(x.sel(metric="mean")), "lo": float(x.sel(metric="ci_lo")), "hi": float(x.sel(metric="ci_hi"))}
        return {"mean": float(x)}
    rows = {c: {"spend": float(ds["spend"].sel(channel=c)), "share": float(ds["pct_of_spend"].sel(channel=c)),
                "outcome": metric("incremental_outcome", c), "roi": metric("roi", c), "mroi": metric("mroi", c),
                "cpik": metric("cpik", c)} for c in channels}
    attrs = {k: (float(np.asarray(v)) if np.asarray(v).dtype.kind in "fi" else (bool(v) if isinstance(v, (bool, np.bool_)) else str(v)))
             for k, v in ds.attrs.items()}
    return {"channels": rows, "attrs": attrs}


def optimize(mmm, checked: dict[str, Any], out_dir: Path | None = None) -> dict[str, Any]:
    """Step 54: run BudgetOptimizer.optimize() with the checked arguments."""
    from meridian.analysis import optimizer

    from ..whatif.scenario import exact

    import warnings

    with warnings.catch_warnings():   # Meridian's own optimal-frequency search passes these to itself
        warnings.filterwarnings("ignore", message=r"A `\w+` value was passed in the `new_data` argument")
        res = optimizer.BudgetOptimizer(mmm).optimize(**checked["args"])
    channels = [str(c) for c in res.optimized_data.channel.values]
    before, after = _table(res.nonoptimized_data, channels), _table(res.optimized_data, channels)
    hist = {r["channel"]: r["historical"] for r in checked["channels"]}
    mult = {c: (after["channels"][c]["spend"] / hist[c] if hist.get(c) else 1.0) for c in channels}
    totals = exact(mmm, mult, checked["period"])          # the optimized split on every draw, against history
    report = None
    if out_dir is not None:
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            res.output_optimization_summary("optimization_summary.html", str(out_dir))
            report = "optimization_summary.html"
        except Exception as e:  # noqa: BLE001 - the report is a download, never a reason to fail
            report = f"error: {type(e).__name__}: {e}"
    return {"plan": checked, "channels": channels, "before": before, "after": after,
            "multipliers": mult, "totals": totals, "report": report}
