"""Phase H: Plan, Optimize (steps 48-54).

Every Plan field maps to one argument of Meridian's BudgetOptimizer.optimize():

  48 Period         start_date, end_date (all weeks, or the last 52)
  49 Budget         budget (fixed budget; defaults to the period's historical spend)
  50 Budget type    fixed_budget; target_roi or target_mroi for a flexible budget;
                    use_kpi (maximise the KPI, or revenue when there is a money value)
  51 Channel limits spend_constraint_lower / _upper, per channel, as a share of its
                    planned spend (the budget split as historically): may go down 30% and
                    up 50% by default; a locked channel gets 0 / 0
  52 Floors         lowest allowed = planned if locked, else the higher of planned x (1 - down)
                    and the floor; highest allowed = planned if locked, else planned x (1 + up).
                    Passed as lower = 1 - lowest / planned and upper = highest / planned - 1
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


FLOOR_ABOVE = ("A floor can only limit cuts. To raise a channel, change its planned spend or its "
               "'may go up' limit.")


def grid_step(budget: float, gtol: float) -> float:
    """The amount Meridian rounds each channel's optimized spend to: 10^n, with n set so the rounding
    stays within budget x gtol (as meridian.analysis.optimizer.get_round_factor)."""
    import math

    tol = budget * gtol
    return 1.0 if tol < 1 else float(10 ** (int(math.log10(tol)) + 1))


def display_name(name: str) -> str:
    """'tv_video' -> 'TV Video', 'google' -> 'Google': words capitalised, short ones with no vowel
    (tv, ctv, ppc, sms) in capitals."""
    import re

    upper = {str(w).lower() for w in settings().get("display_upper") or []}
    words = [w for w in re.split(r"[_\s]+", str(name)) if w]
    return " ".join(w.upper() if w.lower() in upper or len(w) <= 2 or (len(w) == 3 and not set(w.lower()) & set("aeiou"))
                    else w[:1].upper() + w[1:] for w in words) or str(name)


def bounds(planned: float, down: float, up: float, locked: bool, floor: float | None) -> dict[str, Any]:
    """Step 52: what one channel may spend, and which limit sets its lowest."""
    if locked:
        return {"lowest": planned, "highest": planned, "source": "locked", "from_percent": planned, "floor_not_needed": False}
    from_pct = planned * (1 - down / 100)
    highest = planned * (1 + up / 100)
    if floor is not None and floor >= from_pct:
        return {"lowest": floor, "highest": highest, "source": "floor", "from_percent": from_pct, "floor_not_needed": False}
    return {"lowest": from_pct, "highest": highest, "source": f"−{down:g}%", "from_percent": from_pct,
            "floor_not_needed": floor is not None}


def check(plan: dict[str, Any], hist: dict[str, float], money: bool, periods: dict[str, tuple[str, str]],
          peaks: dict[str, float] | None = None) -> dict[str, Any]:
    """Step 53: the plan turned into BudgetOptimizer.optimize() arguments, with what each
    channel may spend. Raises PlanError with every problem at once.

    hist: each paid channel's spend in the chosen period, in the model's channel order.
    peaks: each channel's highest spend in any past window as long as the period (for the warning)."""
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
    lower, upper, rows, warnings = [], [], [], []
    peak_ratio = float(settings()["limits"].get("seen_peak_ratio", 2.0))
    lim_default = settings()["limits"]["fixed" if kind == "fixed" else "flexible"]
    locks, floors = set(plan.get("locks") or []), dict(plan.get("floors") or {})
    for c in sorted(set(locks) | set(floors) | set(plan.get("limits") or {}) - set(channels)):
        if c not in channels:
            problems.append(f"'{c}' is not a channel of this model.")
    for c in channels:
        lim = (plan.get("limits") or {}).get(c, {})
        try:
            down, up = float(lim.get("down", lim_default["down"])), float(lim.get("up", lim_default["up"]))
        except (TypeError, ValueError):
            problems.append(f"'{c}': limits must be numbers.")
            down, up = float(lim_default["down"]), float(lim_default["up"])
        if not (0 <= down <= 100) or not (0 <= up <= max_up):
            problems.append(f"'{c}': it may go down 0-100% and up 0-{max_up:g}%.")
        ps = plan_spend[c]
        floor = None
        if c in floors and floors[c] not in (None, "") and c not in locks:
            try:
                floor = float(floors[c])
            except (TypeError, ValueError):
                problems.append(f"'{c}': the floor must be a number.")
            if floor is not None and floor < 0:
                problems.append(f"'{c}': a floor can't be negative.")
                floor = None
            elif floor is not None and ps <= 0:
                problems.append(f"'{c}': no planned spend in this period, so a floor can't apply.")
                floor = None
            elif floor is not None and floor > ps * (1 + 1e-9):
                problems.append(f"'{c}': {FLOOR_ABOVE}")
                floor = None
        b = bounds(ps, down, up, c in locks, floor)
        note = "locked" if c in locks else ("floor" if b["source"] == "floor" else None)
        if hist[c] <= 0:
            b = {"lowest": 0.0, "highest": 0.0, "source": "no spend", "from_percent": 0.0, "floor_not_needed": False}
            note = "no spend in the period"
        lo = 1 - b["lowest"] / ps if ps > 0 else 0.0           # the ratios Meridian takes, from the effective bounds
        hi = b["highest"] / ps - 1 if ps > 0 else 0.0
        lower.append(round(max(0.0, lo), 12))                 # 0.3, not 0.30000000000000004
        upper.append(round(max(0.0, hi), 12))
        seen = (peaks or {}).get(c)
        beyond = bool(seen and b["highest"] > peak_ratio * seen)
        if beyond:
            warnings.append(f"'{c}': its highest allowed spend is above {peak_ratio:g}x its highest past spend for a period "
                            "this long, outside what the model has seen.")
        rows.append({"channel": c, "display": display_name(c), "historical": hist[c], "planned": ps,
                     "min": b["lowest"], "max": b["highest"], "down": lower[-1], "up": upper[-1], "note": note,
                     "lowest": b["lowest"], "highest": b["highest"], "lowest_source": b["source"],
                     "from_percent": b["from_percent"], "floor": floor, "floor_not_needed": b["floor_not_needed"],
                     "limit_down": down, "limit_up": up, "peak": seen, "beyond_seen": beyond})
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
            "budget_type": kind, "objective": objective, "warnings": warnings,
            "grid_step": grid_step(budget if budget is not None else total, float(settings()["gtol"]))}


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

    args = checked["args"]
    print("BudgetOptimizer.optimize(", flush=True)                 # the run log shows exactly what was passed
    for k, v in args.items():
        print(f"    {k}={v!r},", flush=True)
    print(f")   # channels, in model order: {[r['channel'] for r in checked['channels']]}", flush=True)
    with warnings.catch_warnings():   # Meridian's own optimal-frequency search passes these to itself
        warnings.filterwarnings("ignore", message=r"A `\w+` value was passed in the `new_data` argument")
        res = optimizer.BudgetOptimizer(mmm).optimize(**args)
    channels = [str(c) for c in res.optimized_data.channel.values]
    before, after = _table(res.nonoptimized_data, channels), _table(res.optimized_data, channels)
    hist = {r["channel"]: r["historical"] for r in checked["channels"]}
    mult = {c: (after["channels"][c]["spend"] / hist[c] if hist.get(c) else 1.0) for c in channels}
    totals = exact(mmm, mult, checked["period"])          # the optimized split on every draw, against history
    report = None
    if out_dir is not None:
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            from ..summary.offline import make_offline

            res.output_optimization_summary("optimization_summary.html", str(out_dir))
            make_offline(out_dir / "optimization_summary.html")           # charts work offline
            report = "optimization_summary.html"
        except Exception as e:  # noqa: BLE001 - the report is a download, never a reason to fail
            report = f"error: {type(e).__name__}: {e}"
    return {"plan": checked, "channels": channels, "before": before, "after": after,
            "multipliers": mult, "totals": totals, "report": report}
