"""The forecast: the KPI in each plan week if spend stays like last year.

Meridian can't forecast its baseline (trend, seasons, controls) into weeks it hasn't seen, so the
total for a plan is   total[t] = forecast[t] + (media_plan[t] - media_reference[t]),
where the reference is the template at 100%. The forecast is assumed to include last year's media.

  builtin  the KPI in the same weeks last year x (1 + growth); growth defaults to the latest 52
           weeks against the 52 before. Bounds: step 31's same-week-last-year error (exactly the
           error of this kind of guess), else its holdout error.
  csv      date, forecast, and optionally lower and upper: from any forecasting tool or spreadsheet
  total    one number for the whole period, spread like the template's KPI
  none     media-only results; totals are hidden
"""
from __future__ import annotations

import hashlib
import io
import json
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

SOURCES = ("builtin", "csv", "total", "none")


class ForecastError(ValueError):
    pass


def _hash(mean, lower, upper) -> str:
    body = json.dumps([[round(float(x), 4) for x in v] if v is not None else None for v in (mean, lower, upper)])
    return hashlib.sha256(body.encode()).hexdigest()[:12]


def _out(source: str, plan: list[str], mean, lower=None, upper=None, note: str = "", **extra) -> dict[str, Any]:
    mean = [float(x) for x in mean]
    lower = [float(x) for x in lower] if lower is not None else None
    upper = [float(x) for x in upper] if upper is not None else None
    return {"source": source, "weeks": plan, "mean": mean, "lower": lower, "upper": upper,
            "total": float(sum(mean)), "hash": _hash(mean, lower, upper), "note": note, **extra}


def default_growth(weeks: list[str], kpi: dict[str, float]) -> float:
    """The latest 52 weeks against the 52 before; 0 with under two years."""
    if len(weeks) < 104:
        return 0.0
    a = sum(kpi[w] for w in weeks[-52:])
    b = sum(kpi[w] for w in weeks[-104:-52])
    return float(a / b - 1) if b else 0.0


def builtin(plan: list[str], template: list[str], weeks: list[str], kpi: dict[str, float],
            growth: float | None, step31: dict | None) -> dict[str, Any]:
    last_year = [str((pd.Timestamp(w) - timedelta(days=364)).date()) for w in plan]
    src = last_year if all(w in kpi for w in last_year) else template
    g = default_growth(weeks, kpi) if growth is None else float(growth)
    mean = [kpi[w] * (1 + g) for w in src]
    s = step31 or {}
    err, why = s.get("naive_mape"), "step 31's error of 'same week last year' on held-back weeks"
    if err is None:
        err = ((s.get("accuracy") or {}).get("Test") or {}).get("MAPE")
        why = "the model's holdout error (step 31)"
    lower = upper = None
    note = (f"KPI in {'the same weeks last year' if src is last_year else 'the template weeks'} "
            f"x (1 {'+' if g >= 0 else '−'} {abs(g):.1%})")
    if err is not None:
        lower, upper = [m * (1 - err) for m in mean], [m * (1 + err) for m in mean]
        note += f"; range ±{err:.1%} ({why})"
    if src is last_year and list(template) != last_year:
        note += ". It includes last year's media, while plans change the template's: copy the same weeks last year to match them"
    return _out("builtin", plan, mean, lower, upper, note, growth=g, source_weeks=src,
                bounds_from=why if err is not None else None, error=err)


def _match(plan: list[str], dates: pd.Series) -> list[int | None]:
    """For each plan week, the row whose date is within 3 days of it."""
    out = []
    for w in plan:
        d = (dates - pd.Timestamp(w)).abs()
        i = d.idxmin() if len(d) else None
        out.append(int(i) if i is not None and d[i] <= pd.Timedelta(days=3) else None)
    return out


def from_csv(text: str, plan: list[str]) -> dict[str, Any]:
    """date, forecast and optionally lower and upper; columns found by name, else by position."""
    try:
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    except Exception as e:  # noqa: BLE001 - shown to the user
        raise ForecastError(f"The forecast file couldn't be read: {e}") from e
    cols = {c: str(c).strip().lower() for c in df.columns}
    date = next((c for c, n in cols.items() if any(k in n for k in ("date", "week", "period", "time"))), df.columns[0])
    rest = [c for c in df.columns if c != date]
    low = next((c for c in rest if any(k in cols[c] for k in ("lower", "low", "p05", "min"))), None)
    high = next((c for c in rest if any(k in cols[c] for k in ("upper", "high", "p95", "max"))), None)
    fc = next((c for c in rest if c not in (low, high) and any(k in cols[c] for k in ("forecast", "pred", "mean", "kpi", "value"))),
              next((c for c in rest if c not in (low, high)), None))
    if fc is None:
        raise ForecastError("The forecast file needs a date column and a forecast column.")
    dates = pd.to_datetime(df[date], errors="coerce", dayfirst=False)
    if dates.isna().all():
        raise ForecastError(f"The column '{date}' doesn't hold dates.")
    idx = _match(plan, dates)
    missing = [w for w, i in zip(plan, idx) if i is None]
    if missing:
        raise ForecastError(f"The forecast has no row for {len(missing)} plan week(s), e.g. {missing[:3]}.")
    num = lambda c: [float(pd.to_numeric(df.loc[i, c], errors="coerce")) for i in idx]  # noqa: E731
    mean, lo, hi = num(fc), num(low) if low else None, num(high) if high else None
    if any(np.isnan(mean)) or (lo and any(np.isnan(lo))) or (hi and any(np.isnan(hi))):
        raise ForecastError("Some plan weeks have no number in the forecast file.")
    if (lo is None) != (hi is None):
        lo = hi = None                         # one bound alone isn't a range
    if lo is not None:
        bad = [w for w, m, a, b in zip(plan, mean, lo, hi) if not a <= m <= b]
        if bad:
            raise ForecastError(f"The lower bound must be at or below the forecast and the upper at or above it; "
                                f"not so in {len(bad)} week(s), e.g. {bad[0]}.")
    return _out("csv", plan, mean, lo, hi, f"uploaded forecast ('{fc}'{', with bounds' if lo else ''})")


def from_total(total: float, plan: list[str], template: list[str], kpi: dict[str, float]) -> dict[str, Any]:
    if not total or float(total) <= 0:
        raise ForecastError("Type a total above zero.")
    shape = np.array([kpi[w] for w in template], float)
    shape = shape / shape.sum() if shape.sum() > 0 else np.full(len(plan), 1 / len(plan))
    return _out("total", plan, float(total) * shape, note="one total, spread like the template weeks' KPI")


def none(plan: list[str]) -> dict[str, Any]:
    return {"source": "none", "weeks": plan, "mean": None, "lower": None, "upper": None, "total": None,
            "hash": "none", "note": "no forecast: media effects only"}


def make(spec: dict[str, Any], plan: list[str], template: list[str], weeks: list[str], kpi: dict[str, float],
         step31: dict | None) -> dict[str, Any]:
    src = (spec or {}).get("source", "builtin")
    if src == "builtin":
        return builtin(plan, template, weeks, kpi, spec.get("growth"), step31)
    if src == "csv":
        if not spec.get("csv"):
            raise ForecastError("Upload the forecast CSV.")
        return from_csv(spec["csv"], plan)
    if src == "total":
        return from_total(spec.get("total"), plan, template, kpi)
    if src == "none":
        return none(plan)
    raise ForecastError(f"Unknown forecast source '{src}'; use one of {', '.join(SOURCES)}.")
