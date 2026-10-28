"""Step 13: normalise money.

  Level shift a money column whose level changes at one week by close to a
             power of 10 (thousands for a year, then whole units) is offered a
             conversion of the lower-valued weeks. Thresholds in cleaning.yaml.
  Units      money columns named in a unit word (million, or crore and lakh with the india
             keyword pack) are converted to whole currency units
             when the money columns don't all use the same unit.
  Currency   warns when cells carry different currency symbols.
  Inflation  optional: restate spend and revenue in one base year's prices,
             using an annual inflation rate the user gives.
Revenue per KPI is not touched here: it is calculated once, after this step,
from the cleaned revenue and KPI columns (revenue.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import currency as money_fmt
from ..ingest.keywords import load_keywords, words_of
from ..ingest.types import Answers, Decision, StepResult, Verdict
from .context import SPEND_ROLES, Ctx, settings

STEP = 13
NAME = "Normalise money"


def money_columns(ctx: Ctx) -> list[str]:
    cols = ctx.cols(*SPEND_ROLES, "revenue_for_kpi")
    if ctx.kpi.get("kpi_type") == "revenue":
        cols.append(ctx.kpi_col)
    return cols


def unit_of(col: str) -> tuple[str, float] | None:
    units = load_keywords().money_units
    for word in words_of(col).split():
        if word in units:
            return word, units[word]
    return None


def level_shift(s: pd.Series, dates: pd.Series) -> dict | None:
    """One change point where the column's level moves by close to a power of
    10 and the two sides don't overlap. Returns the change and which side is low."""
    cfg = settings()["level_shift"]
    m, tol, min_power = int(cfg["min_weeks"]), float(cfg["tolerance"]), int(cfg["min_power"])
    pos = s > 0
    lx = np.log10(s[pos].to_numpy(dtype=float))
    d = dates[pos].to_numpy()
    n = len(lx)
    if n < 2 * m:
        return None
    best, cut = 0.0, None
    for t in range(m, n - m + 1):
        diff = float(np.median(lx[t:]) - np.median(lx[:t]))
        if abs(diff) > abs(best):
            best, cut = diff, t
    power = round(best)
    if cut is None or abs(power) < min_power or abs(best - power) > tol:
        return None
    # Noise moves the best median split by a week or so; put the change where
    # the most weeks fall on their own side of the midpoint between the levels.
    mid = (np.median(lx[:cut]) + np.median(lx[cut:])) / 2
    low = (lx < mid) if power > 0 else (lx >= mid)
    score = [int(low[:t].sum() + (~low[t:]).sum()) for t in range(m, n - m + 1)]
    cut = m + int(np.argmax(score))
    lo, hi = (lx[:cut], lx[cut:]) if power > 0 else (lx[cut:], lx[:cut])
    if np.quantile(lo, 0.9) >= np.quantile(hi, 0.1):
        return None
    return {"change": pd.Timestamp(d[cut]), "power": abs(power), "factor": 10 ** abs(power), "low_side": "before" if power > 0 else "after",
            "median_before": float(10 ** np.median(lx[:cut])), "median_after": float(10 ** np.median(lx[cut:])), "ratio": float(10 ** abs(best))}


def unit_named(factor: float) -> str | None:
    names = [w for w, v in load_keywords().money_units.items() if v == factor]
    return max(names, key=len) if names else None


def normalise_money(ctx: Ctx, answers: Answers) -> StepResult:
    df = ctx.df
    money = money_columns(ctx)
    decisions: list[Decision] = []
    warnings: list[str] = []
    notes: list[str] = []

    # Level shifts
    shifts = {}
    for c in money:
        for g, idx in ctx.by_geo():
            found = level_shift(df.loc[idx, c], df.loc[idx, ctx.date])
            if found:
                shifts[c if g is None else f"{c}|{g}"] = {**found, "column": c, "geo": g, "rows": idx}
    shifted = []
    if shifts:
        def say(k, v):
            unit = unit_named(v["factor"])
            side = f"before {v['change'].date()}" if v["low_side"] == "before" else f"from {v['change'].date()}"
            return (f"'{k}' jumps about {v['ratio']:,.0f}x at {v['change'].date()} (median {v['median_before']:,.4g} before, "
                    f"{v['median_after']:,.4g} after), close to {v['factor']:,}x" + (f", one {unit}" if unit else "")
                    + f": multiply the weeks {side} by {v['factor']:,}")
        d = Decision(
            id="level_shifts",
            step=STEP,
            question=("A money column changes level by a power of 10 at one week, which is usually a change of unit, not "
                      "of spend: " + "; ".join(say(k, v) for k, v in shifts.items()) + ". Convert, or keep as recorded?"),
            options=["convert", "keep"],
            default={k: "convert" for k in shifts},
            context={"shifts": {k: {x: (str(y.date()) if isinstance(y, pd.Timestamp) else y) for x, y in v.items() if x != "rows"}
                                for k, v in shifts.items()}},
        )
        decisions.append(d)
        chosen = answers.ask(d) or {}
        if isinstance(chosen, str):
            chosen = {k: chosen for k in shifts}
        for k, v in shifts.items():
            if chosen.get(k, "convert") != "convert":
                warnings.append(f"'{k}' keeps its level change at {v['change'].date()}; the model will read it as a real change in spend.")
                continue
            dates = df.loc[v["rows"], ctx.date]
            side = dates < v["change"] if v["low_side"] == "before" else dates >= v["change"]
            rows = v["rows"][side.to_numpy()]
            df.loc[rows, v["column"]] = df.loc[rows, v["column"]] * v["factor"]
            shifted.append({"column": v["column"], "geo": v["geo"], "weeks": int(len(rows)), "multiplied_by": v["factor"],
                            "change": str(v["change"].date())})
        if shifted:
            done = ", ".join(f"{x['column']} ×{x['multiplied_by']:,}" for x in shifted)
            notes.append(f"{len(shifted)} level shift(s) converted ({done})")

    # Units
    units = {c: unit_of(c) for c in money}
    unit_word = money_fmt.of(ctx.kpi)["unit"]
    named = {c: u for c, u in units.items() if u}
    converted = []
    if named and (len(named) < len(money) or len({u[1] for u in named.values()}) > 1):
        d = Decision(
            id="money_units",
            step=STEP,
            question=("Money columns use different units: "
                      + ", ".join(f"{c} in {u[0]}" for c, u in named.items())
                      + f"; the rest in whole {unit_word}s. Convert them all to whole {unit_word}s?"),
            options=["convert", "keep"],
            default="convert",
            context={"units": {c: {"unit": u[0], "multiplier": u[1]} for c, u in named.items()}},
        )
        decisions.append(d)
        if (answers.ask(d) or "convert") == "convert":
            for c, (unit, mult) in named.items():
                df[c] = df[c] * mult
                converted.append({"column": c, "unit": unit, "multiplied_by": mult})
            notes.append(f"{len(converted)} column(s) converted to whole {unit_word}s")
        else:
            warnings.append("Money columns keep different units; ROI across channels won't be comparable.")

    # Currency symbols seen in step 7
    seen = {c: s for c, s in ctx.currency_symbols.items() if c in money}
    symbols = sorted({x for s in seen.values() for x in s})
    if len(symbols) > 1:
        mix = ", ".join(c + ": " + "/".join(s) for c, s in seen.items())
        warnings.append(f"Money columns mix currencies ({mix}). Convert them to one currency in the export.")

    # Inflation
    years = sorted(df[ctx.date].dt.year.unique().tolist())
    d = Decision(
        id="inflation",
        step=STEP,
        question="Restate money in one year's prices? Give a base year and an annual inflation rate, or leave it as recorded.",
        options=["none"],
        default="none",
        context={"years": years, "columns": money},
    )
    decisions.append(d)
    infl = answers.ask(d) or "none"
    applied = None
    if isinstance(infl, dict):
        try:
            base, rate = int(infl["base_year"]), float(infl["annual_rate_pct"])
        except (KeyError, TypeError, ValueError):
            d.answered, d.blocking = False, True
            return StepResult(STEP, NAME, Verdict.NEEDS_INPUT, "Inflation needs base_year and annual_rate_pct.", {}, warnings, decisions)
        if not (0 <= rate <= 50):
            d.answered, d.blocking = False, True
            return StepResult(STEP, NAME, Verdict.NEEDS_INPUT, "The inflation rate should be between 0 and 50% a year.", {}, warnings, decisions)
        factor = (1 + rate / 100) ** (base - df[ctx.date].dt.year)
        for c in money:
            df[c] = df[c] * factor
        applied = {"base_year": base, "annual_rate_pct": rate, "factor_range": [round(float(factor.min()), 3), round(float(factor.max()), 3)]}
        notes.append(f"money restated in {base} prices at {rate:g}% a year")

    ctx.df = df
    details = {"money_columns": money, "level_shifts": shifted, "units_converted": converted, "currency_symbols": seen, "inflation": applied, "years": years}
    summary = "; ".join(notes) if notes else f"{len(money)} money column(s) checked; same unit, no inflation adjustment"
    return StepResult(STEP, NAME, Verdict.WARN if warnings else Verdict.PASS, summary + ".", details, warnings, decisions)
