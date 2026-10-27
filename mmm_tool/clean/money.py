"""Step 13: normalise money.

  Units      money columns named in crore, lakh or million are converted to ₹
             when the money columns don't all use the same unit.
  Currency   warns when cells carry different currency symbols.
  Inflation  optional: restate spend and revenue in one base year's prices,
             using an annual inflation rate the user gives.
  Fixed value per KPI unit (Phase A step 5) becomes a revenue_per_kpi column,
             because Meridian reads it from a column.
"""
from __future__ import annotations

from ..ingest.keywords import load_keywords, words_of
from ..ingest.types import Answers, Decision, StepResult, Verdict
from .context import SPEND_ROLES, Ctx

STEP = 13
NAME = "Normalise money"


def money_columns(ctx: Ctx) -> list[str]:
    cols = ctx.cols(*SPEND_ROLES, "revenue_per_kpi")
    if ctx.kpi.get("kpi_type") == "revenue":
        cols.append(ctx.kpi_col)
    return cols


def unit_of(col: str) -> tuple[str, float] | None:
    units = load_keywords().money_units
    for word in words_of(col).split():
        if word in units:
            return word, units[word]
    return None


def normalise_money(ctx: Ctx, answers: Answers) -> StepResult:
    df = ctx.df
    money = money_columns(ctx)
    decisions: list[Decision] = []
    warnings: list[str] = []
    notes: list[str] = []

    # Units
    units = {c: unit_of(c) for c in money}
    named = {c: u for c, u in units.items() if u}
    converted = []
    if named and (len(named) < len(money) or len({u[1] for u in named.values()}) > 1):
        d = Decision(
            id="money_units",
            step=STEP,
            question=("Money columns use different units: "
                      + ", ".join(f"{c} in {u[0]}" for c, u in named.items())
                      + "; the rest in ₹. Convert them all to ₹?"),
            options=["convert", "keep"],
            default="convert",
            context={"units": {c: {"unit": u[0], "rupees": u[1]} for c, u in named.items()}},
        )
        decisions.append(d)
        if (answers.ask(d) or "convert") == "convert":
            for c, (unit, rupees) in named.items():
                df[c] = df[c] * rupees
                converted.append({"column": c, "unit": unit, "multiplied_by": rupees})
            notes.append(f"{len(converted)} column(s) converted to ₹")
        else:
            warnings.append("Money columns keep different units; ROI across channels won't be comparable.")

    # Currency symbols seen in step 7
    seen = {c: s for c, s in ctx.currency_symbols.items() if c in money}
    symbols = sorted({x for s in seen.values() for x in s})
    if len(symbols) > 1:
        mix = ", ".join(c + ": " + "/".join(s) for c, s in seen.items())
        warnings.append(f"Money columns mix currencies ({mix}). Convert them to one currency in the export.")

    # Fixed value per KPI unit
    fixed = ctx.kpi.get("revenue_per_kpi_value")
    if fixed and not ctx.kpi.get("revenue_per_kpi_column"):
        df["revenue_per_kpi"] = float(fixed)
        ctx.roles["revenue_per_kpi"] = "revenue_per_kpi"
        ctx.kpi["revenue_per_kpi_column"] = "revenue_per_kpi"
        money = money_columns(ctx)
        notes.append(f"revenue_per_kpi column added at ₹{fixed:,.0f} per unit")

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
    details = {"money_columns": money, "units_converted": converted, "currency_symbols": seen, "inflation": applied, "years": years}
    summary = "; ".join(notes) if notes else f"{len(money)} money column(s) checked; same unit, no inflation adjustment"
    return StepResult(STEP, NAME, Verdict.WARN if warnings else Verdict.PASS, summary + ".", details, warnings, decisions)
