"""Revenue per KPI, calculated once from the cleaned table (after step 13).

A ratio is never combined, interpolated or outlier-checked directly: two
split rows of 100 units at ₹8,000 and 900 units at ₹12,000 are worth
₹11,600 a unit, not the ₹10,000 an average of the two ratios gives. So
Phase B cleans the revenue total like the KPI, and the ratio is made here:

  column           per date (+ region): revenue total ÷ KPI total. Where the KPI is
                   0 the week uses the whole file's total revenue ÷ total KPI, and
                   those weeks are listed in a warning.
  value            the fixed value per unit, every week.
  kpi or none      no revenue per KPI column.

The revenue total itself is dropped: Meridian gets the KPI and revenue per KPI.
"""
from __future__ import annotations

import pandas as pd

from .. import currency as money_fmt
from .context import Ctx, cell_key

NAME = "revenue_per_kpi"


def _name(df: pd.DataFrame) -> str:
    name, i = NAME, 2
    while name in df.columns:
        name, i = f"{NAME}_{i}", i + 1
    return name


def per_unit_to_total(ctx: Ctx) -> str | None:
    """A value-per-unit column becomes a revenue total (value × KPI) before any rows are
    combined, so merged weeks come out as total ÷ total. Runs after step 7 (numbers)."""
    if ctx.kpi.get("revenue_source") != "per_unit_column":
        return None
    col = ctx.kpi.get("revenue_column")
    if col not in ctx.df.columns:
        return None
    total = f"{col}_x_{ctx.kpi_col}"
    ctx.df[total] = ctx.df[col] * ctx.df[ctx.kpi_col]
    ctx.df = ctx.df.drop(columns=col)
    ctx.roles[col], ctx.roles[total] = "excluded", "revenue_for_kpi"
    ctx.kpi.update(revenue_source="column", revenue_column=total, revenue_from_per_unit=col)
    return f"'{col}' × '{ctx.kpi_col}' gives a revenue total, so merged weeks are valued total ÷ total"


def add_revenue_per_kpi(ctx: Ctx) -> tuple[list[str], list[str], dict]:
    """(notes, warnings, details) after the revenue per KPI column is added to ctx.df."""
    src, df = ctx.kpi.get("revenue_source"), ctx.df
    cur = money_fmt.of(ctx.kpi)
    if ctx.kpi.get("kpi_type") == "revenue" or src not in ("column", "value"):
        return [], [], {"source": src}
    name = _name(df)
    if src == "value":
        value = float(ctx.kpi["revenue_per_kpi_value"])
        df[name] = value
        ctx.roles[name] = "revenue_per_kpi"
        ctx.kpi.update(revenue_per_kpi_column=name, revenue_per_kpi_median=value)
        return [f"{name} = {money_fmt.money(value, cur)} every week (the fixed value per unit)"], [], {"source": src, "value": value}

    rev, kpi = ctx.kpi["revenue_column"], ctx.kpi_col
    overall = float(df[rev].sum() / df[kpi].sum()) if df[kpi].sum() > 0 else float("nan")
    zero = df[kpi] <= 0
    df[name] = (df[rev] / df[kpi].where(~zero)).where(~zero, overall)
    zero_weeks = [cell_key(r[ctx.date], r[ctx.geo] if ctx.geo else None) for _, r in df[zero].iterrows()]
    df.drop(columns=rev, inplace=True)                 # never passed to Meridian, only the ratio
    ctx.roles[rev], ctx.roles[name] = "excluded", "revenue_per_kpi"
    median = float(df[name].median())
    ctx.kpi.update(revenue_per_kpi_column=name, revenue_per_kpi_median=median, revenue_per_kpi_overall=overall)
    notes = [f"{name} = {rev} ÷ {kpi} per week, from the cleaned columns (median {money_fmt.money(median, cur)})"]
    warnings = []
    if zero_weeks:
        warnings.append(f"'{kpi}' is 0 in {len(zero_weeks)} week(s), so revenue per unit can't be worked out there; "
                        f"they use the whole file's {money_fmt.money(overall, cur)} (total {rev} ÷ total {kpi}): "
                        + ", ".join(zero_weeks[:10]) + ("…" if len(zero_weeks) > 10 else "") + ".")
    return notes, warnings, {"source": src, "revenue_column": rev, "median": median, "overall": overall,
                             "zero_kpi_weeks": zero_weeks, "column": name}
