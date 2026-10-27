"""Step 5: set KPI type.

Revenue if the KPI is money, otherwise non-revenue. For a non-revenue KPI
(new policies) ROI needs the money value of one unit: a revenue-per-KPI
column or a fixed value such as the average premium per policy. Without
either, the run continues and ROI is reported in KPI units.
"""
from __future__ import annotations

import re

import pandas as pd

from .roles import tokens
from .types import Answers, Decision, StepResult, Verdict
from .values import numeric_view

STEP = 5
NAME = "Set KPI type"

MONEY = re.compile(r"(revenue|sales_value|value|premium|gwp|gmv|turnover|amount|inr|rs|bookings|income)")
COUNT = re.compile(r"(policies|policy|count|conversions|leads|applications|orders|units|customers|quotes|signups)")


def kpi_type_guess(kpi_col: str) -> str:
    t = tokens(kpi_col)
    if COUNT.search(t):
        return "non_revenue"
    return "revenue" if MONEY.search(t) else "non_revenue"


def set_kpi_type(df: pd.DataFrame, roles: dict[str, str], answers: Answers) -> tuple[dict, StepResult]:
    kpi_col = next(c for c, r in roles.items() if r == "kpi")
    rpk_col = next((c for c, r in roles.items() if r == "revenue_per_kpi"), None)
    guess = kpi_type_guess(kpi_col)
    d_type = Decision(
        id="kpi_type",
        step=STEP,
        question=f"Is '{kpi_col}' money (revenue) or a count (non-revenue)?",
        options=["revenue", "non_revenue"],
        default=guess,
    )
    decisions = [d_type]
    kpi_type = answers.ask(d_type) or guess
    warnings: list[str] = []
    out = {"kpi": kpi_col, "kpi_type": kpi_type, "revenue_per_kpi_column": None, "revenue_per_kpi_value": None, "roi_units": "₹ per ₹"}

    if kpi_type == "revenue":
        if rpk_col:
            warnings.append(f"KPI is already money, so '{rpk_col}' is not used as revenue per KPI.")
            roles[rpk_col] = "excluded"
        summary = f"'{kpi_col}' is revenue; ROI is ₹ returned per ₹ spent."
    elif rpk_col:
        values = numeric_view(df[rpk_col])
        out["revenue_per_kpi_column"] = rpk_col
        out["revenue_per_kpi_median"] = float(values.median())
        if (values <= 0).any() or values.isna().mean() > 0.2:
            warnings.append(f"'{rpk_col}' has zero, negative or missing values; check it before fitting.")
        summary = f"'{kpi_col}' is a count; '{rpk_col}' (median {values.median():,.0f}) turns it into ₹ for ROI."
    else:
        d_value = Decision(
            id="revenue_per_kpi_value",
            step=STEP,
            question=f"What is one '{kpi_col}' worth in ₹ (for example, average premium per policy)? Leave blank to report ROI in KPI units.",
            options=[],
            default=None,
        )
        decisions.append(d_value)
        value = answers.ask(d_value)
        if value not in (None, ""):
            value = float(value)
            if value <= 0:
                return out, StepResult(STEP, NAME, Verdict.NEEDS_INPUT, "Revenue per KPI must be above zero.", out, warnings, decisions)
            out["revenue_per_kpi_value"] = value
            summary = f"'{kpi_col}' is a count, valued at a fixed ₹{value:,.0f} per unit for ROI."
        else:
            out["roi_units"] = f"{kpi_col} per ₹"
            warnings.append(f"No value per '{kpi_col}'; ROI is reported as {kpi_col} per ₹ and cost per {kpi_col}.")
            summary = f"'{kpi_col}' is a count with no ₹ value; ROI is reported in KPI units."
    return out, StepResult(STEP, NAME, Verdict.WARN if warnings else Verdict.PASS, summary, out, warnings, decisions)
