"""Step 5: set KPI type.

Revenue if the KPI is money, otherwise non-revenue. For a non-revenue KPI
(new policies) ROI needs the money value of one unit: a revenue-per-KPI
column or a fixed value such as the average premium per policy. Without
either, the run continues and ROI is reported in KPI units.
"""
from __future__ import annotations

import pandas as pd

from .keywords import load_keywords, words_of
from .types import Answers, Decision, StepResult, Verdict
from .values import numeric_view

STEP = 5
NAME = "Set KPI type"



def kpi_type_guess(kpi_col: str) -> str:
    """Count words (kpi_count_words in role_keywords.yaml) win over money words."""
    kw, w = load_keywords(), words_of(kpi_col)
    if kw.kpi_count and kw.kpi_count.search(w):
        return "non_revenue"
    return "revenue" if kw.kpi_money and kw.kpi_money.search(w) else "non_revenue"


def _derived_name(df: pd.DataFrame) -> str:
    name, i = "revenue_per_kpi", 2
    while name in df.columns:
        name, i = f"revenue_per_kpi_{i}", i + 1
    return name


def choose_revenue_kpi(df: pd.DataFrame, roles: dict[str, str], answers: Answers,
                       decisions: list[Decision], notes: list[str]) -> tuple[pd.DataFrame, str | None]:
    """With a "Revenue for KPI" column (premium beside policies), offer:
    count    KPI stays the count; revenue per KPI = revenue / count, passed to
             Meridian as the per-unit value, never the total (default)
    revenue  the revenue column becomes the KPI, type revenue
    Returns the table and the forced KPI type (None if unchanged)."""
    rev = next((c for c, r in roles.items() if r == "revenue_for_kpi"), None)
    if rev is None:
        return df, None
    kpi_col = next(c for c, r in roles.items() if r == "kpi")
    d = Decision(
        id="kpi_choice",
        step=STEP,
        question=(f"'{rev}' is revenue for '{kpi_col}'. Keep '{kpi_col}' as the KPI and value each one at "
                  f"{rev} ÷ {kpi_col}, or make '{rev}' the KPI?"),
        options=["count", "revenue"],
        default="count",
        context={"kpi": kpi_col, "revenue": rev},
    )
    decisions.append(d)
    choice = answers.ask(d) or "count"
    if choice == "revenue":
        roles[rev], roles[kpi_col] = "kpi", "excluded"
        notes.append(f"'{rev}' is the KPI; '{kpi_col}' is set aside")
        return df, "revenue"
    k = numeric_view(df[kpi_col])
    per_unit = numeric_view(df[rev]) / k.where(k > 0)
    name = _derived_name(df)
    df = df.copy()
    df[name] = per_unit
    old = [c for c, r in roles.items() if r == "revenue_per_kpi"]
    for c in old:
        roles[c] = "excluded"
    roles[name], roles[rev] = "revenue_per_kpi", "excluded"
    notes.append(f"'{name}' = {rev} ÷ {kpi_col} (median ₹{per_unit.median():,.0f}); the total '{rev}' isn't passed to Meridian"
                 + (f"; '{old[0]}' set aside" if old else ""))
    return df, "non_revenue"


def set_kpi_type(df: pd.DataFrame, roles: dict[str, str], answers: Answers) -> tuple[dict, StepResult, pd.DataFrame]:
    decisions: list[Decision] = []
    notes: list[str] = []
    df, forced = choose_revenue_kpi(df, roles, answers, decisions, notes)
    kpi_col = next(c for c, r in roles.items() if r == "kpi")
    rpk_col = next((c for c, r in roles.items() if r == "revenue_per_kpi"), None)
    guess = forced or kpi_type_guess(kpi_col)
    if forced:
        kpi_type = forced
    else:
        d_type = Decision(
            id="kpi_type",
            step=STEP,
            question=f"Is '{kpi_col}' money (revenue) or a count (non-revenue)?",
            options=["revenue", "non_revenue"],
            default=guess,
        )
        decisions.append(d_type)
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
                return out, StepResult(STEP, NAME, Verdict.NEEDS_INPUT, "Revenue per KPI must be above zero.", out, warnings, decisions), df
            out["revenue_per_kpi_value"] = value
            summary = f"'{kpi_col}' is a count, valued at a fixed ₹{value:,.0f} per unit for ROI."
        else:
            out["roi_units"] = f"{kpi_col} per ₹"
            warnings.append(f"No value per '{kpi_col}'; ROI is reported as {kpi_col} per ₹ and cost per {kpi_col}.")
            summary = f"'{kpi_col}' is a count with no ₹ value; ROI is reported in KPI units."
    if notes:  # the KPI choice explains itself better than the generic sentence
        summary = f"'{kpi_col}' is {'revenue' if kpi_type == 'revenue' else 'a count'}; " + "; ".join(notes) + "."
    return out, StepResult(STEP, NAME, Verdict.WARN if warnings else Verdict.PASS, summary, out, warnings, decisions), df
