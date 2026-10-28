"""Step 5: set KPI type.

Revenue if the KPI is money, otherwise non-revenue. For a non-revenue KPI
(new policies) ROI needs the money value of one unit. This step only records
where it comes from; Phase B calculates revenue per KPI once, after cleaning
(clean/revenue.py), so it is never merged, interpolated or outlier-checked
as a ratio:

  kpi            the KPI is revenue: no revenue per KPI
  column         a revenue column for the KPI (premium beside policies):
                 revenue total ÷ KPI total per week
  per_unit_column a value-per-unit column: turned into a revenue total
                 (value × KPI) before rows are combined, then as above
  value          a fixed ₹ value per unit the user types: the same every week
  None           no ₹ value: ROI is reported in KPI units
"""
from __future__ import annotations

import pandas as pd

from .keywords import load_keywords, words_of
from .types import Answers, Decision, StepResult, Verdict

STEP = 5
NAME = "Set KPI type"



def kpi_type_guess(kpi_col: str) -> str:
    """Count words (kpi_count_words in role_keywords.yaml) win over money words."""
    kw, w = load_keywords(), words_of(kpi_col)
    if kw.kpi_count and kw.kpi_count.search(w):
        return "non_revenue"
    return "revenue" if kw.kpi_money and kw.kpi_money.search(w) else "non_revenue"


def choose_revenue_kpi(roles: dict[str, str], answers: Answers, decisions: list[Decision],
                       notes: list[str]) -> str | None:
    """With a "Revenue for KPI" column (premium beside policies), offer:
    count    KPI stays the count, valued at revenue ÷ count per week, calculated after
             cleaning; the total is never passed to Meridian (default)
    revenue  the revenue column becomes the KPI, type revenue
    Returns the forced KPI type (None without a revenue column)."""
    rev = next((c for c, r in roles.items() if r == "revenue_for_kpi"), None)
    if rev is None:
        return None
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
        return "revenue"
    old = [c for c, r in roles.items() if r == "revenue_per_kpi"]
    for c in old:
        roles[c] = "excluded"
    notes.append(f"revenue per {kpi_col} is calculated after cleaning, week by week, as {rev} ÷ {kpi_col}; "
                 f"the total '{rev}' isn't passed to Meridian" + (f"; '{old[0]}' set aside" if old else ""))
    return "non_revenue"


def set_kpi_type(df: pd.DataFrame, roles: dict[str, str], answers: Answers) -> tuple[dict, StepResult, pd.DataFrame]:
    decisions: list[Decision] = []
    notes: list[str] = []
    rev_col = next((c for c, r in roles.items() if r == "revenue_for_kpi"), None)
    forced = choose_revenue_kpi(roles, answers, decisions, notes)
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
    out = {"kpi": kpi_col, "kpi_type": kpi_type, "revenue_source": None, "revenue_column": None,
           "revenue_per_kpi_column": None, "revenue_per_kpi_value": None, "roi_units": "₹ per ₹"}

    if kpi_type == "revenue":
        out["revenue_source"] = "kpi"
        for c in (rpk_col, rev_col if forced is None else None):
            if c and roles.get(c) in ("revenue_per_kpi", "revenue_for_kpi"):
                warnings.append(f"KPI is already money, so '{c}' is not used as revenue per KPI.")
                roles[c] = "excluded"
        summary = f"'{kpi_col}' is revenue; ROI is ₹ returned per ₹ spent."
    elif rev_col and roles.get(rev_col) == "revenue_for_kpi":
        out.update(revenue_source="column", revenue_column=rev_col)
        summary = f"'{kpi_col}' is a count; revenue per {kpi_col} = {rev_col} ÷ {kpi_col}, calculated after cleaning."
    elif rpk_col:
        out.update(revenue_source="per_unit_column", revenue_column=rpk_col)
        summary = (f"'{kpi_col}' is a count; '{rpk_col}' is a value per unit. Phase B turns it into a revenue total "
                   f"before rows are combined and calculates revenue per {kpi_col} after cleaning.")
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
            out.update(revenue_source="value", revenue_per_kpi_value=value)
            summary = f"'{kpi_col}' is a count, valued at a fixed ₹{value:,.0f} per unit for ROI."
        else:
            out["roi_units"] = f"{kpi_col} per ₹"
            warnings.append(f"No value per '{kpi_col}'; ROI is reported as {kpi_col} per ₹ and cost per {kpi_col}.")
            summary = f"'{kpi_col}' is a count with no ₹ value; ROI is reported in KPI units."
    if notes:  # the KPI choice explains itself better than the generic sentence
        summary = f"'{kpi_col}' is {'revenue' if kpi_type == 'revenue' else 'a count'}; " + "; ".join(notes) + "."
    return out, StepResult(STEP, NAME, Verdict.WARN if warnings else Verdict.PASS, summary, out, warnings, decisions), df
