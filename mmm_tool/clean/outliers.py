"""Step 11: flag outliers.

A robust z-score on the KPI (0.6745 × distance from the median ÷ MAD) lists
the weeks above 3.5. For each one the user adds an event flag (the default:
a 0/1 control column for that week, so the model doesn't credit media with
it) or keeps the week as it is. Rows are never deleted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..ingest.types import Answers, Decision, StepResult, Verdict
from .context import Ctx, cell_key

STEP = 11
NAME = "Flag outliers"
THRESHOLD = 3.5


def robust_z(x: pd.Series) -> pd.Series:
    med = x.median()
    mad = (x - med).abs().median()
    if not mad or np.isnan(mad):
        return pd.Series(0.0, index=x.index)
    return 0.6745 * (x - med) / mad


def flag_outliers(ctx: Ctx, answers: Answers) -> StepResult:
    df, kpi = ctx.df, ctx.kpi_col
    skip = (df[kpi] == 0) & (df[kpi].median() > 0)  # zero weeks are step 12's job
    if ctx.imputed_kpi is not None:
        skip |= ctx.imputed_kpi.reindex(df.index, fill_value=False)
    found = []
    for g, idx in ctx.by_geo():
        x = df.loc[idx, kpi][~skip[idx]]
        z = robust_z(x)
        for i in z.index[z.abs() > THRESHOLD]:
            found.append({"key": cell_key(df.at[i, ctx.date], g), "date": str(df.at[i, ctx.date].date()), "geo": g,
                          "value": float(df.at[i, kpi]), "z": round(float(z[i]), 2), "row": int(i)})

    series = df.groupby(ctx.date)[kpi].sum()
    chart = [{"date": str(d.date()), "value": float(v)} for d, v in series.items()]
    if not found:
        return StepResult(STEP, NAME, Verdict.PASS, f"No KPI week has a robust z-score above {THRESHOLD}.",
                          {"outliers": [], "series": chart, "threshold": THRESHOLD})

    d = Decision(
        id="outliers",
        step=STEP,
        question="For each unusual week: add an event flag so media isn't credited with it, or keep it as is?",
        options=["flag", "keep"],
        default={o["key"]: "flag" for o in found},
        context={"outliers": found},
    )
    chosen = {**d.default, **(answers.ask(d) or {})}
    flagged = []
    for o in found:
        o["action"] = chosen.get(o["key"], "flag")
        if o["action"] == "flag":
            name = "event_" + o["date"].replace("-", "_")
            if name not in df.columns:
                ctx.add_control(name, np.zeros(len(df)))
            df.at[o["row"], name] = 1.0
            flagged.append(o["date"])
    ctx.df = df
    summary = f"{len(found)} unusual KPI week(s) (|z| > {THRESHOLD}); {len(set(flagged))} flagged as events, {len(found) - len(flagged)} kept"
    details = {"outliers": found, "series": chart, "threshold": THRESHOLD, "event_columns": sorted({f"event_{x.replace('-', '_')}" for x in flagged})}
    return StepResult(STEP, NAME, Verdict.WARN, summary + ".", details, [], [d])
