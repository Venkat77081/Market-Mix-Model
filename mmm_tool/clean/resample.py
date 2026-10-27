"""Step 9: resample to weeks.

Daily data becomes weekly with one fixed week start; weekly data whose
weeks start on different weekdays is aligned to one. Columns are combined
with the step 8 rules. Partial first and last weeks of daily data can be dropped
by choice; by default they are kept, with a warning, since spend and KPI there
are partial alike. Monthly data is kept monthly.
"""
from __future__ import annotations

import pandas as pd

from ..ingest.granularity import WEEKDAYS, frequency_of
from ..ingest.types import Answers, Decision, StepResult, Verdict
from .context import Ctx, aggregate

STEP = 9
NAME = "Resample to weeks"


def resample_weeks(ctx: Ctx, answers: Answers) -> StepResult:
    df = ctx.df
    freq, _ = frequency_of(df[ctx.date])
    warnings: list[str] = []

    if freq == "monthly":
        ctx.freq = "monthly"
        warnings.append("Monthly data is kept monthly; it can't be split into weeks without inventing numbers.")
        return StepResult(STEP, NAME, Verdict.WARN, "Monthly data, kept as months.", {"frequency": freq}, warnings)
    if freq not in ("daily", "weekly"):
        ctx.freq = "weekly"
        warnings.append(f"Dates are {freq}; they are grouped into weeks, so check the result.")

    detected = WEEKDAYS[int(df[ctx.date].dt.dayofweek.mode()[0])]
    default = detected if freq == "weekly" else "Monday"
    d = Decision(
        id="week_start",
        step=STEP,
        question="Which day does a week start on?",
        options=WEEKDAYS,
        default=default,
        context={"detected": detected, "frequency": freq},
    )
    start = answers.ask(d) or default
    if start not in WEEKDAYS:
        d.answered, d.blocking = False, True
        return StepResult(STEP, NAME, Verdict.NEEDS_INPUT, f"'{start}' isn't a weekday.", {}, [], [d])
    start_dow = WEEKDAYS.index(start)

    week = df[ctx.date] - pd.to_timedelta((df[ctx.date].dt.dayofweek - start_dow) % 7, unit="D")
    shifted = int((week != df[ctx.date]).sum())
    days = df.assign(_week=week).groupby(([ctx.geo] if ctx.geo else []) + ["_week"])[ctx.date].nunique()
    work = df.assign(**{ctx.date: week})
    rows_before = len(work)
    if work.duplicated(ctx.keys).any():
        work = aggregate(work, ctx.keys, ctx.rules)

    decisions = [d]
    dropped: list[str] = []
    kept_partial: dict[str, int] = {}
    short_inside: list[str] = []
    if freq == "daily":
        per_week = days.groupby(level="_week").min()
        edges = {per_week.index.min(), per_week.index.max()}
        partial = per_week[per_week < 7]
        edge_weeks = {str(w.date()): int(n) for w, n in partial.items() if w in edges}
        short_inside = sorted(str(w.date()) for w in partial.index if w not in edges)
        if edge_weeks:
            dp = Decision(
                id="partial_weeks",
                step=STEP,
                question=("The first or last week isn't complete ("
                          + ", ".join(f"{w}: {n} of 7 days" for w, n in edge_weeks.items())
                          + "). Keep it (its spend and KPI are both partial) or drop it?"),
                options=["keep", "drop"],
                default="keep",
                context={"weeks": edge_weeks},
            )
            decisions.append(dp)
            if (answers.ask(dp) or "keep") == "drop":
                dropped = sorted(edge_weeks)
                work = work[~work[ctx.date].isin(pd.to_datetime(dropped))]
            else:
                kept_partial = edge_weeks
                warnings.append("Partial edge week(s) kept: " + ", ".join(f"{w} ({n} days)" for w, n in edge_weeks.items())
                                + "; spend and KPI there cover fewer than 7 days.")
        if short_inside:
            warnings.append(f"{len(short_inside)} week(s) have fewer than 7 days of data, e.g. {short_inside[:3]}; their sums are low.")
    ctx.df = work.sort_values(ctx.keys).reset_index(drop=True)
    ctx.freq = "weekly"
    ctx.granularity["week_start"] = start

    if freq == "daily":
        summary = f"{rows_before:,} days → {len(ctx.df):,} weeks starting {start}"
        if dropped:
            summary += f"; {len(dropped)} partial edge week(s) dropped"
        elif kept_partial:
            summary += f"; {len(kept_partial)} partial edge week(s) kept"
    elif shifted:
        summary = f"{shifted} row(s) moved to a {start} week start; {len(ctx.df):,} weeks"
    else:
        summary = f"Already weekly, starting {start}; nothing to resample"
    details = {"frequency_in": freq, "week_start": start, "rows_shifted": shifted, "rows_before": rows_before,
               "rows_after": len(ctx.df), "partial_weeks_dropped": dropped, "partial_weeks_kept": kept_partial, "short_weeks": short_inside}
    return StepResult(STEP, NAME, Verdict.WARN if warnings else Verdict.PASS, summary + ".", details, warnings, decisions)
