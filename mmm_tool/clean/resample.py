"""Step 9: resample to weeks.

Daily data becomes weekly with one fixed week start: each day goes into the
week that contains it. Weekly data whose weeks start on different weekdays is
aligned to one: each week moves to the nearest week start (a Sunday label
moves forward a day to Monday, not back six), so no week is invented or merged
where the export switched its start day. Columns are combined with the step 8
rules. Partial first and last weeks of daily data can be dropped by choice; by
default they are kept, with a warning, since spend and KPI there are partial
alike. A week inside the period with days missing is never summed as if
complete: its sums are scaled up to a full week (default), interpolated from
the neighbouring weeks, or kept; either way a partial_week_flag marks it.
Monthly data is kept monthly.
"""
from __future__ import annotations

import pandas as pd

from ..ingest.granularity import WEEKDAYS, frequency_of
from ..ingest.types import Answers, Decision, StepResult, Verdict
import numpy as np

from .context import Ctx, aggregate, settings

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

    offset = (df[ctx.date].dt.dayofweek - start_dow) % 7
    if freq == "weekly":  # nearest week start: -3..+3 days
        offset = offset.where(offset <= 3, offset - 7)
    week = df[ctx.date] - pd.to_timedelta(offset, unit="D")
    shifted = int((week != df[ctx.date]).sum())
    days = df.assign(_week=week).groupby(([ctx.geo] if ctx.geo else []) + ["_week"])[ctx.date].nunique()
    work = df.assign(**{ctx.date: week})
    rows_before = len(work)
    if work.duplicated(ctx.keys).any():
        work = aggregate(work, ctx.keys, ctx.rules, ctx.weights)

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
            fix = fix_short_weeks(ctx, answers, work, days, short_inside, decisions)
            if isinstance(fix, StepResult):
                return fix
            work, short_action = fix
            warnings.append(f"{len(short_inside)} week(s) inside the period have days missing ({', '.join(short_inside[:4])}"
                            f"{'…' if len(short_inside) > 4 else ''}): "
                            + {"scale": "sums scaled up to 7 days", "interpolate": "interpolated from the neighbouring weeks",
                               "keep": "kept as they are, so their sums are low"}[short_action] + "; marked in partial_week_flag.")
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
    details = {"short_weeks_action": short_action if short_inside else None, "frequency_in": freq, "week_start": start, "rows_shifted": shifted, "rows_before": rows_before,
               "rows_after": len(ctx.df), "partial_weeks_dropped": dropped, "partial_weeks_kept": kept_partial, "short_weeks": short_inside}
    return StepResult(STEP, NAME, Verdict.WARN if warnings else Verdict.PASS, summary + ".", details, warnings, decisions)


def fix_short_weeks(ctx: Ctx, answers: Answers, work: pd.DataFrame, days: pd.Series, short: list[str],
                    decisions: list[Decision]):
    """Weeks inside the period with fewer days than a full week: scale, interpolate or keep, and flag them."""
    full = int(settings()["short_weeks"]["full_days"])
    per_week = days.groupby(level="_week").min()
    found = {w: int(per_week[pd.Timestamp(w)]) for w in short}
    d = Decision(
        id="short_weeks",
        step=STEP,
        question=("These weeks have days missing: " + ", ".join(f"{w} ({n} of {full} days)" for w, n in found.items())
                  + ". Scale their sums up to a full week, interpolate them from the neighbouring weeks, or keep them as they are?"),
        options=["scale", "interpolate", "keep"],
        default="scale",
        context={"weeks": found},
    )
    decisions.append(d)
    action = answers.ask(d) or "scale"
    if action not in d.options:
        d.answered, d.blocking = False, True
        return StepResult(STEP, NAME, Verdict.NEEDS_INPUT, f"'{action}' isn't an option; choose scale, interpolate or keep.", {}, [], decisions)
    work = work.reset_index(drop=True)
    in_short = work[ctx.date].isin(pd.to_datetime(short))
    summed = [c for c, r in ctx.rules.items() if r == "sum" and c in work.columns and c not in ctx.keys]
    if action == "scale":
        count = days.to_dict()
        keys = zip(work[ctx.geo], work[ctx.date]) if ctx.geo else work[ctx.date]
        n = pd.Series([count.get(k, full) for k in keys], index=work.index, dtype=float)
        factor = (full / n).where(in_short, 1.0)
        for c in summed:
            work[c] = work[c] * factor
    elif action == "interpolate":
        levels = summed + [c for c, r in ctx.rules.items() if r == "mean" and c in work.columns and c not in ctx.keys]
        work.loc[in_short, levels] = np.nan
        groups = work.groupby(ctx.geo).groups.values() if ctx.geo else [work.index]
        for idx in groups:
            work.loc[idx, levels] = work.loc[idx, levels].interpolate(limit_area="inside")
    work["partial_week_flag"] = in_short.astype(float)
    ctx.roles["partial_week_flag"] = "control"
    return work, action
