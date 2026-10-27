"""Step 19: spend timed with seasons.

When a channel spends mostly in the weeks a season flag is on (tax season,
festive weeks), the season and the channel compete for the same sales and the
model can split them either way. Warning only, with a badge on the channel.

Seasons come from two places: 0/1 columns already in the data, and the
calendar events step 21 will add as controls (the preset events still
switched on there, plus custom events). A calendar event that agrees with a
data column in nearly every week is that column, and is checked once.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..ingest.types import Answers, StepResult, Verdict
from .context import SEASON_CORR, CCtx


def _is_flag(s) -> bool:
    v = set(s.dropna().unique())
    return len(v) == 2 and v <= {0.0, 1.0}


STEP = 19
NAME = "Spend timed with seasons"


def calendar_seasons(weeks: pd.DatetimeIndex, data_flags: pd.DataFrame, answers: Answers) -> pd.DataFrame:
    """Weekly 0/1 columns for the calendar events step 21 will add, named as step 21 names them."""
    from ..configure.calendar import SAME_EVENT_AGREEMENT, planned_events, week_flags

    out = {}
    for e in planned_events(answers.values):
        flag = week_flags(weeks, e)
        if flag.sum() in (0, len(flag)):
            continue
        if any(np.mean(data_flags[c].to_numpy() == flag) >= SAME_EVENT_AGREEMENT for c in data_flags.columns):
            continue
        out[f"cal_{e['name']}"] = flag
    return pd.DataFrame(out, index=weeks)


def check_seasons(ctx: CCtx, answers: Answers) -> StepResult:
    data_flags = [c for c in ctx.cols("control", "non_media_treatment") if _is_flag(ctx.df[c]) and c != "kpi_imputed"
                  and not c.startswith("event_") and c != "partial_week_flag"]
    spend = ctx.spend()
    weeks = pd.DatetimeIndex(spend.index)
    season = ctx.weekly_mean(data_flags).round() if data_flags else pd.DataFrame(index=weeks)
    calendar = calendar_seasons(weeks, season, answers)
    season = season.join(calendar)
    flags = list(season.columns)
    found = []
    if flags:
        for ch in spend.columns:
            for f in flags:
                r = spend[ch].corr(season[f])
                if r == r and r > SEASON_CORR:
                    found.append({"channel": ch, "flag": f, "correlation": round(float(r), 3)})
                    ctx.badge(ch, "season_timed")
    for x in found:
        x["source"] = "calendar (added at step 21)" if x["flag"] in calendar.columns else "data"
    warnings = [f"'{x['channel']}' spends mostly when '{x['flag']}'"
                + (" (a calendar flag step 21 adds)" if x["source"] != "data" else "")
                + f" is on (correlation {x['correlation']:.2f}): the season and the channel will compete for the same sales."
                for x in found]
    details = {"flags": flags, "calendar_flags": list(calendar.columns), "pairs": found, "threshold": SEASON_CORR}
    if not flags:
        return StepResult(STEP, NAME, Verdict.PASS, "No season flags in the data or the calendar to check against.", details)
    if not found:
        return StepResult(STEP, NAME, Verdict.PASS, f"No channel's spend follows the {len(flags)} season flag(s).", details)
    return StepResult(STEP, NAME, Verdict.WARN,
                      f"{len(found)} channel(s) timed with a season: "
                      + ", ".join(f"{x['channel']} with {x['flag']}" for x in found) + ".", details, warnings)
