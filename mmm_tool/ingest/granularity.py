"""Step 6: detect granularity.

Daily, weekly or monthly; one region or several. With several regions the
user picks a regional (geo) model or one national series. Meridian's geo
model needs a population per region, so regional is the default only when
a population column is set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .types import Answers, Decision, StepResult, Verdict

STEP = 6
NAME = "Detect granularity"
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def frequency_of(dates: pd.Series) -> tuple[str, float]:
    uniq = np.sort(dates.dropna().unique())
    if len(uniq) < 2:
        return "unknown", float("nan")
    gap = float(np.median(np.diff(uniq).astype("timedelta64[D]").astype(float)))
    if gap <= 1.5:
        return "daily", gap
    if 6 <= gap <= 8:
        return "weekly", gap
    if 27 <= gap <= 32:
        return "monthly", gap
    return "irregular", gap


def detect_granularity(df: pd.DataFrame, roles: dict[str, str], answers: Answers) -> tuple[dict, StepResult]:
    date_col = next(c for c, r in roles.items() if r == "date")
    geo_col = next((c for c, r in roles.items() if r == "geo"), None)
    pop_col = next((c for c, r in roles.items() if r == "population"), None)
    dates = df[date_col]
    freq, gap = frequency_of(dates)
    regions = sorted(map(str, df[geo_col].dropna().unique())) if geo_col else []
    warnings: list[str] = []
    decisions: list[Decision] = []
    span_days = (dates.max() - dates.min()).days
    out = {
        "frequency": freq,
        "median_gap_days": gap,
        "periods": int(dates.nunique()),
        "weeks_of_history": round(span_days / 7) + (1 if freq == "weekly" else 0),
        "regions": regions,
        "geo_mode": "national",
    }

    if freq == "weekly":
        starts = dates.dt.dayofweek.value_counts()
        out["week_start"] = WEEKDAYS[int(starts.idxmax())]
        if len(starts) > 1:
            warnings.append(f"Weeks start on {len(starts)} different weekdays; step 9 aligns them to one week start.")
    elif freq == "daily":
        warnings.append("Daily data; step 9 resamples it to weeks.")
    elif freq == "monthly":
        warnings.append("Monthly data gives Meridian about a quarter of the observations of weekly data; expect wide ROI ranges.")
    else:
        warnings.append(f"Dates are irregular (median gap {gap:.0f} days); check the export before going on.")

    if len(regions) > 1:
        default = "regional" if pop_col else "national"
        d = Decision(
            id="geo_mode",
            step=STEP,
            question=f"The data has {len(regions)} regions. Model them together as regions, or add them up into one national series?",
            options=["regional", "national"],
            default=default,
            context={"regions": regions, "has_population": bool(pop_col)},
        )
        decisions.append(d)
        mode = answers.ask(d) or default
        if mode == "regional" and not pop_col:
            d.blocking, d.answered = True, False
            return out, StepResult(
                STEP, NAME, Verdict.NEEDS_INPUT,
                "A regional model needs a population column. Set one in step 3, or choose national.",
                out, warnings, decisions,
            )
        out["geo_mode"] = mode
        per_region = df.groupby(geo_col)[date_col].nunique()
        if per_region.nunique() > 1:
            warnings.append(f"Regions have different numbers of {freq} periods ({per_region.min()}–{per_region.max()}); step 10 labels the gaps.")

    where = f"{len(regions)} regions, modelled {out['geo_mode']}ly" if len(regions) > 1 else "one national series"
    summary = f"{freq.capitalize()}, {out['periods']} periods (~{out['weeks_of_history']} weeks), {where}"
    if "week_start" in out:
        summary += f"; weeks start on {out['week_start']}"
    verdict = Verdict.WARN if warnings else Verdict.PASS
    return out, StepResult(STEP, NAME, verdict, summary + ".", out, warnings, decisions)
