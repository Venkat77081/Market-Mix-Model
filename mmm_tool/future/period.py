"""The plan's weeks and the weeks it copies (no Meridian).

A future period is P weeks on the model's weekly grid (same weekday as the data). Its template
is P weeks of the training data whose week-by-week shape each channel keeps: the same weeks last
year (by default), the most recent P weeks, or a date range of the user's. Meridian also needs the
max_lag weeks just before the plan (the lead-in) so carryover from earlier spend reaches it: the
last real weeks when the plan starts right after the data, else the weeks before the template.
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

SETTINGS = Path(__file__).with_name("future.yaml")
TEMPLATES = ("last_year", "recent", "custom", "upload")


class PeriodError(ValueError):
    pass


def settings() -> dict[str, Any]:
    return yaml.safe_load(SETTINGS.read_text(encoding="utf-8"))


def defaults(weeks: list[str]) -> dict[str, Any]:
    """The plan a new future period starts from: the week after the data, default length."""
    n = int(settings()["plan_weeks"])
    start = pd.Timestamp(weeks[-1]) + timedelta(days=7)
    return {"start": str(start.date()), "weeks": n, "template": {"kind": "last_year"}, "cost": {},
            "forecast": {"source": "builtin"}}


def plan_weeks(start: str, n: int, weeks: list[str]) -> tuple[list[str], list[str]]:
    """(the plan's weeks, warnings). The start must sit on the data's weekly grid."""
    if not 1 <= int(n) <= int(settings()["max_plan_weeks"]):
        raise PeriodError(f"A plan is 1 to {settings()['max_plan_weeks']} weeks long.")
    last, s = pd.Timestamp(weeks[-1]), pd.Timestamp(start)
    gap = (s - last).days
    if gap % 7:
        raise PeriodError(f"The plan must start on the model's weekly grid: a {last.day_name()} "
                          f"(the data's weeks start on {last.day_name()}s).")
    if gap <= 0:
        raise PeriodError(f"The plan must start after the last data week ({last.date()}).")
    warn = []
    if gap // 7 > int(settings()["stale_after_weeks"]):
        warn.append(f"The plan starts {gap // 7} weeks after the last data week: the model may be out of date.")
    return [str((s + timedelta(days=7 * k)).date()) for k in range(int(n))], warn


def template_weeks(kind: str, plan: list[str], weeks: list[str], custom_start: str | None = None) -> list[str]:
    """The training weeks the plan copies, one per plan week."""
    have, n = set(weeks), len(plan)
    if kind in ("last_year", "upload"):
        out = [str((pd.Timestamp(w) - timedelta(days=364)).date()) for w in plan]       # same week of the year
        if all(w in have for w in out):
            return out
        if kind == "upload":
            return weeks[-n:]                      # an uploaded plan only borrows the other inputs' shape
        raise PeriodError("The same weeks last year aren't all in the data; copy the most recent weeks or a date range instead.")
    if kind == "recent":
        if n > len(weeks):
            raise PeriodError(f"The data has {len(weeks)} weeks, fewer than the plan's {n}.")
        return weeks[-n:]
    if kind == "custom":
        if not custom_start or custom_start not in have:
            raise PeriodError("Choose a template start that is one of the data's weeks.")
        i = weeks.index(custom_start)
        if i + n > len(weeks):
            raise PeriodError(f"From {custom_start} the data has only {len(weeks) - i} weeks; the plan needs {n}.")
        return weeks[i:i + n]
    raise PeriodError(f"Unknown template '{kind}'; use one of {', '.join(TEMPLATES)}.")


def lead_in(plan: list[str], template: list[str], weeks: list[str], lag: int) -> tuple[list[str], list[str]]:
    """(the lead-in's dates, the training weeks its values come from). Right after the data: the
    last real weeks. Later: the weeks before the template (repeating its first week if there are none)."""
    start = pd.Timestamp(plan[0])
    dates = [str((start - timedelta(days=7 * (lag - k))).date()) for k in range(lag)]
    if lag == 0:
        return [], []
    if (start - pd.Timestamp(weeks[-1])).days == 7:
        src = weeks[-lag:] if len(weeks) >= lag else [weeks[0]] * (lag - len(weeks)) + weeks
    else:
        i = weeks.index(template[0])
        src = [weeks[max(0, i - lag + k)] for k in range(lag)]
    return dates, src


def resolve(setup: dict[str, Any], weeks: list[str], lag: int) -> dict[str, Any]:
    """A setup's weeks, all worked out: plan, template, lead-in, warnings."""
    plan, warn = plan_weeks(setup["start"], int(setup["weeks"]), weeks)
    t = setup.get("template") or {"kind": "last_year"}
    tmpl = template_weeks(t.get("kind", "last_year"), plan, weeks, t.get("start"))
    lead_dates, lead_src = lead_in(plan, tmpl, weeks, lag)
    return {"plan": plan, "template": tmpl, "lead_dates": lead_dates, "lead_src": lead_src, "warnings": warn}
