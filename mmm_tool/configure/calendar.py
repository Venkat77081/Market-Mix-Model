"""Step 21: calendar flags.

Adds a 0/1 control column per business event: by default the India
insurance preset (tax-saving season, financial year-end, Diwali and festive
weeks) from calendars.yaml. The user turns events on or off and adds their
own date ranges. A flag that agrees with a column already in the data in at
least 95% of weeks, or that is the same in every week, is not added.
"""
from __future__ import annotations

import re
from datetime import date

import numpy as np
import pandas as pd

from ..ingest.types import Answers, Decision, StepResult, Verdict
from .context import DCtx, load_calendars

STEP = 21
NAME = "Calendar flags"
NAME_OK = re.compile(r"^[a-z][a-z0-9_]{0,40}$")
SAME_EVENT_AGREEMENT = 0.95
PRESET = "india_insurance"


def _days(event: dict, years: range) -> set[date]:
    out: set[date] = set()
    if "months" in event:
        for y in years:
            for m in event["months"]:
                out |= {d.date() for d in pd.date_range(f"{y}-{m:02d}-01", periods=31) if d.month == m}
    if "recurring" in event:
        for y in years:
            out |= {d.date() for d in pd.date_range(f"{y}-{event['recurring']['from']}", f"{y}-{event['recurring']['to']}")}
    for d in event.get("dates", []):
        d = pd.Timestamp(d)
        start = d - pd.Timedelta(weeks=event.get("weeks_before", 0))
        end = d + pd.Timedelta(weeks=event.get("weeks_after", 0))
        out |= {x.date() for x in pd.date_range(start, end)}
    for r in event.get("ranges", []):
        out |= {x.date() for x in pd.date_range(r["start"], r["end"])}
    return out


def week_flags(weeks: pd.DatetimeIndex, event: dict) -> np.ndarray:
    """1 for each week (start + 6 days) that touches the event."""
    years = range(weeks.min().year - 1, weeks.max().year + 2)
    days = _days(event, years)
    return np.array([float(any((w + pd.Timedelta(days=i)).date() in days for i in range(7))) for w in weeks])


def planned_events(values: dict) -> list[dict]:
    """The events step 21 will add, from the answers given so far: the preset
    events switched on (all by default) and any readable custom events. Step 19
    checks spend against these before step 21 runs."""
    preset = load_calendars()["presets"][PRESET]
    events = {e["name"]: e for e in preset["events"]}
    chosen = values.get("calendar_events")
    out = [events[n] for n in (list(events) if chosen is None else chosen) if n in events]
    for c in values.get("custom_events") or []:
        name = str(c.get("name", "")).strip().lower().replace(" ", "_")
        try:
            ranges = [{"start": pd.Timestamp(r["start"]), "end": pd.Timestamp(r["end"])} for r in c.get("ranges", [])]
        except (KeyError, ValueError, TypeError, AttributeError):
            continue
        if NAME_OK.match(name) and ranges and all(r["start"] <= r["end"] for r in ranges):
            out.append({"name": name, "label": c.get("label") or name.replace("_", " "), "ranges": ranges})
    return out


def add_calendar_flags(ctx: DCtx, answers: Answers) -> StepResult:
    cal = load_calendars()
    preset_name = PRESET
    preset = cal["presets"][preset_name]
    events = {e["name"]: e for e in preset["events"]}
    weeks = ctx.weeks

    d_on = Decision(
        id="calendar_events",
        step=STEP,
        question=f"Which {preset['label']} events should be modelled as calendar flags?",
        options=list(events),
        default=list(events),
        context={"events": {n: e["label"] for n, e in events.items()}, "preset": preset_name},
    )
    d_custom = Decision(
        id="custom_events",
        step=STEP,
        question="Add your own events: a name and one or more date ranges (a launch, a price change, a regulator's ruling).",
        options=[],
        default=[],
    )
    chosen = answers.ask(d_on)
    chosen = list(events) if chosen is None else chosen
    custom = answers.ask(d_custom) or []
    decisions = [d_on, d_custom]

    problems = [f"unknown preset event '{n}'" for n in chosen if n not in events]
    todo = [events[n] | {"source": "preset"} for n in chosen if n in events]
    for c in custom:
        name = str(c.get("name", "")).strip().lower().replace(" ", "_")
        if not NAME_OK.match(name):
            problems.append(f"event name '{c.get('name')}' should be letters, digits and _")
            continue
        try:
            ranges = [{"start": pd.Timestamp(r["start"]), "end": pd.Timestamp(r["end"])} for r in c.get("ranges", [])]
        except (KeyError, ValueError, TypeError):
            problems.append(f"'{name}' has a date range that can't be read")
            continue
        if not ranges or any(r["start"] > r["end"] for r in ranges):
            problems.append(f"'{name}' needs date ranges whose start is on or before the end")
            continue
        todo.append({"name": name, "label": c.get("label") or name.replace("_", " "), "ranges": ranges, "source": "yours"})
    if problems:
        d_custom.blocking, d_custom.answered = True, False
        return StepResult(STEP, NAME, Verdict.NEEDS_INPUT, "Calendar events: " + "; ".join(problems) + ".", {}, [], decisions)

    existing = {c: ctx.df.groupby(ctx.date)[c].max().reindex(weeks).to_numpy(dtype=float)
                for c in ctx.df.columns if ctx.roles.get(c) == "control"}
    rows, added, warnings = [], [], []
    for e in todo:
        flag = week_flags(weeks, e)
        col = f"cal_{e['name']}"
        row = {"event": e["name"], "label": e["label"], "source": e["source"], "weeks_on": int(flag.sum()), "column": col}
        # A data column that agrees in nearly every week (built from week starts,
        # say, so a boundary week differs) already carries this event.
        twin = next((c for c, v in existing.items() if np.mean(v == flag) >= SAME_EVENT_AGREEMENT), None)
        if twin:
            row["status"] = f"already in your data as '{twin}'"
        elif flag.sum() in (0, len(flag)):
            row["status"] = "never on in this period" if flag.sum() == 0 else "on in every week"
            warnings.append(f"'{e['name']}' is {row['status']}, so it can't be a control; not added.")
        else:
            by_week = dict(zip(weeks, flag))
            ctx.df[col] = ctx.df[ctx.date].map(by_week).astype(float)
            ctx.roles[col] = "control"
            row["status"] = "added"
            added.append(col)
        row["weeks"] = [str(w.date()) for w, f in zip(weeks, flag) if f]
        rows.append(row)
    ctx.config["calendar"] = {"preset": preset_name, "events": rows, "added": added}
    summary = (f"{len(added)} calendar flag(s) added as controls: {', '.join(added)}" if added
               else "No calendar flags added")
    skipped = [r for r in rows if r["status"].startswith("already")]
    if skipped:
        summary += "; " + ", ".join(f"{r['event']} {r['status']}" for r in skipped)
    return StepResult(STEP, NAME, Verdict.WARN if warnings else Verdict.PASS, summary + ".",
                      {"events": rows, "preset": preset["label"]}, warnings, decisions)
