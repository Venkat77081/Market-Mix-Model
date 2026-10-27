"""Step 3: map column roles.

Suggests a role for each column from its name and values. The user confirms
or changes each one. The run is blocked until a date, a KPI and at least one
paid media channel are set and the mapping has been confirmed.
"""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass

import pandas as pd

from .keywords import Keywords, load_keywords, words_of
from .memory import RoleMemory
from .types import ROLES, Answers, Decision, StepResult, Verdict
from .values import is_null_token, numeric_share, numeric_view

STEP = 3
NAME = "Map column roles"


@dataclass
class RoleSuggestion:
    column: str
    role: str
    confidence: str  # high | medium | low
    reason: str
    channel: str | None = None
    source: str = "built-in words"  # memory | your words | built-in words | values


def tokens(name: str) -> str:
    """Normalise a column name to lowercase words joined by underscores."""
    return words_of(name).replace(" ", "_")


def channel_of(name: str, kw: Keywords | None = None) -> str:
    kw = kw or load_keywords()
    parts = [p for p in words_of(name).split() if p not in kw.strip]
    stem = "_".join(parts) or tokens(name)
    return kw.aliases.get(stem, stem)


def _is_binary(values: pd.Series) -> bool:
    v = values.dropna().unique()
    return len(v) > 0 and set(v) <= {0.0, 1.0}


def _when(ts: float) -> str:
    return time.strftime("%d %b %Y", time.localtime(ts))


def suggest_role(col: str, series: pd.Series, date_col: str, kw: Keywords | None = None,
                 memory: RoleMemory | None = None) -> RoleSuggestion:
    kw = kw or load_keywords()
    if col == date_col:
        return RoleSuggestion(col, "date", "high", "parsed as dates in step 2", source="values")

    remembered = memory.recall(col) if memory else None
    if remembered and remembered["role"] != "date":
        role = remembered["role"]
        channel = channel_of(col, kw) if role in ("media", "media_spend", "reach", "frequency", "rf_spend", "organic_media") else None
        reason = f"you confirmed this on {_when(remembered['confirmed_at'])} ({remembered['file']})"
        return RoleSuggestion(col, role, "high", reason, channel, source="memory")

    if numeric_share(series) < 0.8:
        distinct = series[~series.map(is_null_token)].nunique()
        rule = kw.match(col, text=True)
        if rule is not None:
            return RoleSuggestion(col, "geo", "high", rule.reason, source=f"{rule.origin} words")
        if distinct <= 60 and distinct < 0.2 * len(series):
            return RoleSuggestion(col, "geo", "medium", f"text column with {distinct} repeating values", source="values")
        return RoleSuggestion(col, "excluded", "low", "text column that isn't a region", source="values")

    values = numeric_view(series)
    rule = kw.match(col)
    if rule is not None:
        confidence, reason = rule.confidence, rule.reason
        if rule.role == "control" and _is_binary(values):
            confidence, reason = "high", "0/1 flag"
        channel = channel_of(col, kw) if rule.role in ("media", "media_spend", "reach", "frequency", "rf_spend", "organic_media") else None
        return RoleSuggestion(col, rule.role, confidence, reason, channel, source=f"{rule.origin} words")
    if _is_binary(values):
        return RoleSuggestion(col, "control", "medium", "0/1 column, treated as a flag", source="values")
    return RoleSuggestion(col, "control", "low", "numeric column with an unrecognised name; assumed an outside factor", source="values")


def suggest_roles(df: pd.DataFrame, date_col: str, memory: RoleMemory | None = None) -> list[RoleSuggestion]:
    kw = load_keywords()
    out = [suggest_role(col, df[col], date_col, kw, memory) for col in df.columns]

    geos = [s for s in out if s.role == "geo"]
    for s in geos[1:]:
        s.role, s.confidence = "excluded", "low"
        s.reason = f"second region-like column; '{geos[0].column}' is used as the region"
    for s in out:
        if s.role == "population":
            if not geos:
                s.role, s.confidence, s.reason = "excluded", "medium", "population without a region column isn't used"
            elif df.groupby(geos[0].column)[s.column].nunique().max() > 1:
                s.confidence = "medium"
                s.reason += "; it changes over time within a region (Meridian takes one value per region)"

    # Only one KPI is modelled. Prefer a count (policies) over money (premium),
    # since money can then come in as revenue per KPI.
    kpis = [s for s in out if s.role == "kpi"]
    if len(kpis) > 1:
        keep = next((s for s in kpis if s.confidence == "high"), kpis[0])
        for s in kpis:
            if s is not keep:
                s.role, s.confidence = "excluded", "medium"
                s.reason = f"second outcome column; '{keep.column}' is the KPI and this one is caused by the same marketing"
    return out


def validate_roles(roles: dict[str, str]) -> list[str]:
    problems = []
    unknown = sorted({r for r in roles.values() if r not in ROLES})
    if unknown:
        problems.append(f"Unknown role(s): {unknown}. Use one of {list(ROLES)}.")
    count = {r: sum(1 for v in roles.values() if v == r) for r in ROLES}
    if count["date"] != 1:
        problems.append("Set exactly one date column.")
    if count["kpi"] != 1:
        problems.append("Set exactly one KPI column." if count["kpi"] == 0 else f"{count['kpi']} columns are marked KPI; keep one.")
    if count["media_spend"] + count["rf_spend"] == 0:
        problems.append("Set at least one paid media channel (a spend column).")
    for r in ("geo", "population", "revenue_per_kpi"):
        if count[r] > 1:
            problems.append(f"Only one column can be {r}.")
    if count["population"] and not count["geo"]:
        problems.append("A population column needs a region column.")
    return problems


def map_roles(df: pd.DataFrame, date_col: str, answers: Answers, memory: RoleMemory | None = None,
              source_file: str = "") -> tuple[dict[str, str] | None, StepResult]:
    suggestions = suggest_roles(df, date_col, memory)
    suggested = {s.column: s.role for s in suggestions}
    d = Decision(
        id="roles",
        step=STEP,
        question="Confirm or change the role of each column.",
        options=list(ROLES),
        default=suggested,
        blocking=True,
        context={"suggestions": [asdict(s) for s in suggestions]},
    )
    chosen = answers.ask(d)
    roles = dict(suggested)
    if chosen:
        missing = [c for c in chosen if c not in df.columns]
        if missing:
            return None, StepResult(STEP, NAME, Verdict.FAIL, f"Roles name column(s) not in the file: {missing}.", {}, [], [d])
        roles.update(chosen)

    problems = validate_roles(roles)
    details = {
        "suggestions": [asdict(s) for s in suggestions],
        "remembered": sum(1 for s in suggestions if s.source == "memory"),
        "roles": roles,
        "by_role": {r: [c for c, v in roles.items() if v == r] for r in ROLES if r in roles.values()},
    }
    low = [s.column for s in suggestions if s.confidence == "low" and roles[s.column] == s.role]
    warnings = [f"Low-confidence role(s) kept as suggested: {low}."] if low and d.answered and d.answer == suggested else []

    if problems:
        d.answered = False  # an invalid mapping must be fixed before continuing
        return None, StepResult(STEP, NAME, Verdict.NEEDS_INPUT, " ".join(problems), details, warnings, [d])
    if not d.answered:
        return None, StepResult(
            STEP, NAME, Verdict.NEEDS_INPUT,
            f"{len(roles)} roles suggested ({len(low)} low-confidence"
            + (f", {details['remembered']} from roles you confirmed before" if details["remembered"] else "")
            + "). Confirm them to continue.",
            details, warnings, [d],
        )
    if memory is not None and "roles" in answers.values:
        details["newly_remembered"] = memory.remember(roles, source_file)
    by_role = details["by_role"]
    n_media = len(by_role.get("media_spend", [])) + len(by_role.get("rf_spend", []))
    summary = (
        f"KPI '{by_role['kpi'][0]}', {n_media} paid channel spend column(s), "
        f"{len(by_role.get('control', []))} control(s), {len(by_role.get('excluded', []))} excluded."
    )
    return roles, StepResult(STEP, NAME, Verdict.WARN if warnings else Verdict.PASS, summary, details, warnings, [d])
