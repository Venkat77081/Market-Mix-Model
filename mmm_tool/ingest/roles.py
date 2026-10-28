"""Step 3: map column roles.

Suggests a role for each column from its name and values. The user confirms
or changes each one. The run is blocked until a date, a KPI and at least one
paid media channel are set and the mapping has been confirmed.
"""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .. import currency as money_fmt
from .keywords import Keywords, load_keywords, words_of
from .memory import RoleMemory
from .types import ROLES, Answers, Decision, StepResult, Verdict
from .wide import resolve_wide_regions
from .values import is_null_token, numeric_share, numeric_view

STEP = 3
NAME = "Map column roles"
YES_NO = {"yes", "no", "y", "n", "true", "false"}


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


def is_row_index(values: pd.Series) -> bool:
    """0, 1, 2… or 1, 2, 3… in order on every row: a row number left by an export."""
    v = values.to_numpy()
    if len(v) < 3 or np.isnan(v).any():
        return False
    return v[0] in (0, 1) and bool((np.diff(v) == 1).all())


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

    present = series[~series.map(is_null_token)].astype(str).str.strip().str.lower()
    if len(present) and present.isin(YES_NO).all():
        rule = kw.match(col)
        role = rule.role if rule is not None and rule.role in ("control", "non_media_treatment") else "control"
        return RoleSuggestion(col, role, "high", "yes/no flag, read as 1/0", source="values")

    if numeric_share(series) < 0.8:
        filled = present.map(len).gt(0).sum() / max(len(series), 1)
        distinct = present.nunique()
        # A region is named on every row and is short; notes are sparse or long.
        if filled < 0.8 or present.map(len).mean() > 40:
            return RoleSuggestion(col, "excluded", "high", f"free text (filled in {filled:.0%} of rows), not a variable", source="values")
        rule = kw.match(col, text=True)
        if rule is not None:
            return RoleSuggestion(col, "geo", "high", rule.reason, source=f"{rule.origin} words")
        if distinct <= 60 and distinct < 0.2 * len(series):
            return RoleSuggestion(col, "geo", "medium", f"text column with {distinct} repeating values", source="values")
        return RoleSuggestion(col, "excluded", "low", "text column that isn't a region", source="values")

    values = numeric_view(series)
    if is_row_index(values):
        return RoleSuggestion(col, "excluded", "high", "row numbers (0, 1, 2… or 1, 2, 3…), not a variable", source="values")
    rule = kw.match(col)
    if rule is not None:
        confidence, reason = rule.confidence, rule.reason
        if rule.role == "control" and _is_binary(values):
            confidence, reason = "high", "0/1 flag"
        channel = channel_of(col, kw) if rule.role in ("media", "media_spend", "reach", "frequency", "rf_spend", "organic_media") else None
        return RoleSuggestion(col, rule.role, confidence, reason, channel, source=f"{rule.origin} words")
    if _is_binary(values):
        return RoleSuggestion(col, "control", "medium", "0/1 column, treated as a flag", source="values")
    # Never default an unknown number to a control: an outcome used as a
    # control absorbs the sales and pushes every media ROI towards zero.
    return RoleSuggestion(col, "needs_role", "low", "numeric column with an unrecognised name; choose its role", source="values")


# Value checks against the chosen KPI (Fix: an outcome must not become a control).
REVENUE_RATIO_CV = 0.15      # column / KPI varies less than this...
REVENUE_STEADIER = 0.75      # ...and clearly less than the column itself varies
OUTCOME_CORRELATION = 0.8    # moves with the KPI this closely: possibly an outcome
CHECKED_ROLES = {"control", "non_media_treatment", "needs_role"}


def _cv(x: pd.Series) -> float:
    """Robust coefficient of variation: 1.4826 x MAD / median, so a few odd
    weeks (an outage, a duplicated row) don't hide an otherwise steady ratio."""
    med = x.median()
    if not med:
        return float("inf")
    return float(1.4826 * (x - med).abs().median() / abs(med))


def value_checks(df: pd.DataFrame, out: list[RoleSuggestion], kpi: RoleSuggestion, second_kpis: set[str]) -> None:
    """Compare every other numeric column with the KPI.

    revenue for KPI: column / KPI is nearly constant (premium = policies x price)
    possible outcome: correlation above 0.8; it can't become a control unconfirmed
    """
    k = numeric_view(df[kpi.column])
    revenue: list[tuple[float, RoleSuggestion, str]] = []
    for s in out:
        if s.column == kpi.column or s.source == "memory" or (s.role not in CHECKED_ROLES and s.column not in second_kpis):
            continue
        x = numeric_view(df[s.column])
        if _is_binary(x):
            continue
        m = (x > 0) & (k > 0)
        if m.sum() >= 10:
            r = x[m] / k[m]
            rcv, ocv = _cv(r), _cv(x[m])
            if rcv < REVENUE_RATIO_CV and rcv < REVENUE_STEADIER * ocv:
                cur = money_fmt.get(money_fmt.detect(df, [s.column]))
                reason = (f"{s.column} ÷ {kpi.column} stays between {money_fmt.money(r.min(), cur)} and {money_fmt.money(r.max(), cur)} "
                          f"(varies {rcv:.0%}): revenue for the KPI. Use it as revenue per KPI = {s.column} ÷ {kpi.column}")
                revenue.append((rcv, s, reason))
                continue
        corr = x.corr(k)
        if corr is not None and corr > OUTCOME_CORRELATION:
            s.role, s.confidence, s.source = "needs_role", "medium", "values"
            s.reason = (f"possible outcome: moves with '{kpi.column}' (correlation {corr:.2f}). "
                        f"Confirm its role; as a control it would absorb the KPI")
        elif s.column in second_kpis:
            s.role, s.confidence, s.source = "needs_role", "low", "values"
            s.reason = f"outcome-like name but it doesn't track '{kpi.column}'; choose its role"
    revenue.sort(key=lambda t: t[0])
    for i, (_, s, reason) in enumerate(revenue):
        if i == 0:
            s.role, s.confidence, s.reason, s.source = "revenue_for_kpi", "high", reason, "values"
        else:
            s.role, s.confidence, s.source = "needs_role", "medium", "values"
            s.reason = f"also tracks '{kpi.column}' like '{revenue[0][1].column}'; choose its role"


def suggest_roles(df: pd.DataFrame, date_col: str, memory: RoleMemory | None = None,
                  wide: dict | None = None) -> list[RoleSuggestion]:
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

    # Region columns kept side by side: never crown one region's outcome.
    region_cols = set()
    if wide and wide.get("action") == "keep":
        region_cols = {c for by_tag in wide["columns"].values() for c in by_tag.values()}
        for s in out:
            if s.role == "kpi" and s.column in region_cols:
                s.role, s.confidence, s.source = "needs_role", "low", "values"
                s.reason = (f"one region's outcome ({', '.join(wide['tags'])} each have one); "
                            "choose one as the KPI or reshape the regions")

    # Only one KPI is modelled. Prefer a count (policies) over money (premium);
    # the other outcome-like columns are then checked against it by value.
    kpis = [s for s in out if s.role == "kpi"]
    if kpis:
        keep = next((s for s in kpis if s.confidence == "high"), kpis[0])
        second = {s.column for s in kpis if s is not keep}
        for s in kpis:
            if s is not keep:
                s.role = "needs_role"
        value_checks(df, out, keep, second)
    return out


def identical_columns(df: pd.DataFrame, date_col: str) -> list[list[str]]:
    """Groups of columns whose values are the same in every row."""
    groups: dict = {}
    for col in df.columns:
        if col == date_col:
            continue
        s = df[col]
        if s.map(is_null_token).all():
            continue
        vals = numeric_view(s) if numeric_share(s) >= 0.8 else s.astype(str).str.strip()
        key = tuple(None if pd.isna(v) else v for v in vals)
        groups.setdefault(key, []).append(col)
    return [g for g in groups.values() if len(g) > 1]


def validate_roles(roles: dict[str, str]) -> list[str]:
    problems = []
    unknown = sorted({r for r in roles.values() if r not in ROLES})
    if unknown:
        problems.append(f"Unknown role(s): {unknown}. Use one of {list(ROLES)}.")
    count = {r: sum(1 for v in roles.values() if v == r) for r in ROLES}
    unassigned = [c for c, r in roles.items() if r == "needs_role"]
    if unassigned:
        problems.append(f"Choose a role for: {', '.join(unassigned)}.")
    if count["date"] != 1:
        problems.append("Set exactly one date column.")
    if count["kpi"] != 1:
        problems.append("Set exactly one KPI column." if count["kpi"] == 0 else f"{count['kpi']} columns are marked KPI; keep one.")
    if count["media_spend"] + count["rf_spend"] == 0:
        problems.append("Set at least one paid media channel (a spend column).")
    for r in ("geo", "population", "revenue_per_kpi", "revenue_for_kpi"):
        if count[r] > 1:
            problems.append(f"Only one column can be {r}.")
    if count["population"] and not count["geo"]:
        problems.append("A population column needs a region column.")
    return problems


def map_roles(df: pd.DataFrame, date_col: str, answers: Answers, memory: RoleMemory | None = None,
              source_file: str = "") -> tuple[dict[str, str] | None, pd.DataFrame, StepResult]:
    """Returns (roles, table, step). The table changes when wide regions are reshaped or summed."""
    # Wide-format regions are settled before any role is suggested.
    df, wide, decisions, warnings = resolve_wide_regions(df, date_col, answers, STEP)
    if wide and wide.get("action") is None:
        pending = wide.get("problem") or (
            f"Region columns found: {', '.join(wide['stems'])} for {', '.join(wide['tags'])}. "
            "Choose how to handle them before roles are suggested.")
        return None, df, StepResult(STEP, NAME, Verdict.NEEDS_INPUT, pending, {"wide_regions": wide}, warnings, decisions)

    suggestions = suggest_roles(df, date_col, memory, wide)

    # Columns identical in every row: keep one of each group.
    for group in identical_columns(df, date_col):
        dd = Decision(
            id=f"identical:{group[0]}",
            step=STEP,
            question=f"{', '.join(group)} have the same value in every row. Which one should be kept?",
            options=group,
            default=group[0],
            blocking=True,
            context={"columns": group},
        )
        decisions.append(dd)
        keep = answers.ask(dd) or group[0]
        if keep not in group:
            dd.answered = False
        for s in suggestions:
            if s.column in group and s.column != keep:
                s.role, s.confidence, s.source = "excluded", "high", "values"
                s.reason = f"identical to '{keep}' in every row"

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
    decisions.append(d)
    chosen = answers.ask(d)
    roles = dict(suggested)
    if chosen:
        missing = [c for c in chosen if c not in df.columns]
        if missing:
            return None, df, StepResult(STEP, NAME, Verdict.FAIL, f"Roles name column(s) not in the file: {missing}.", {}, warnings, decisions)
        roles.update(chosen)

    problems = validate_roles(roles)
    details = {
        "suggestions": [asdict(s) for s in suggestions],
        "remembered": sum(1 for s in suggestions if s.source == "memory"),
        "roles": roles,
        "by_role": {r: [c for c, v in roles.items() if v == r] for r in ROLES if r in roles.values()},
        "wide_regions": wide or None,
    }
    low = [s.column for s in suggestions if s.confidence == "low" and roles[s.column] == s.role]
    if low and d.answered and d.answer == suggested:
        warnings.append(f"Low-confidence role(s) kept as suggested: {low}.")
    pending_other = [x for x in decisions if x is not d and x.pending]

    if problems:
        d.answered = False  # an invalid mapping must be fixed before continuing
        return None, df, StepResult(STEP, NAME, Verdict.NEEDS_INPUT, " ".join(problems), details, warnings, decisions)
    if not d.answered or pending_other:
        return None, df, StepResult(
            STEP, NAME, Verdict.NEEDS_INPUT,
            f"{len(roles)} roles suggested ({len(low)} low-confidence"
            + (f", {details['remembered']} from roles you confirmed before" if details["remembered"] else "")
            + "). Confirm them to continue.",
            details, warnings, decisions,
        )
    if memory is not None and "roles" in answers.values:
        details["newly_remembered"] = memory.remember(roles, source_file)
    by_role = details["by_role"]
    n_media = len(by_role.get("media_spend", [])) + len(by_role.get("rf_spend", []))
    summary = f"KPI '{by_role['kpi'][0]}'"
    if by_role.get("revenue_for_kpi"):
        summary += f" with revenue '{by_role['revenue_for_kpi'][0]}'"
    summary += (f", {n_media} paid channel spend column(s), "
                f"{len(by_role.get('control', []))} control(s), {len(by_role.get('excluded', []))} excluded")
    if wide and wide.get("action") == "reshape":
        summary += f"; regions reshaped into '{wide['region_column']}' ({wide['rows_before']} → {wide['rows_after']} rows)"
    elif wide and wide.get("action") == "sum":
        summary += f"; {len(wide['tags'])} regions added into one national series"
    return roles, df, StepResult(STEP, NAME, Verdict.WARN if warnings else Verdict.PASS, summary + ".", details, warnings, decisions)
