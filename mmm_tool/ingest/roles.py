"""Step 3: map column roles.

Suggests a role for each column from its name and values. The user confirms
or changes each one. The run is blocked until a date, a KPI and at least one
paid media channel are set and the mapping has been confirmed.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import pandas as pd

from .types import ROLES, Answers, Decision, StepResult, Verdict
from .values import is_null_token, numeric_share, numeric_view

STEP = 3
NAME = "Map column roles"

# Tokens stripped from a column name to find its channel ("tv_spend" -> "tv").
METRIC_TOKENS = {
    "spend", "spends", "cost", "costs", "investment", "inr", "rs", "cr", "lakh", "lakhs", "amount",
    "grp", "grps", "trp", "trps", "impressions", "impression", "impr", "imps", "clicks", "click",
    "views", "view", "reach", "frequency", "freq", "rf", "sends", "send", "opens", "posts",
    "media", "paid", "weekly", "total",
}
CHANNEL_ALIASES = {"yt": "youtube", "fb": "facebook", "sem": "search", "paid_search": "search", "dooh": "outdoor", "ooh": "outdoor"}

KPI_COUNT = re.compile(r"\b(kpi|policies|policy count|conversions?|leads?|applications?|orders?|units?|sign_?ups?|acquisitions?|customers?|quotes?)\b")
KPI_MONEY = re.compile(r"\b(revenue|sales|gwp|gross written premium|premium income|total premium|gmv|turnover|bookings)\b")
REV_PER_KPI = re.compile(r"(revenue_per|per_policy|per_kpi|per_unit|avg_premium|average_premium|avg_ticket|aov|arpu)")
GEO = re.compile(r"\b(geo|region|state|zone|dma|market|territory|city|province|area|cluster)\b")
POPULATION = re.compile(r"\b(population|pop|households|adult_pop)\b")
SPEND = re.compile(r"\b(spend|spends|cost|costs|investment|budget|outlay)\b")
EXPOSURE = re.compile(r"\b(grps?|trps?|impressions?|impr|imps|clicks?|views?|insertions|spots)\b")
REACH = re.compile(r"\breach\b")
FREQUENCY = re.compile(r"\b(frequency|freq)\b")
ORGANIC = re.compile(r"\b(email|emails|whatsapp|sms|push|newsletter|organic|owned|seo|crm)\b")
NON_MEDIA = re.compile(r"\b(discount|discounts|promo|promotion|offer|cashback|incentive|incentives|commission)\b")
CONTROL = re.compile(r"\b(index|price|rate|competitor|competition|registrations?|holiday|season|festive|diwali|flag|temperature|cpi|inflation|gdp|unemployment|event|covid|lockdown|trend)\b")
MEDIATOR = re.compile(r"\bbrand(ed)?\b.*\b(search|queries|query|volume|clicks|traffic)\b|\b(search|queries)\b.*\bbrand(ed)?\b")


@dataclass
class RoleSuggestion:
    column: str
    role: str
    confidence: str  # high | medium | low
    reason: str
    channel: str | None = None


def tokens(name: str) -> str:
    """Normalise a column name to lowercase words joined by underscores."""
    s = re.sub(r"[₹$€£()\[\]%#]", " ", name)
    s = re.sub(r"([a-z])([A-Z])", r"\1_\2", s).lower()
    return "_".join(re.findall(r"[a-z0-9]+", s))


def _words(name: str) -> str:
    return tokens(name).replace("_", " ")


def channel_of(name: str) -> str:
    parts = [p for p in tokens(name).split("_") if p not in METRIC_TOKENS]
    stem = "_".join(parts) or tokens(name)
    return CHANNEL_ALIASES.get(stem, stem)


def _is_binary(values: pd.Series) -> bool:
    v = values.dropna().unique()
    return len(v) > 0 and set(v) <= {0.0, 1.0}


def suggest_role(col: str, series: pd.Series, date_col: str) -> RoleSuggestion:
    w = _words(col)
    t = tokens(col)
    if col == date_col:
        return RoleSuggestion(col, "date", "high", "parsed as dates in step 2")

    if numeric_share(series) < 0.8:
        distinct = series[~series.map(is_null_token)].nunique()
        if GEO.search(w) or (distinct <= 60 and distinct < 0.2 * len(series)):
            return RoleSuggestion(col, "geo", "high" if GEO.search(w) else "medium", f"text column with {distinct} repeating values")
        return RoleSuggestion(col, "excluded", "low", "text column that isn't a region")

    values = numeric_view(series)
    if POPULATION.search(w):
        return RoleSuggestion(col, "population", "high", "population by name")
    if MEDIATOR.search(w) and not SPEND.search(w):
        return RoleSuggestion(col, "excluded", "medium", "branded search rises because of other marketing; modelling it would steal their credit")
    if REV_PER_KPI.search(t):
        return RoleSuggestion(col, "revenue_per_kpi", "high", "value of one KPI unit")
    if REACH.search(w):
        return RoleSuggestion(col, "reach", "high", "reach by name", channel_of(col))
    if FREQUENCY.search(w):
        return RoleSuggestion(col, "frequency", "high", "frequency by name", channel_of(col))
    if ORGANIC.search(w) and not SPEND.search(w):
        return RoleSuggestion(col, "organic_media", "medium", "unpaid channel by name", channel_of(col))
    if NON_MEDIA.search(w):
        return RoleSuggestion(col, "non_media_treatment", "medium", "a lever you control that isn't media")
    if CONTROL.search(w):
        if _is_binary(values):
            return RoleSuggestion(col, "control", "high", "0/1 flag")
        return RoleSuggestion(col, "control", "high" if "index" in w or "competitor" in w else "medium", "outside factor by name")
    if SPEND.search(w):
        return RoleSuggestion(col, "media_spend", "high", "spend by name", channel_of(col))
    if EXPOSURE.search(w):
        return RoleSuggestion(col, "media", "high", "exposure metric by name", channel_of(col))
    if KPI_COUNT.search(w):
        return RoleSuggestion(col, "kpi", "high", "count of outcomes by name")
    if KPI_MONEY.search(w):
        return RoleSuggestion(col, "kpi", "medium", "money outcome by name")
    if _is_binary(values):
        return RoleSuggestion(col, "control", "medium", "0/1 column, treated as a flag")
    return RoleSuggestion(col, "control", "low", "numeric column with an unrecognised name; assumed an outside factor")


def suggest_roles(df: pd.DataFrame, date_col: str) -> list[RoleSuggestion]:
    out = [suggest_role(col, df[col], date_col) for col in df.columns]

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


def map_roles(df: pd.DataFrame, date_col: str, answers: Answers) -> tuple[dict[str, str] | None, StepResult]:
    suggestions = suggest_roles(df, date_col)
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
            f"{len(roles)} roles suggested ({len(low)} low-confidence). Confirm them to continue.",
            details, warnings, [d],
        )
    by_role = details["by_role"]
    n_media = len(by_role.get("media_spend", [])) + len(by_role.get("rf_spend", []))
    summary = (
        f"KPI '{by_role['kpi'][0]}', {n_media} paid channel spend column(s), "
        f"{len(by_role.get('control', []))} control(s), {len(by_role.get('excluded', []))} excluded."
    )
    return roles, StepResult(STEP, NAME, Verdict.WARN if warnings else Verdict.PASS, summary, details, warnings, [d])
