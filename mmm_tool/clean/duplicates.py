"""Step 8: collapse duplicate weeks.

Rows that share a date (and region) are combined by each column's rule:
spend, exposure and KPI are summed, indexes and prices averaged, flags take
the maximum. A national model also adds its regions up here, with
population summed across regions. The user picks the rule for any column the
tool can't classify; the run is blocked until every column it needs has one.
"""
from __future__ import annotations

from ..ingest.types import Answers, Decision, StepResult, Verdict
from .context import RULES, Ctx, aggregate, default_rule

STEP = 8
NAME = "Collapse duplicate weeks"


def rules_needed(ctx: Ctx) -> list[str]:
    """Why the rules matter for this file (empty if they don't)."""
    why = []
    raw_geo = next((c for c, r in ctx.roles.items() if r == "geo"), None)
    keys = [ctx.date] + ([raw_geo] if raw_geo else [])
    if ctx.df.duplicated(keys).any():
        why.append("duplicate rows")
    if raw_geo and not ctx.geo:
        why.append("regions added up to national")
    freq = ctx.granularity.get("frequency")
    if freq == "daily":
        why.append("days combined into weeks")
    elif freq == "weekly" and ctx.df[ctx.date].dt.dayofweek.nunique() > 1:
        why.append("weeks aligned to one start day")
    return why


def collapse_duplicates(ctx: Ctx, answers: Answers) -> StepResult:
    df = ctx.df
    raw_geo = next((c for c, r in ctx.roles.items() if r == "geo"), None)
    keys = [ctx.date] + ([raw_geo] if raw_geo else [])
    value_cols = [c for c in df.columns if c not in keys]

    table, unclassified, defaults = [], [], {}
    for col in value_cols:
        rule, reason = default_rule(col, ctx.roles[col], df[col])
        defaults[col] = rule or "mean"
        if rule is None:
            unclassified.append(col)
        table.append({"column": col, "role": ctx.roles[col], "rule": rule, "reason": reason,
                      "editable": ctx.roles[col] in ("control", "non_media_treatment")})

    why = rules_needed(ctx)
    d = Decision(
        id="aggregation_rules",
        step=STEP,
        question="How should these columns be combined when rows are merged: added up, averaged, or the maximum?",
        options=list(RULES),
        default={c: defaults[c] for c in unclassified},
        blocking=bool(why and unclassified),
        context={"unclassified": unclassified, "why": why},
    )
    chosen = answers.ask(d) or {}
    bad = {c: r for c, r in chosen.items() if r not in RULES or c not in value_cols}
    if bad:
        d.answered = False
        return StepResult(STEP, NAME, Verdict.NEEDS_INPUT, f"Unknown column or rule in {bad}; rules are sum, mean or max.",
                          {"rules": table}, [], [d])
    rules = {**defaults, **chosen}
    if raw_geo:
        rules[raw_geo] = "max"
    for row in table:
        row["applied"] = rules[row["column"]]
    ctx.rules = rules

    if d.pending:
        return StepResult(
            STEP, NAME, Verdict.NEEDS_INPUT,
            f"{len(unclassified)} column(s) need a rule before rows can be merged ({', '.join(why)}): {', '.join(unclassified)}.",
            {"rules": table, "why": why}, [], [d],
        )

    rows_before = len(df)
    dup_rows = int(df.duplicated(keys).sum())
    dup_keys = int(df[df.duplicated(keys, keep=False)].drop_duplicates(keys).shape[0])
    flags_with_dupes = [c for c in value_cols if rules[c] == "max" and ctx.roles[c] != "population" and dup_rows]
    if dup_rows:
        df = aggregate(df, keys, rules)

    regions_rolled = 0
    if raw_geo and not ctx.geo:
        regions_rolled = int(df[raw_geo].nunique())
        roll = {c: ("sum" if ctx.roles[c] == "population" else rules[c]) for c in value_cols}
        df = aggregate(df.drop(columns=[raw_geo]), [ctx.date], roll)
        ctx.roles[raw_geo] = "excluded"
        for c in [c for c, r in ctx.roles.items() if r == "population"]:
            df = df.drop(columns=[c], errors="ignore")
            ctx.roles[c] = "excluded"
    ctx.df = df.sort_values(ctx.keys).reset_index(drop=True)

    parts = []
    if dup_rows:
        parts.append(f"{dup_rows} duplicate row(s) on {dup_keys} date(s) merged")
    else:
        parts.append("no duplicate rows")
    if regions_rolled:
        parts.append(f"{regions_rolled} regions added up to one national series")
    parts.append(f"{rows_before:,} rows → {len(ctx.df):,}")
    warnings = []
    if unclassified and not why:
        warnings.append(f"No rule needed yet for {unclassified}; they are averaged if rows are merged later.")
    details = {"rules": table, "why": why, "duplicate_rows": dup_rows, "duplicate_dates": dup_keys,
               "regions_rolled_up": regions_rolled, "rows_before": rows_before, "rows_after": len(ctx.df),
               "flags_max": flags_with_dupes}
    return StepResult(STEP, NAME, Verdict.WARN if warnings else Verdict.PASS, "; ".join(parts) + ".", details, warnings, [d])
