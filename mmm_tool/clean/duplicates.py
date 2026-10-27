"""Step 8: collapse duplicate weeks.

Rows that are exact copies of another row (every cell the same) are dropped:
adding them up would double spend and KPI. Rows that still share a date (and
region) are split rows and are combined by each column's rule. A column whose
value is the same in every row of every split week (a market figure repeated
on each row) is kept once, whatever its rule. Otherwise:
spend, exposure and KPI are summed, indexes and prices averaged, flags take
the maximum. A national model also adds its regions up here, with
population summed across regions. The user picks the rule for any column the
tool can't classify; the run is blocked until every column it needs has one.
"""
from __future__ import annotations

from ..ingest.types import Answers, Decision, StepResult, Verdict
import pandas as pd

from .context import RULES, Ctx, aggregate, default_rule, per_unit_weights

STEP = 8
NAME = "Collapse duplicate weeks"


def rules_needed(ctx: Ctx, df: pd.DataFrame) -> list[str]:
    """Why the rules matter for this file (empty if they don't)."""
    why = []
    raw_geo = next((c for c, r in ctx.roles.items() if r == "geo"), None)
    keys = [ctx.date] + ([raw_geo] if raw_geo else [])
    if df.duplicated(keys).any():
        why.append("duplicate rows")
    if raw_geo and not ctx.geo:
        why.append("regions added up to national")
    freq = ctx.granularity.get("frequency")
    if freq == "daily":
        why.append("days combined into weeks")
    elif freq == "weekly" and df[ctx.date].dt.dayofweek.nunique() > 1:
        why.append("weeks aligned to one start day")
    return why


def repeated_columns(df: pd.DataFrame, keys: list[str], cols: list[str]) -> list[str]:
    """Columns whose value is identical across the rows of every split key
    (and not zero or empty everywhere, where every rule gives the same answer)."""
    split = df[df.duplicated(keys, keep=False)]
    if split.empty:
        return []
    g = split.groupby(keys, sort=False)
    out = []
    for c in cols:
        same = g[c].nunique(dropna=False).eq(1).all()
        if same and (split[c].fillna(0) != 0).any():
            out.append(c)
    return out


def collapse_duplicates(ctx: Ctx, answers: Answers) -> StepResult:
    df = ctx.df
    raw_geo = next((c for c, r in ctx.roles.items() if r == "geo"), None)
    keys = [ctx.date] + ([raw_geo] if raw_geo else [])
    value_cols = [c for c in df.columns if c not in keys]

    copies = df.duplicated(keep="first")
    copy_dates = sorted({str(pd.Timestamp(x).date()) for x in df.loc[copies, ctx.date]})
    exact_copies = int(copies.sum())
    if exact_copies:
        df = df[~copies].reset_index(drop=True)
    repeated = repeated_columns(df, keys, value_cols)

    table, unclassified, defaults = [], [], {}
    for col in value_cols:
        rule, reason = default_rule(col, ctx.roles[col], df[col])
        defaults[col] = rule or "mean"
        if col in repeated:
            reason = "same value on every row of each split week: kept once, not " + (rule or "combined")
        elif rule is None:
            unclassified.append(col)
        table.append({"column": col, "role": ctx.roles[col], "rule": rule, "reason": reason,
                      "editable": ctx.roles[col] in ("control", "non_media_treatment")})

    why = rules_needed(ctx, df)
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
    merge_rules = {**rules, **{c: "first" for c in repeated if c not in chosen}}
    for row in table:
        row["applied"] = merge_rules[row["column"]]
    ctx.rules = rules

    if d.pending:
        return StepResult(
            STEP, NAME, Verdict.NEEDS_INPUT,
            f"{len(unclassified)} column(s) need a rule before rows can be merged ({', '.join(why)}): {', '.join(unclassified)}.",
            {"rules": table, "why": why}, [], [d],
        )

    rows_before = len(ctx.df)
    dup_rows = int(df.duplicated(keys).sum())
    dup_keys = int(df[df.duplicated(keys, keep=False)].drop_duplicates(keys).shape[0])
    flags_with_dupes = [c for c in value_cols if rules[c] == "max" and ctx.roles[c] != "population" and dup_rows]
    if dup_rows:
        df = aggregate(df, keys, merge_rules, per_unit_weights(ctx))

    regions_rolled = 0
    if raw_geo and not ctx.geo:
        regions_rolled = int(df[raw_geo].nunique())
        roll = {c: ("sum" if ctx.roles[c] == "population" else rules[c]) for c in value_cols}
        df = aggregate(df.drop(columns=[raw_geo]), [ctx.date], roll, per_unit_weights(ctx))
        ctx.roles[raw_geo] = "excluded"
        for c in [c for c, r in ctx.roles.items() if r == "population"]:
            df = df.drop(columns=[c], errors="ignore")
            ctx.roles[c] = "excluded"
    ctx.df = df.sort_values(ctx.keys).reset_index(drop=True)

    parts = []
    if exact_copies:
        parts.append(f"{exact_copies} exact copy row(s) dropped")
    if dup_rows:
        parts.append(f"{dup_rows} duplicate row(s) on {dup_keys} date(s) merged")
    elif not exact_copies:
        parts.append("no duplicate rows")
    if dup_rows and repeated:
        parts.append(f"{len(repeated)} column(s) repeated on split rows kept once")
    if regions_rolled:
        parts.append(f"{regions_rolled} regions added up to one national series")
    parts.append(f"{rows_before:,} rows → {len(ctx.df):,}")
    warnings = []
    if unclassified and not why:
        warnings.append(f"No rule needed yet for {unclassified}; they are averaged if rows are merged later.")
    details = {"rules": table, "why": why, "duplicate_rows": dup_rows, "duplicate_dates": dup_keys,
               "regions_rolled_up": regions_rolled,
               "exact_copies": exact_copies, "exact_copy_dates": copy_dates, "repeated_columns": repeated, "rows_before": rows_before, "rows_after": len(ctx.df),
               "flags_max": flags_with_dupes}
    return StepResult(STEP, NAME, Verdict.WARN if warnings else Verdict.PASS, "; ".join(parts) + ".", details, warnings, [d])
