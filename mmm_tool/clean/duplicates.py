"""Step 8: duplicate rows, split weeks and revisions.

Rows are grouped by date, or by date + region when the file has a region
column: one row per region for a date is normal and never a split.

  Exact copies    every value equal after rounding to 6 significant digits:
                  the copies are dropped, never added up. The one exception:
                  identical rows that together make one normal week (below)
                  are an even split, as two halves of a week look alike.
  Multi-row week  any other date (+ region) with more than one row. It is
                  judged against a normal week, the median of each spend and
                  KPI column over the 4 single-row weeks on each side (same
                  region; other multi-row weeks left out). The KPI decides, as
                  it moves least week to week; total paid spend decides when
                  the KPI can't:
                    split     the rows add up to about one normal week (within
                              30%): combined by each column's rule, below
                    revision  each row is about a full normal week: asked, Keep
                              the later row by default (never added up unasked)
                    unclear   anything else: asked
Rules for split rows come from the step 3 role and the values only: spend,
KPI, the revenue column and exposure add up; reach is a choice (Maximum by
default: audiences overlap); frequency is averaged weighted by impressions
(by reach without them); 0/1 columns take the maximum; any other column is
kept when it is the same on every row of every split week, and is a choice
(Average by default) when it differs in at least one. The rule shown is the
rule applied. Thresholds are in cleaning.yaml (multi_row_weeks).

A national model built from a regional file also adds its regions up here,
population summed across regions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..ingest.types import Answers, Decision, StepResult, Verdict
from .context import RULES, Ctx, aggregate, cell_key, is_binary, settings

STEP = 8
NAME = "Collapse duplicate weeks"

ADD_UP = {"kpi": "the KPI adds up", "revenue_for_kpi": "revenue adds up", "media_spend": "spend adds up",
          "rf_spend": "spend adds up", "media": "impressions, GRPs and clicks add up",
          "organic_media": "exposure adds up"}
REACH_HINT = "true combined reach is between the larger row and the sum (audiences overlap)"
OTHER_HINT = ("Add up if each row covers part of the total (e.g. two sources); "
              "Average if each row reports the same figure")
ACTIONS = {"later": "Keep the later row", "sum": "Add up", "mean": "Average"}
RULE_WORDS = {"sum": "added up", "mean": "averaged", "max": "maximum", "first": "kept (same on every row)"}


def _cfg() -> dict:
    return settings()["multi_row_weeks"]


def sig_round(s: pd.Series, digits: int) -> pd.Series:
    """Numbers rounded to `digits` significant digits; text and blanks unchanged."""
    if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return s
    x = s.to_numpy(dtype=float)
    out = x.copy()
    ok = np.isfinite(x) & (x != 0)
    e = np.floor(np.log10(np.abs(x[ok])))
    out[ok] = np.round(x[ok] / 10 ** e, digits - 1) * 10 ** e
    return pd.Series(out, index=s.index, name=s.name)


def rules_needed(ctx: Ctx, df: pd.DataFrame, raw_geo: str | None) -> list[str]:
    """Why rows get combined after this step too (resampling, regions to national)."""
    why = []
    if raw_geo and not ctx.geo:
        why.append("regions added up to national")
    freq = ctx.granularity.get("frequency")
    if freq == "daily":
        why.append("days combined into weeks")
    elif freq == "weekly" and df[ctx.date].dt.dayofweek.nunique() > 1:
        why.append("weeks aligned to one start day")
    return why


def _normal(single: pd.DataFrame, date, geo, raw_geo: str | None, cols: list[str], n: int) -> pd.Series:
    """Median of each column over the n single-row periods before and after `date` (same region)."""
    s = single if raw_geo is None else single[single[raw_geo] == geo]
    before = s[s["_date"] < date].tail(n)
    after = s[s["_date"] > date].head(n)
    return pd.concat([before, after])[cols].median()


def _reading(rows: pd.Series | pd.DataFrame, normal: float, tol: float) -> dict | None:
    """Sum and per-row ratios of one series (the KPI, or total paid spend) to its normal week."""
    if not (normal == normal and normal > 0) or rows.isna().all():
        return None
    vals = rows.astype(float)
    sum_ratio = float(vals.sum() / normal)
    row_ratios = [float(v / normal) for v in vals.fillna(0)]
    kind = ("split" if abs(sum_ratio - 1) <= tol
            else "revision" if all(abs(x - 1) <= tol for x in row_ratios) else "unclear")
    return {"kind": kind, "sum_ratio": sum_ratio, "row_ratios": row_ratios}


def judge(rows: pd.DataFrame, normal: pd.Series, kpi: str, spend: list[str], tol: float,
          digits: int = 6) -> tuple[str, dict, dict]:
    """(kind, KPI reading, spend reading). The KPI decides: it moves least from week to week,
    while spend runs in flights. Total paid spend decides only when the KPI can't. A KPI split
    whose rows all carry the same full week of spend (repeated on each row, not split) is
    asked about: adding it up would double the spend."""
    k = _reading(rows[kpi], normal.get(kpi, float("nan")), tol) if kpi in rows else None
    sp = [c for c in spend if pd.notna(normal.get(c))]
    s = _reading(rows[sp].sum(axis=1, min_count=1), float(normal[sp].sum()), tol) if sp else None
    if k is None:
        return (s["kind"] if s else "unclear"), k, s
    repeated = bool(sp) and (rows[sp].fillna(0).sum(axis=1) > 0).all() and \
        all(sig_round(rows[c], digits).nunique(dropna=False) == 1 for c in sp)
    if k["kind"] == "split" and repeated and (s is None or s["kind"] != "split"):
        return "unclear", k, s
    return k["kind"], k, s


def _suggest(kind: str, reading: dict | None) -> str:
    """Keep the later row for a likely revision; for an unclear week, whichever reading is closer:
    the rows' total as one week, or each row as one."""
    if kind == "revision" or reading is None:
        return "later"
    return "sum" if abs(reading["sum_ratio"] - 1) < abs(float(np.mean(reading["row_ratios"])) - 1) else "later"


def _weight_for(ctx: Ctx, freq_col: str, df: pd.DataFrame) -> tuple[str | None, str]:
    ch = next((c for c in ctx.channels if c.frequency == freq_col), None)
    if ch is not None and getattr(ch, "impressions", None) in df.columns:
        return ch.impressions, "impressions"
    if ch is not None and ch.reach in df.columns:
        return ch.reach, "reach"
    return None, ""


def column_rules(ctx: Ctx, df: pd.DataFrame, rounded: pd.DataFrame, value_cols: list[str],
                 add_up: list[dict], needed: bool) -> tuple[list[dict], dict[str, str]]:
    """One row per column: a fixed rule, or a choice where there is a real one."""
    table, weights = [], {}
    for col in value_cols:
        role = ctx.roles.get(col)
        row = {"column": col, "role": role, "choice": False, "options": [], "default": None, "hint": "", "differs": []}
        if role == "excluded":
            row.update(rule="sum", label="Add up", reason="kept only to weight frequency", hidden=True)
        elif role in ADD_UP:
            row.update(rule="sum", label="Add up", reason=ADD_UP[role])
        elif role == "reach":
            row.update(rule="max", label="Maximum", reason="reach", choice=True, options=["max", "sum"], default="max",
                       hint=REACH_HINT)
        elif role == "frequency":
            w, by = _weight_for(ctx, col, df)
            if w:
                weights[col] = w
            row.update(rule="mean", label=f"Average weighted by {by}" if w else "Average",
                       reason=f"weighted by '{w}'" if w else "no impressions or reach to weight by", weight=w)
        elif role == "population":
            row.update(rule="max", label="Maximum", reason="one value per region")
        elif is_binary(df[col]):
            row.update(rule="max", label="Maximum", reason="0/1 column: the maximum, never the sum")
        elif add_up:
            differs = [g["date"] + (f" ({g['geo']})" if g["geo"] is not None else "") for g in add_up
                       if rounded.loc[g["index"], col].nunique(dropna=False) > 1]
            if differs:
                row.update(rule="mean", label="Average", reason="differs between rows", choice=True,
                           options=["sum", "mean", "max"], default="mean", hint=OTHER_HINT, differs=differs)
            else:
                row.update(rule="first", label="Keep (same on every row)", reason="same on every row of every split week",
                           identical=True)
        elif needed:
            row.update(rule="mean", label="Average", reason="rows are combined later", choice=True,
                       options=["sum", "mean", "max"], default="mean", hint=OTHER_HINT)
        else:
            row.update(rule="mean", label="Not needed", reason="no rows are combined")
        table.append(row)
    return table, weights


def collapse_duplicates(ctx: Ctx, answers: Answers) -> StepResult:
    cfg = _cfg()
    digits, tol, n = int(cfg["significant_digits"]), float(cfg["tolerance"]), int(cfg["neighbours"])
    df = ctx.df.reset_index(drop=True)
    raw_geo = next((c for c, r in ctx.roles.items() if r == "geo"), None)
    keys = [ctx.date] + ([raw_geo] if raw_geo else [])
    value_cols = [c for c in df.columns if c not in keys]
    rows_before = len(df)

    spend = [c for c in ctx.cols("media_spend", "rf_spend") if c in df.columns]
    judged = [ctx.kpi_col] + spend
    df["_row"] = np.arange(len(df))                                   # file order, for "the later row"
    # The normal week comes from dates with a single row (other multi-row dates left out).
    single = df[~df.duplicated(keys, keep=False)].assign(_date=df[ctx.date]).sort_values(ctx.date)

    def key_of(key):
        return (key[0], key[1]) if raw_geo else ((key[0] if isinstance(key, tuple) else key), None)

    # Exact copies: every value equal after rounding. They are dropped, never added up, unless
    # the identical rows together make one normal week: an even split (two halves look alike).
    rounded = df[value_cols].apply(lambda s: sig_round(s, digits))
    copies = pd.concat([df[keys], rounded], axis=1).duplicated(keep="first")
    even = []
    for key, part in df[df.duplicated(keys, keep=False)].groupby(keys, sort=True):
        if rounded.loc[part.index].duplicated(keep="first").sum() == len(part) - 1:      # every row the same
            date, geo = key_of(key)
            kind, _, _ = judge(part, _normal(single, date, geo, raw_geo, judged, n), ctx.kpi_col, spend, tol, digits)
            if kind == "split":
                copies[part.index] = False
                even.append(str(pd.Timestamp(date).date()) + (f" ({geo})" if geo is not None else ""))
    copy_dates = sorted({str(pd.Timestamp(x).date()) for x in df.loc[copies, ctx.date]})
    exact_copies = int(copies.sum())
    df, rounded = df[~copies].reset_index(drop=True), rounded[~copies].reset_index(drop=True)

    # Multi-row weeks, judged against the normal week around them.
    multi = df.duplicated(keys, keep=False)
    groups = []
    for key, part in df[multi].groupby(keys, sort=True):
        date, geo = key_of(key)
        normal = _normal(single, date, geo, raw_geo, judged, n)
        part = part.sort_values("_row")
        kind, k, sp = judge(part, normal, ctx.kpi_col, spend, tol, digits)
        reading = k or sp
        groups.append({
            "key": cell_key(date, geo), "date": str(pd.Timestamp(date).date()), "geo": geo, "kind": kind,
            "index": list(part.index), "kpi": k, "spend": sp, "basis": "kpi" if k else "spend" if sp else None,
            "sum_ratio": reading["sum_ratio"] if reading else None,
            "row_ratios": reading["row_ratios"] if reading else [],
            "rows": [{"order": int(r["_row"]), "values": {c: _num(r[c]) for c in judged}} for _, r in part.iterrows()],
            "normal": {c: _num(normal.get(c)) for c in judged}, "suggest": _suggest(kind, reading),
        })

    asked = [g for g in groups if g["kind"] != "split"]
    d_multi = Decision(
        id="multi_row_weeks", step=STEP,
        question=("These dates have more than one row, and the rows don't add up to a normal week. "
                  "Keep the later row, add them up, or average them?"),
        options=list(ACTIONS), default={g["key"]: g["suggest"] for g in asked}, blocking=bool(asked),
        context={"labels": ACTIONS, "tolerance": tol,
                 "weeks": [{k: v for k, v in g.items() if k != "index"} for g in asked]},
    )
    decisions = [d_multi] if asked else []
    chosen: dict = {}
    if asked:
        got = answers.ask(d_multi)
        chosen = {g["key"]: got for g in asked} if isinstance(got, str) else dict(got or {})
        bad = {k: v for k, v in chosen.items() if k not in d_multi.default or v not in ACTIONS}
        if bad:
            d_multi.answered = False
            return StepResult(STEP, NAME, Verdict.NEEDS_INPUT,
                              f"Unknown date or answer in {bad}; answer later, sum or mean for {sorted(d_multi.default)}.",
                              {"groups": d_multi.context["weeks"]}, [], decisions)
    for g in groups:
        g["action"] = "sum" if g["kind"] == "split" else chosen.get(g["key"], g["suggest"])
        g["source"] = "tool" if g["kind"] == "split" else ("user" if g["key"] in chosen and "multi_row_weeks" in answers.values else "default")

    # Rules for the rows that get added up.
    add_up = [g for g in groups if g["action"] == "sum"]
    why = rules_needed(ctx, df, raw_geo)
    table, weights = column_rules(ctx, df, rounded, value_cols, add_up, bool(why))
    choices = {r["column"]: r for r in table if r["choice"]}
    overridable = {r["column"]: r for r in table if r["choice"] or r.get("identical")}
    d_rules = Decision(
        id="aggregation_rules", step=STEP,
        question="How should these columns be combined when rows are merged?",
        options=list(RULES), default={c: r["default"] for c, r in choices.items()},
        context={"choices": list(choices), "unclassified": [c for c, r in choices.items() if r["role"] != "reach"],
                 "why": why + (["split weeks"] if add_up else [])},
    )
    if choices or answers.values.get("aggregation_rules"):
        decisions.append(d_rules)
    picked = (answers.ask(d_rules) or {}) if d_rules in decisions else {}
    bad = {c: r for c, r in picked.items()
           if c not in overridable or r not in (overridable[c]["options"] or ["sum", "mean", "max"])}
    if bad:
        d_rules.answered, d_rules.blocking = False, True
        return StepResult(STEP, NAME, Verdict.NEEDS_INPUT,
                          f"Can't use {bad}: a rule can be chosen only for {sorted(overridable)}.",
                          {"rules": _public(table), "groups": _public_groups(groups)}, [], decisions)
    warnings: list[str] = []
    for row in table:
        c = row["column"]
        if c in picked:
            row["rule"], row["label"], row["source"] = picked[c], {"sum": "Add up", "mean": "Average", "max": "Maximum"}[picked[c]], "user"
            if row.get("identical") and picked[c] == "sum":
                row["doubles"] = True
                warnings.append(f"'{c}' is the same on every row of each split week; adding it up doubles it.")
        row["applied"] = row["rule"]

    if d_multi.pending:
        return StepResult(STEP, NAME, Verdict.NEEDS_INPUT,
                          f"{len(asked)} date(s) have rows that don't add up to a normal week: "
                          + ", ".join(g["date"] + (f" ({g['geo']})" if g["geo"] is not None else "") for g in asked) + ".",
                          {"rules": _public(table), "groups": _public_groups(groups), "why": why, "tolerance": tol,
                           "exact_copies": exact_copies, "exact_copy_dates": copy_dates, "rows_before": rows_before,
                           "rows_after": None}, warnings, decisions)

    # Combine.
    merge_rules = {r["column"]: r["rule"] for r in table}
    later = [max(g["index"], key=lambda i: df.at[i, "_row"]) for g in groups if g["action"] == "later"]
    by_action = lambda a: [i for g in groups if g["action"] == a for i in g["index"]]  # noqa: E731
    parts = [df[~multi], df.loc[later]]
    if add_up:
        parts.append(aggregate(df.loc[by_action("sum")].drop(columns="_row"), keys, merge_rules, weights))
    if by_action("mean"):
        mean_rules = {c: ("max" if is_binary(df[c]) else "mean") for c in value_cols}
        parts.append(aggregate(df.loc[by_action("mean")].drop(columns="_row"), keys, mean_rules))
    df = pd.concat([p.drop(columns="_row", errors="ignore") for p in parts], ignore_index=True)

    # Later combining (days into weeks, regions into national) uses the same rules; a column
    # kept because split rows repeat it is averaged there.
    ctx.rules = {c: ("mean" if r == "first" else r) for c, r in merge_rules.items()}
    ctx.weights = weights
    regions_rolled = 0
    if raw_geo and not ctx.geo:
        regions_rolled = int(df[raw_geo].nunique())
        roll = {c: ("sum" if ctx.roles.get(c) == "population" else ctx.rules[c]) for c in value_cols}
        df = aggregate(df.drop(columns=[raw_geo]), [ctx.date], roll, weights)
        ctx.roles[raw_geo] = "excluded"
        for c in [c for c, r in ctx.roles.items() if r == "population"]:
            df = df.drop(columns=[c], errors="ignore")
            ctx.roles[c] = "excluded"
    ctx.df = df.sort_values(ctx.keys).reset_index(drop=True)

    # Decision log: what the tool did on its own, and the choices.
    def said(g: dict) -> str:
        where = g["date"] + (f" ({g['geo']})" if g["geo"] is not None else "")
        kind = {"split": "split", "revision": "likely revision"}.get(g["kind"], "unclear")
        return f"{where}: {kind}, " + ("rows added up" if g["action"] == "sum" else ACTIONS[g["action"]].lower())

    auto = [f"{exact_copies} exact copy row(s) dropped ({', '.join(copy_dates[:6])}{', …' if len(copy_dates) > 6 else ''})"] if exact_copies else []
    auto += [said(g) for g in groups if g["kind"] == "split"]
    if auto:
        answers.note(STEP, "duplicate_rows", auto)
    if asked and d_multi.answered:
        answers.describe(d_multi, [said(g) for g in asked])
    if d_rules.answered:
        mine = set(answers.values.get("aggregation_rules") or {})
        answers.describe(d_rules, [f"{r['column']}: {r['label'].lower()}" for r in table if r["column"] in mine])
        rest = [f"{r['column']}: {r['label'].lower()}" for r in table if r["choice"] and r["column"] not in mine]
        if rest:
            answers.note(STEP, "aggregation_rules_default", rest, source="default")

    splits = [g for g in groups if g["kind"] == "split"]
    merged_away = sum(len(g["index"]) - 1 for g in groups)
    parts_txt = []
    if exact_copies:
        parts_txt.append(f"{exact_copies} exact copy row(s) dropped")
    if splits:
        parts_txt.append(f"{len(splits)} split week(s) added up")
    if asked:
        parts_txt.append(f"{len(asked)} multi-row week(s) resolved as asked")
    if not exact_copies and not groups:
        parts_txt.append("no duplicate rows")
    kept = [r["column"] for r in table if r.get("identical") and r["column"] not in picked]
    if add_up and kept:
        parts_txt.append(f"{len(kept)} column(s) the same on every split row kept once")
    if regions_rolled:
        parts_txt.append(f"{regions_rolled} regions added up to one national series")
    parts_txt.append(f"{rows_before:,} rows → {len(ctx.df):,}")
    details = {
        "rules": _public(table), "groups": _public_groups(groups), "why": why, "exact_copies": exact_copies,
        "exact_copy_dates": copy_dates, "rows_before": rows_before, "rows_after": len(ctx.df),
        "duplicate_rows": merged_away, "duplicate_dates": len(groups), "split_weeks": [g["date"] for g in splits],
        "repeated_columns": kept if add_up else [], "regions_rolled_up": regions_rolled,
        "flags_max": [r["column"] for r in table if r["rule"] == "max" and r["role"] != "population"] if add_up else [],
        "tolerance": tol, "even_splits": even,
    }
    return StepResult(STEP, NAME, Verdict.WARN if warnings else Verdict.PASS, "; ".join(parts_txt) + ".", details, warnings, decisions)


def _num(v):
    return None if v is None or pd.isna(v) else float(v)


def _public(table: list[dict]) -> list[dict]:
    return [r for r in table if not r.get("hidden")]


def _public_groups(groups: list[dict]) -> list[dict]:
    return [{k: v for k, v in g.items() if k != "index"} for g in groups]
