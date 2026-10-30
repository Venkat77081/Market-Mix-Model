"""Step 10: label every gap.

Weeks absent from the export are added, then every empty cell is labelled:
not started (before a column's first value), ended (after its last value),
or missing (a hole in between). A zero is a real zero and is never touched.

  KPI at the start or end   nothing on one side to interpolate from: trimmed
                  by default, or the nearest value carried and flagged.
  KPI inside      the user chooses: interpolate and add a kpi_imputed flag
                  (default), or trim to the longest stretch with no KPI gaps.
                  Blocks until answered. A KPI gap is never 0.
  Media           not started or ended is no spend (0). Holes are judged per
                  column: a column that never records a zero and is blank in
                  a large share of its weeks writes "no spend" as a blank, so
                  0 is suggested; blanks in any other column are missing and
                  interpolated. Spend in weeks absent from the export is
                  unknown: interpolated by default, 0 only if the user says the
                  channels were off. Asked; blocks when weeks were absent.
  Everything else interpolated between known values, carried to the edges;
                  0/1 flags are 0 where missing.

"Real zeros" counts only media columns (paid spend, exposure, reach, organic
media) and only zeros the export itself has: never a cell this step filled.
A paid channel that spends in most of its active weeks (always on) but has a
run of 2+ zero weeks is flagged: a real pause (kept, the default) or a data
gap (interpolated, with the channel's exposure columns for the same weeks).
"""
from __future__ import annotations

import pandas as pd

from ..ingest.types import Answers, Decision, StepResult, Verdict
from .context import MEDIA_ROLES, SPEND_ROLES, Ctx, is_binary, settings

STEP = 10
NAME = "Label every gap"


def _calendar(dates: pd.Series, freq: str) -> pd.DatetimeIndex:
    lo, hi = dates.min(), dates.max()
    if freq == "weekly":
        return pd.date_range(lo, hi, freq="7D")
    if freq == "monthly" and (dates.dt.day == 1).all():
        return pd.date_range(lo, hi, freq="MS")
    return pd.DatetimeIndex(sorted(dates.unique()))


def _edges(s: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Masks for (not started, ended, missing in between) on one series."""
    valid = s.notna().to_numpy()
    n = len(s)
    if not valid.any():
        none = pd.Series(False, index=s.index)
        return pd.Series(True, index=s.index), none, none.copy()
    first, last = valid.argmax(), n - 1 - valid[::-1].argmax()
    pos = pd.Series(range(n), index=s.index)
    na = s.isna()
    return na & (pos < first), na & (pos > last), na & (pos > first) & (pos < last)


def _longest_run(ok: pd.Series) -> pd.Series:
    """Mask of the longest run of consecutive True values."""
    best, start, best_span = 0, None, (0, -1)
    vals = ok.to_numpy()
    for i, v in enumerate(list(vals) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > best:
                best, best_span = i - start, (start, i - 1)
            start = None
    keep = pd.Series(False, index=ok.index)
    keep.iloc[best_span[0]: best_span[1] + 1] = True
    return keep


def _label(df: pd.DataFrame, groups, cols) -> dict[str, tuple[pd.Series, pd.Series, pd.Series]]:
    labels = {}
    for col in cols:
        ns, en, mi = (pd.Series(False, index=df.index) for _ in range(3))
        for _, idx in groups:
            a, b, c = _edges(df.loc[idx, col])
            ns[idx], en[idx], mi[idx] = a, b, c
        labels[col] = (ns, en, mi)
    return labels


def media_holes(df: pd.DataFrame, col: str, label, inserted: pd.Series, share: float) -> dict:
    """How one media column's holes should be read: blanks in rows the export
    has, and cells in weeks the export left out."""
    ns, en, mi = label
    blank = mi & ~inserted
    active = ~(ns | en) & ~inserted
    n_blank = int(blank.sum())
    blank_share = n_blank / max(int(active.sum()), 1)
    has_zero = bool((df[col] == 0).any())
    kind = "blank_means_zero" if n_blank and not has_zero and blank_share >= share else "missing"
    return {"blank": blank, "absent": mi & inserted, "kind": kind, "blank_share": blank_share, "has_zero": has_zero}


def zero_runs(values: pd.Series, active: pd.Series, min_run: int) -> list[tuple[int, int]]:
    """(first, last) positions of each run of real zeros at least min_run long, inside the active weeks."""
    z = ((values == 0) & active).to_numpy()
    runs, start = [], None
    for i, v in enumerate(list(z) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_run:
                runs.append((start, i - 1))
            start = None
    return runs


def _span(a: pd.Timestamp, b: pd.Timestamp) -> str:
    """'2024-04-29 → 06-17', the year left off the end when it's the same."""
    return f"{a.date()} → {b.strftime('%m-%d') if a.year == b.year else b.date()}"


def always_on_zeros(df: pd.DataFrame, ctx: Ctx, labels: dict, inserted: pd.Series, groups) -> dict[str, dict]:
    """Paid channels that spend in most active weeks but have runs of real zeros: {spend column: details}."""
    cfg = settings().get("zero_runs") or {}
    share, min_run = float(cfg.get("always_on_share", 0.8)), int(cfg.get("min_run_weeks", 2))
    names = {c.spend: c for c in ctx.channels}
    out = {}
    for col in [c for c in ctx.value_cols if ctx.roles.get(c) in SPEND_ROLES]:
        ns, en, mi = labels[col]
        mask = pd.Series(False, index=df.index)
        spans, spending, active_n = [], 0, 0
        for g, idx in groups:
            vals = df.loc[idx, col]
            active = ~(ns[idx] | en[idx] | inserted[idx]) & vals.notna()
            active_n += int(active.sum())
            spending += int(((vals > 0) & active).sum())
            for a, b in zero_runs(vals, active, min_run):
                rows = vals.index[a:b + 1]
                mask[rows] = True
                d = df.loc[rows, ctx.date]
                spans.append((f"{g}: " if ctx.geo else "") + _span(d.iloc[0], d.iloc[-1]))
        if spans and active_n and spending / active_n >= share:
            ch = names.get(col)
            related = [col] + ([c for c in (ch.exposure, ch.reach, ch.frequency) if c and c != col and c in df.columns] if ch else [])
            n = int(mask.sum())
            out[col] = {"channel": ch.name if ch else col, "weeks": n, "runs": len(spans), "spans": spans, "mask": mask,
                        "related": related, "spend_share": spending / active_n,
                        "note": (f"{ch.name if ch else col}: {n} zero week{'s' if n != 1 else ''} in {len(spans)} run{'s' if len(spans) != 1 else ''} "
                                 f"({', '.join(spans[:4])}{', …' if len(spans) > 4 else ''}) in a channel that otherwise spends every week")}
    return out


def _dates(df: pd.DataFrame, date: str, mask: pd.Series) -> list[str]:
    return sorted({str(x.date()) for x in df.loc[mask, date]})


def label_gaps(ctx: Ctx, answers: Answers) -> StepResult:
    df = ctx.df.sort_values(ctx.keys).reset_index(drop=True)
    cal = _calendar(df[ctx.date], ctx.freq)
    if ctx.geo:
        full = pd.MultiIndex.from_product([cal, sorted(df[ctx.geo].unique())], names=ctx.keys).to_frame(index=False)
    else:
        full = pd.DataFrame({ctx.date: cal})
    before = len(df)
    df = full.merge(df, on=ctx.keys, how="left", indicator=True)
    df = df.sort_values(ctx.keys).reset_index(drop=True)
    inserted = df.pop("_merge").eq("left_only")
    missing_weeks = _dates(df, ctx.date, inserted)
    ctx.df = df

    value_cols = ctx.value_cols
    decisions: list[Decision] = []
    warnings: list[str] = []
    notes: list[str] = []
    trimmed: list[str] = []
    kpi = ctx.kpi_col

    def keep_rows(keep: pd.Series) -> None:
        nonlocal df, inserted
        df = df[keep.to_numpy()].reset_index(drop=True)
        inserted = inserted[keep.to_numpy()].reset_index(drop=True)
        ctx.df = df

    # KPI first: trimming changes which weeks the other columns cover.
    ns, en, mi = _label(df, list(ctx.by_geo()), [kpi])[kpi]
    imputed = pd.Series(False, index=df.index)
    edge = ns | en
    if edge.any():
        weeks = _dates(df, ctx.date, edge)
        d = Decision(
            id="kpi_edge_gaps",
            step=STEP,
            question=(f"'{kpi}' is empty in {len(weeks)} week(s) at the start or end, with nothing on one side to "
                      "interpolate from. Trim those weeks, or carry the nearest value and flag it?"),
            options=["trim", "fill"],
            default="trim",
            context={"weeks": weeks[:50]},
        )
        decisions.append(d)
        choice = answers.ask(d) or "trim"
        if choice == "trim":
            trimmed += weeks
            keep_rows(~df[ctx.date].isin(pd.to_datetime(weeks)))
            notes.append(f"{len(weeks)} KPI week(s) at the start or end trimmed")
        elif choice == "fill":
            for _, idx in ctx.by_geo():
                df.loc[idx, kpi] = df.loc[idx, kpi].ffill().bfill()
            imputed = edge.copy()
            notes.append(f"{int(edge.sum())} KPI week(s) at the start or end carried from the nearest week and flagged")
        else:
            d.answered, d.blocking = False, True
            return StepResult(STEP, NAME, Verdict.NEEDS_INPUT, f"'{choice}' isn't an option; choose trim or fill.", {}, [], decisions)
        imputed = imputed.reindex(df.index, fill_value=False) if choice == "fill" else pd.Series(False, index=df.index)

    _, _, mi = _label(df, list(ctx.by_geo()), [kpi])[kpi]
    if mi.any():
        weeks = _dates(df, ctx.date, mi)
        d = Decision(
            id="kpi_gaps",
            step=STEP,
            question=(f"'{kpi}' is empty in {len(weeks)} week(s) between weeks that have it. Interpolate them and add a "
                      "kpi_imputed flag, or trim to the longest stretch with no gaps?"),
            options=["interpolate", "trim"],
            default="interpolate",
            blocking=True,
            context={"weeks": weeks[:50]},
        )
        decisions.append(d)
        choice = answers.ask(d)
        if choice is None:
            return StepResult(STEP, NAME, Verdict.NEEDS_INPUT, f"'{kpi}' has {len(weeks)} empty week(s); choose how to handle them.",
                              {"missing_weeks": missing_weeks}, warnings, decisions)
        if choice == "trim":
            present = df.groupby(ctx.date)[kpi].apply(lambda x: x.notna().all()).sort_index()
            keep_dates = present.index[_longest_run(present)]
            gone = _dates(df, ctx.date, ~df[ctx.date].isin(keep_dates))
            trimmed += gone
            keep = df[ctx.date].isin(keep_dates)
            imputed = imputed[keep.to_numpy()].reset_index(drop=True)
            keep_rows(keep)
            notes.append(f"trimmed {len(gone)} week(s); {df[ctx.date].min().date()} to {df[ctx.date].max().date()} kept")
        elif choice == "interpolate":
            for _, idx in ctx.by_geo():
                df.loc[idx, kpi] = df.loc[idx, kpi].interpolate(limit_area="inside")
            imputed = imputed | mi
            notes.append(f"{int(mi.sum())} KPI week(s) interpolated and flagged in kpi_imputed")
        else:
            d.answered = False
            return StepResult(STEP, NAME, Verdict.NEEDS_INPUT, f"'{choice}' isn't an option; choose interpolate or trim.", {}, [], decisions)
    if imputed.any():
        ctx.imputed_kpi = imputed
        ctx.add_control("kpi_imputed", imputed.astype(float))
    missing_weeks = [w for w in missing_weeks if w not in set(trimmed)]

    groups = list(ctx.by_geo())
    labels = _label(df, groups, value_cols)

    # Media holes, judged per column.
    share = float(settings()["media_gaps"]["blank_as_zero_share"])
    media = [c for c in value_cols if ctx.roles.get(c) in MEDIA_ROLES]
    plans = {c: media_holes(df, c, labels[c], inserted, share) for c in media}
    with_blanks = {c: p for c, p in plans.items() if p["blank"].any()}
    absent = any(p["absent"].any() for p in plans.values())
    actions: dict[str, str] = {c: ("zero" if p["kind"] == "blank_means_zero" else "interpolate") for c, p in with_blanks.items()}
    if absent:
        actions["missing_weeks"] = "interpolate"
    if actions:
        lines = []
        for c, p in with_blanks.items():
            n = int(p["blank"].sum())
            if p["kind"] == "blank_means_zero":
                lines.append(f"'{c}': blank in {n} week(s) ({p['blank_share']:.0%} of its weeks) and never 0, "
                             "so a blank looks like no spend (suggested: 0)")
            else:
                lines.append(f"'{c}': blank in {n} week(s) of a channel that otherwise records every week, "
                             f"{'including zeros' if p['has_zero'] else 'never 0'} (suggested: interpolate)")
        if absent:
            lines.append(f"{len(missing_weeks)} week(s) absent from the export: spend there is unknown "
                         "(suggested: interpolate; 0 only if the channels were off)")
        d = Decision(
            id="media_gaps",
            step=STEP,
            question="Media cells with no value: " + "; ".join(lines) + ". Read each as no spend (0) or interpolate?",
            options=["zero", "interpolate"],
            default=actions,
            blocking=absent,
            context={"columns": {c: {"blank_weeks": int(p["blank"].sum()), "kind": p["kind"],
                                     "blank_share": round(p["blank_share"], 3), "has_zero": p["has_zero"]}
                                 for c, p in with_blanks.items()},
                     "missing_weeks": missing_weeks},
        )
        decisions.append(d)
        chosen = answers.ask(d)
        if chosen is None:
            if absent:
                return StepResult(STEP, NAME, Verdict.NEEDS_INPUT,
                                  f"{len(missing_weeks)} week(s) were absent from the export; say how to read media spend there.",
                                  {"missing_weeks": missing_weeks}, warnings, decisions)
            chosen = {}
        if isinstance(chosen, str):
            chosen = {k: chosen for k in actions}
        bad = {k: v for k, v in chosen.items() if k not in actions or v not in ("zero", "interpolate")}
        if bad:
            d.answered, d.blocking = False, True
            return StepResult(STEP, NAME, Verdict.NEEDS_INPUT,
                              f"Unknown column or answer in {bad}; answer zero or interpolate for {sorted(actions)}.", {}, [], decisions)
        actions = {**actions, **chosen}

    # Zero runs in always-on channels: a real pause or a data gap?
    flagged = always_on_zeros(df, ctx, labels, inserted, groups)
    gap_zero: dict[str, pd.Series] = {}             # column -> zero cells read as a data gap
    if flagged:
        warnings += [f["note"] for f in flagged.values()]
        d = Decision(
            id="zero_runs",
            step=STEP,
            question=("These channels spend almost every week but have runs of zero weeks. Is each a real pause "
                      "(keep 0) or a gap in the data (interpolate)?"),
            options=["pause", "gap"],
            default={f["channel"]: "pause" for f in flagged.values()},
            context={"channels": {f["channel"]: {"column": c, "weeks": f["weeks"], "runs": f["runs"], "spans": f["spans"],
                                                 "spend_share": round(f["spend_share"], 3)} for c, f in flagged.items()}},
        )
        decisions.append(d)
        chosen = answers.ask(d) or {}
        if isinstance(chosen, str):
            chosen = {f["channel"]: chosen for f in flagged.values()}
        bad = {k: v for k, v in chosen.items() if k not in d.default or v not in ("pause", "gap")} if isinstance(chosen, dict) else chosen
        if bad:
            d.answered, d.blocking = False, True
            return StepResult(STEP, NAME, Verdict.NEEDS_INPUT,
                              f"Unknown channel or answer in {bad}; answer pause or gap for {sorted(d.default)}.", {}, [], decisions)
        picks = {**d.default, **chosen}
        lines = []
        for f in flagged.values():
            gap = picks[f["channel"]] == "gap"
            lines.append(f"{f['channel']}: {f['weeks']} zero weeks {'read as a data gap, interpolated' if gap else 'kept as a real pause'}")
            if gap:
                for c in f["related"]:
                    gap_zero[c] = gap_zero.get(c, pd.Series(False, index=df.index)) | (f["mask"] & (df[c] == 0))
        answers.describe(d, lines)

    report = []
    imputed_media: dict[str, int] = {}
    for col in value_cols:
        ns, en, mi = labels[col]
        if col == kpi:
            report.append({"column": col, "role": "kpi", "not_started": 0, "ended": 0,
                           "missing": int(imputed.sum()), "zeros": None, "media": False,
                           "fill": "interpolated, flagged" if imputed.any() else "trimmed" if trimmed else "none needed"})
            continue
        role = ctx.roles.get(col)
        n_gaps = int((ns | en | mi).sum())
        gz = gap_zero.get(col)
        # Real zeros: the export's own zeros in a media column; never a filled cell or a zero read as a gap.
        zeros = int(((df[col] == 0) & ~(gz if gz is not None else False)).sum()) if role in MEDIA_ROLES else None
        if role in MEDIA_ROLES:
            p = plans[col]
            fill = ["0 outside its active weeks"] if (ns | en).any() else []
            df.loc[ns | en, col] = 0.0
            blank_act = actions.get(col, "interpolate")
            week_act = actions.get("missing_weeks", "interpolate")
            to_zero = (p["blank"] & (blank_act == "zero")) | (p["absent"] & (week_act == "zero"))
            df.loc[to_zero, col] = 0.0
            if p["blank"].any():
                fill.append(f"{int(p['blank'].sum())} blank(s) read as 0" if blank_act == "zero" else f"{int(p['blank'].sum())} blank(s) interpolated")
            if p["absent"].any():
                fill.append(f"absent weeks {'read as 0' if week_act == 'zero' else 'interpolated'}")
            if gz is not None and gz.any():
                df.loc[gz, col] = float("nan")
                fill.append(f"{int(gz.sum())} zero week(s) read as a data gap, interpolated")
            to_interp = (mi & ~to_zero) | (gz if gz is not None else False)
            if to_interp.any():
                for _, idx in groups:
                    df.loc[idx, col] = df.loc[idx, col].interpolate(limit_area="inside")
                imputed_media[col] = int(to_interp.sum())
            if gz is not None:
                edge_left = gz & df[col].isna()      # a run at the start or end: nothing to interpolate from
                if edge_left.any():
                    df.loc[edge_left, col] = 0.0
                    fill.append(f"{int(edge_left.sum())} at the edge kept as 0 (nothing on one side to interpolate from)")
            fill = "; ".join(fill)
        elif is_binary(df[col]):
            df[col] = df[col].fillna(0.0)
            fill = "0 where missing (flag off)"
        else:
            for _, idx in groups:
                df.loc[idx, col] = df.loc[idx, col].interpolate(limit_area="inside").ffill().bfill()
            fill = "interpolated, carried to the edges"
        if df[col].isna().any():
            n = int(df[col].isna().sum())
            df[col] = df[col].fillna(0.0 if role in MEDIA_ROLES else df[col].median())
            warnings.append(f"'{col}' is empty for a whole region; filled {n} cell(s) with {'0' if role in MEDIA_ROLES else 'its median'}.")
        report.append({"column": col, "role": role, "not_started": int(ns.sum()), "ended": int(en.sum()),
                       "missing": int(mi.sum()), "zeros": zeros, "media": role in MEDIA_ROLES,
                       "fill": fill if (n_gaps or (gz is not None and gz.any())) else "none needed"})
    ctx.df = df

    if missing_weeks:
        warnings.append(f"{len(missing_weeks)} week(s) were absent from the export and added, e.g. {missing_weeks[:3]}.")
    if imputed_media:
        notes.append(f"{sum(imputed_media.values())} media cell(s) interpolated in {', '.join(imputed_media)}")
    labelled = sum(r["not_started"] + r["ended"] + r["missing"] for r in report)
    parts = [f"{len(missing_weeks)} missing week(s) added" if missing_weeks else "no missing weeks",
             f"{labelled:,} empty cell(s) labelled and filled",
             f"{sum(r['zeros'] or 0 for r in report):,} zeros kept as real zeros in media columns"]
    summary = "; ".join(parts + notes) + "."
    details = {"columns": report, "missing_weeks": missing_weeks, "rows_added": int(inserted.sum()),
               "rows_before": before, "rows_after": len(df), "trimmed_weeks": sorted(set(trimmed)),
               "imputed_media": imputed_media,
               "real_zeros": {r["column"]: r["zeros"] for r in report if r.get("media")},
               "zero_runs": {f["channel"]: {"column": c, "weeks": f["weeks"], "runs": f["runs"], "spans": f["spans"]}
                             for c, f in flagged.items()}}
    return StepResult(STEP, NAME, Verdict.WARN if warnings else Verdict.PASS, summary, details, warnings, decisions)
